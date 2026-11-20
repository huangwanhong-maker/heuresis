// Checks of heuresis/sequential/policies, and the comparison the survey found nowhere: on a family of knapsacks whose price
// rises with every grant (so that order matters), a policy learned once from thirty training instances, linear by CMA-ES or a
// program by genetic programming, against a sequence searched for each of thirty held-out instances, both set beside each
// instance's best order found by enumeration. Exact: following a policy decodes a feasible word; a constant score follows the
// index order and a threshold above it grants nothing; no policy and no searched sequence is worth more than the instance's best
// order. Measured: the policies' worth on held-out instances against their training worth (generalisation), and against the
// searched sequences and the best.

#include <heuresis/sequential/policies.hpp>

#include <calx/random/stream.hpp>
#include <calx/verification/check.hpp>

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace heuresis::sequential;
namespace hp = heuresis::programs;

namespace {

struct Knap {
  double room = 0;
  int step = 0;
  std::uint32_t taken = 0;
  double value = 0;
  bool operator==(const Knap&) const = default;
};

// Eight grants, each with a value and a size; the capacity 45% of the sizes; a grant made at step k costs size · 1.12^k.
Instance<Knap> knapsack(std::uint64_t seed) {
  calx::random::Stream r(seed, 0x6b6e6170u, 0, 0);
  const std::size_t n = 8;
  std::vector<double> v(n), w(n);
  double total = 0;
  for (std::size_t i = 0; i < n; i++) w[i] = r.uniform(1, 4), v[i] = r.uniform(1, 10) + 0.8 * w[i], total += w[i];
  Instance<Knap> in;
  in.system.grants = n;
  in.system.initial = {0.45 * total, 0, 0, 0};
  in.system.apply = [v, w](std::size_t g, const Knap& s) -> std::optional<Knap> {
    if (s.taken >> g & 1u) return std::nullopt;
    const double cost = w[g] * std::pow(1.12, s.step);
    if (cost > s.room + 1e-12) return std::nullopt;
    return Knap{s.room - cost, s.step + 1, s.taken | (1u << g), s.value + v[g]};
  };
  in.features = [v, w](std::size_t g, const Knap& s) {
    const double cost = w[g] * std::pow(1.12, s.step);
    return std::vector<double>{1.0, v[g], cost, v[g] / cost, cost / std::max(s.room, 1e-9)};
  };
  in.worth = [](const Knap& s) { return s.value; };
  return in;
}

}  // namespace

int main() {
  calx::verification::Suite s;
  std::vector<Instance<Knap>> train, test;
  for (int k = 0; k < 30; k++) train.push_back(knapsack(100 + k)), test.push_back(knapsack(900 + k));
  {
    const auto& in = test[0];
    const auto d = follow(in.system, in.features, [](const std::vector<double>&) { return 0.0; });
    bool feasible = true;
    Knap x = in.system.initial;
    for (std::size_t g : d.granted) {
      auto y = in.system.apply(g, x);
      feasible = feasible && y.has_value();
      if (y) x = *y;
    }
    s.require("following a policy decodes a feasible word, reaching the state it reports", feasible && x == d.state);
    bool ascending = true;
    for (std::size_t i = 1; i < d.granted.size(); i++) ascending = ascending && d.granted[i] > d.granted[i - 1];
    s.require("a constant score follows the index order", ascending && !d.granted.empty());
    s.require("a threshold above every score grants nothing", follow(in.system, in.features, [](const std::vector<double>&) { return 0.0; }, 1.0).granted.empty());
  }
  // Each test instance's best order, by enumeration of its 40,320 orders.
  std::vector<double> best;
  double best_mean = 0;
  for (const auto& in : test) best.push_back(best_order(in)), best_mean += best.back() / 30;
  // A linear policy learned once from the training instances.
  const auto L = train_linear(train, 5, 600, 7);
  // A program learned once, over the four features (value, cost, their ratio, the share of the room a grant takes).
  hp::Signature sig(3);
  for (auto p : {hp::library::add(), hp::library::subtract(), hp::library::multiply(), hp::library::divide()}) sig.add(p);
  for (const char* name : {"one", "value", "cost", "ratio", "fill"}) sig.add(hp::Variable{name, hp::dimensionless()});
  sig.add(hp::Constant{"c", hp::dimensionless(), 0, true, -1, 1});
  hp::GpOptions o;
  o.population = 60, o.max_depth = 4;
  o.limits.depth = 6;
  const auto P = train_program(train, sig, 1200, 11, o);
  // A sequence searched for each test instance, with 300 evaluations each.
  double seq_mean = 0, lin_mean = 0, prog_mean = 0;
  bool below_best = true;
  for (std::size_t i = 0; i < test.size(); i++) {
    const double q = search_sequence(test[i], 300, 31 + i), l = worth(test[i], linear(L.weights)), p = worth(test[i], program(sig, P.tree));
    below_best = below_best && q <= best[i] + 1e-9 && l <= best[i] + 1e-9 && p <= best[i] + 1e-9;
    seq_mean += q / 30, lin_mean += l / 30, prog_mean += p / 30;
  }
  s.require("no policy and no searched sequence is worth more than an instance's best order", below_best);
  std::printf("best %.4f; sequences searched per instance %.4f (%.1f%%); linear policy %.4f (%.1f%%), trained %.4f; program %.4f (%.1f%%), trained %.4f, %s\n",
              best_mean, seq_mean, 100 * seq_mean / best_mean, lin_mean, 100 * lin_mean / best_mean, L.train, prog_mean, 100 * prog_mean / best_mean, P.train,
              hp::to_string(sig, P.tree).c_str());
  s.require("a sequence searched for each held-out instance reaches nearly its best order (above 95%)", seq_mean > 0.95 * best_mean, std::to_string(seq_mean / best_mean));
  s.require("a linear policy learned once reaches most of the best on instances it never saw (above 90%)", lin_mean > 0.90 * best_mean, std::to_string(lin_mean / best_mean));
  s.require("a program learned once does too (above 90%)", prog_mean > 0.90 * best_mean, std::to_string(prog_mean / best_mean));
  // Where the sequence overtakes the policy: the share of the best a sequence searched with each budget reaches.
  std::vector<double> share;
  for (double budget : {10.0, 30.0, 100.0, 300.0}) {
    double m = 0;
    for (std::size_t i = 0; i < test.size(); i++) m += search_sequence(test[i], budget, 31 + i) / 30;
    share.push_back(m / best_mean);
    std::printf("sequences with %.0f evaluations each: %.1f%% of the best\n", budget, 100 * m / best_mean);
  }
  s.require("a sequence searched with 10 evaluations per instance does worse than the policy learned once", share[0] < lin_mean / best_mean,
            std::to_string(share[0]) + " against " + std::to_string(lin_mean / best_mean));
  s.require("... and with 300 it does better: the policy is worth its training where instances are many and evaluations dear", share[3] > lin_mean / best_mean);
  // Training instances are drawn as test ones: the mean best of the two families differs by sampling alone, so the policy's
  // generalisation is read relative to each family's best.
  double train_best = 0;
  for (const auto& in : train) train_best += best_order(in) / 30;
  s.check("the linear policy generalises: its share of the best on held-out instances against on its training ones", lin_mean / best_mean - L.train / train_best, 0.05);
  return s.done();
}
