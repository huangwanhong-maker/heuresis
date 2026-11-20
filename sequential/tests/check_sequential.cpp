// Checks of heuresis/sequential: orders of grants, where what is to be found is known.
//
// - Independence tested by commutators equals exact commutation found over every reachable state; a price that rises after each
//   sale makes order matter, with a witness; a declared independence that a test refutes is reported as a contradiction.
// - Decoders on 100 random systems whose grants consume and produce, against brute force: every output feasible; the deferring
//   decoder's outputs over all orders are exactly the maximal feasible words; the skipping decoder reaches every maximal word and
//   sometimes stops short of one; the stopping decoder gives feasible prefixes. The stepped decoder cannot wait for a falling
//   price without a separator, and can with one.
// - Six grants on three stocks: 720 orders in 8 traces of 90 words, each as many as morphe counts; a cache and an evolution keyed
//   by trace evaluate each trace once.
// - Each crossover keeps what it claims on 500 pairs of related parents: PPX shared precedences, edge recombination shared edges,
//   cycle crossover and PMX shared positions, OX2 and POS the second parent's order among the items they move, NWOX a segment of
//   the second parent with the first's order around it.
// - Starkweather's effect: on a depletion system an objective of precedence, on setups one of adjacency, by the feature probe
//   and by evolutions with operators that keep each feature.
// - Mallows's laws under Kendall's, Cayley's and Hamming's metrics: probabilities summing to one, samples by χ² against them,
//   expected distances; a law fitted to its own samples recovers its centre and dispersion.
// - Words with fixed letter counts: the laws of spaces; the distance as the least number of adjacent exchanges, by breadth-first
//   search; the unit over every pair; an evolution of a picking sequence finds the brute-force optimum.
// - On 50 random stochastic knapsacks the best sequence, the best adaptive policy (by dynamic programming) and clairvoyance are
//   ordered, and the adaptivity gap is at most 4 (Dean, Goemans and Vondrák 2008).

#include <heuresis/sequential/sequential.hpp>
#include <heuresis/spaces/adapters.hpp>
#include <heuresis/spaces/laws.hpp>

#include <calx/verification/check.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <map>
#include <numeric>
#include <set>
#include <string>
#include <vector>

namespace hq = heuresis::sequential;
namespace hs = heuresis::spaces;
namespace he = heuresis::evolution;
using hq::Word;
using heuresis::variation::Permutation;

namespace {

calx::random::Stream stream(std::uint32_t id) { return calx::random::Stream(20261120, 900 + id, 0, 0); }

// Every state reachable from the initial one, by breadth-first search.
template <class S>
std::vector<S> reachable(const hq::GrantSystem<S>& sys) {
  std::vector<S> seen{sys.initial};
  std::deque<S> queue{sys.initial};
  while (!queue.empty()) {
    const S x = queue.front();
    queue.pop_front();
    for (std::size_t g = 0; g < sys.grants; g++)
      if (auto y = sys.apply(g, x))
        if (std::find(seen.begin(), seen.end(), *y) == seen.end()) seen.push_back(*y), queue.push_back(*y);
  }
  return seen;
}

using Stock = std::array<int, 2>;

struct Consumer {
  std::vector<Stock> need, gain;
  Stock start;
};

hq::GrantSystem<Stock> consumer_system(const Consumer& c) {
  hq::GrantSystem<Stock> sys;
  sys.grants = c.need.size();
  sys.initial = c.start;
  sys.apply = [c](std::size_t g, const Stock& x) -> std::optional<Stock> {
    if (x[0] < c.need[g][0] || x[1] < c.need[g][1]) return std::nullopt;
    return Stock{x[0] - c.need[g][0] + c.gain[g][0], x[1] - c.need[g][1] + c.gain[g][1]};
  };
  return sys;
}

double correlation_free(double) { return 0; }

}  // namespace

int main() {
  calx::verification::Suite s;

  // ---- independence ----
  {
    // Three stocks (one unit of the first, three and five of the others) and what each of four recipients holds; two recipients
    // take from the first stock, one from each of the others.
    hq::GrantSystem<std::vector<int>> sys;
    sys.grants = 4;
    sys.names = {"first to a", "first to b", "second to c", "third to d"};
    sys.initial = {1, 3, 5, 0, 0, 0, 0};
    const std::vector<std::size_t> from = {0, 0, 1, 2};
    sys.apply = [from](std::size_t g, const std::vector<int>& x) -> std::optional<std::vector<int>> {
      if (x[from[g]] < 1) return std::nullopt;
      auto y = x;
      y[from[g]]--, y[3 + g]++;
      return y;
    };
    sys.footprints = {{{"first"}, {"first"}}, {{"first"}, {"first"}}, {{"second"}, {"second"}}, {{"third"}, {"third"}}};
    auto st = stream(1);
    const auto r = hq::independence(sys, st, 200);
    // Exact commutation over every reachable state.
    const auto states = reachable(sys);
    bool agree = true;
    for (std::size_t g = 0; g < 4; g++)
      for (std::size_t h = g + 1; h < 4; h++) {
        bool commute = true;
        for (const auto& x : states) {
          const auto gx = sys.apply(g, x), hx = sys.apply(h, x);
          if (!gx && !hx) continue;
          // Either order, each grant skipped where it is not allowed; both orders granting both, or neither.
          const auto y1 = gx ? *gx : x, y2 = hx ? *hx : x;
          const auto gh = sys.apply(h, y1), hg = sys.apply(g, y2);
          const auto end1 = gh ? *gh : y1, end2 = hg ? *hg : y2;
          commute = commute && end1 == end2 && (gx.has_value() && gh.has_value()) == (hx.has_value() && hg.has_value());
        }
        agree = agree && r.tested.commute(g, h) == commute;
      }
    s.require("commutator tests agree with exact commutation over all " + std::to_string(states.size()) + " reachable states", agree);
    s.require("two recipients of a stock of one fail to commute, though both orders are infeasible; grants from different stocks commute",
              !r.tested.commute(0, 1) && r.tested.commute(0, 2) && r.tested.commute(2, 3) && r.contradictions.empty());

    // A price that rises after each sale.
    hq::GrantSystem<std::array<int, 3>> market;  // units sold, the budgets of two buyers
    market.grants = 2;
    market.names = {"sale to the first buyer", "sale to the second buyer"};
    market.initial = {0, 3, 3};
    market.apply = [](std::size_t g, const std::array<int, 3>& x) -> std::optional<std::array<int, 3>> {
      const int price = 2 + x[0];
      if (x[1 + g] < price) return std::nullopt;
      auto y = x;
      y[0]++, y[1 + g] -= price;
      return y;
    };
    market.footprints = {{{"budget 1"}, {"budget 1"}}, {{"budget 2"}, {"budget 2"}}};  // the price forgotten: wrongly independent
    const auto m = hq::independence(market, st, 50);
    s.require("a price that rises after each sale makes order matter, with a witness", m.order_may_matter() && !m.witnesses.empty(), m.witnesses.empty() ? "" : m.witnesses[0]);
    s.require("a footprint that forgets the price is a contradiction the tests report", !m.contradictions.empty(), m.contradictions.empty() ? "" : m.contradictions[0]);
  }

  // ---- decoders against brute force ----
  {
    auto st = stream(2);
    bool feasible = true, defer_exact = true, skip_covers = true, prefixes = true;
    std::size_t short_of_maximal = 0, systems = 100;
    for (std::size_t t = 0; t < systems; t++) {
      Consumer c;
      const std::size_t k = 5;
      for (std::size_t g = 0; g < k; g++) {
        c.need.push_back({static_cast<int>(st.below(4)), static_cast<int>(st.below(4))});
        c.gain.push_back({static_cast<int>(st.below(3)), static_cast<int>(st.below(3))});
      }
      c.start = {2 + static_cast<int>(st.below(4)), 2 + static_cast<int>(st.below(4))};
      const auto sys = consumer_system(c);
      // Maximal feasible words of distinct grants, by enumeration of every word of distinct grants.
      std::set<Word> maximal;
      std::function<void(Word&, std::vector<char>&, const Stock&)> grow = [&](Word& w, std::vector<char>& used, const Stock& x) {
        bool extended = false;
        for (std::size_t g = 0; g < k; g++)
          if (!used[g])
            if (auto y = sys.apply(g, x)) {
              extended = true;
              used[g] = 1, w.push_back(g);
              grow(w, used, *y);
              used[g] = 0, w.pop_back();
            }
        if (!extended) maximal.insert(w);
      };
      Word w;
      std::vector<char> used(k, 0);
      grow(w, used, sys.initial);
      std::set<Word> deferred, skipped;
      Word order(k);
      std::iota(order.begin(), order.end(), 0);
      do {
        const auto d = hq::serial(sys, order, hq::Unmet::defer), sk = hq::serial(sys, order, hq::Unmet::skip), sp = hq::serial(sys, order, hq::Unmet::stop);
        feasible = feasible && hq::feasible(sys, d.granted) && hq::feasible(sys, sk.granted) && hq::feasible(sys, sp.granted);
        prefixes = prefixes && std::equal(sp.granted.begin(), sp.granted.end(), order.begin());
        deferred.insert(d.granted), skipped.insert(sk.granted);
        if (!maximal.count(sk.granted)) short_of_maximal++;
      } while (std::next_permutation(order.begin(), order.end()));
      defer_exact = defer_exact && deferred == maximal;
      for (const auto& mw : maximal) skip_covers = skip_covers && skipped.count(mw);
    }
    s.require("every decoded allocation is feasible, on 100 random systems", feasible);
    s.require("the deferring decoder's outputs over all orders are exactly the maximal feasible words", defer_exact);
    s.require("the skipping decoder reaches every maximal feasible word", skip_covers);
    s.require("and stops short of a maximal word on some orders (" + std::to_string(short_of_maximal) + " of 12000)", short_of_maximal > 0);
    s.require("the stopping decoder gives prefixes of the word", prefixes);
    s.require("the decoders declare their reach", hq::reach(hq::Unmet::defer).maximal && hq::reach(hq::Unmet::defer).every_maximal && !hq::reach(hq::Unmet::skip).maximal);
  }
  {
    // A price that falls by one each step from 4 to 1; a budget of 6; two items. Buying both at once at 2 leaves 2.
    struct Shop {
      int budget, price;
      bool operator==(const Shop&) const = default;
    };
    hq::GrantSystem<Shop> shop;
    shop.grants = 2;
    shop.initial = {6, 4};
    shop.apply = [](std::size_t, const Shop& x) -> std::optional<Shop> {
      if (x.budget < x.price) return std::nullopt;
      return Shop{x.budget - x.price, x.price};
    };
    shop.advance = [](const Shop& x) { return Shop{x.budget, std::max(1, x.price - 1)}; };
    int best_greedy = -1;
    for (const Word& w : {Word{0, 1}, Word{1, 0}}) {
      const auto t = hq::stepped(shop, w, 4);
      best_greedy = std::max(best_greedy, t.unmet.empty() ? t.state.budget : -1);
    }
    const auto waited = hq::stepped(shop, Word{2, 2, 0, 1}, 4, std::size_t{2});
    s.require("the stepped decoder buys while it can: no order of the grants alone waits for the falling price (leaves " + std::to_string(best_greedy) + ")", best_greedy == 0);
    s.require("with a separator the plan that waits is reached, leaving 2", waited.unmet.empty() && waited.state.budget == 2);
  }

  // ---- traces of orders, and evaluating each once ----
  {
    hq::GrantSystem<std::array<int, 3>> sys;
    sys.grants = 6;
    sys.initial = {2, 2, 2};
    sys.apply = [](std::size_t g, const std::array<int, 3>& x) -> std::optional<std::array<int, 3>> {
      auto y = x;
      y[g / 2] -= static_cast<int>(g % 2) + 1;  // grants 2i and 2i + 1 take 1 and 2 units of stock i
      return y;
    };
    for (std::size_t g = 0; g < 6; g++) {
      const std::string item = "stock " + std::to_string(g / 2);
      sys.footprints.push_back({{item}, {item}});
    }
    const auto I = morphe::formal::traces::independence(sys.footprints);
    const auto red = hq::redundancy_of_orders(I);
    s.require("720 orders of six grants on three stocks fall into 8 traces (" + std::to_string(red.traces) + ")", red.words == 720 && red.traces == 8);
    Word w(6);
    std::iota(w.begin(), w.end(), 0);
    bool counts = true;
    std::map<std::string, std::size_t> per;
    do per[hq::trace_key(w, I)]++;
    while (std::next_permutation(w.begin(), w.end()));
    for (const auto& [k, c] : per) {
      (void)k;
      counts = counts && c == 90;
    }
    std::iota(w.begin(), w.end(), 0);
    counts = counts && morphe::formal::traces::linearisations(morphe::formal::traces::causal_order(w, I)).str() == "90";
    s.require("each trace has 90 words, as morphe's linearisations count", counts);
    hq::TraceCache<double> cache(I);
    std::size_t calls = 0;
    do cache.evaluate(w, [&](const Word&) { calls++; return 0.0; });
    while (std::next_permutation(w.begin(), w.end()));
    s.require("a cache keyed by trace evaluates each of the 8 traces once among 720 orders", calls == 8 && cache.hits() == 712);
    // An evolution over orders of the six grants, keyed by trace.
    auto ops = hq::keyed_by_trace(hq::order_operators(6), I, std::function<Word(const hs::Value&)>([](const hs::Value& v) {
                                    return Word(v.ints.begin(), v.ints.end());
                                  }));
    he::Strategy strategy;
    strategy.mu = 10, strategy.lambda = 10, strategy.generations = 30, strategy.from_archive = 0;
    he::Evolution<hs::Value> e(ops, strategy, {}, 5);
    std::size_t evaluated = 0;
    he::evolve(e, [&](const hs::Value& g) {
      evaluated++;
      const auto end = hq::run(sys, Word(g.ints.begin(), g.ints.end()));
      return heuresis::search::scalar(static_cast<double>((*end)[0]));
    });
    s.require("an evolution keyed by trace evaluates at most the 8 traces (" + std::to_string(evaluated) + ")", evaluated <= 8 && evaluated > 0);
  }

  // ---- what each crossover keeps ----
  {
    auto st = stream(3);
    const std::size_t n = 12;
    bool ppx = true, erx = true, cx = true, pmx = true, ox2 = true, pos = true, nwox = true, perms = true;
    auto in_order_of = [](const Permutation& seq, const Permutation& ref) {  // seq's items appear in ref's relative order
      std::vector<std::size_t> at(ref.size());
      for (std::size_t i = 0; i < ref.size(); i++) at[ref[i]] = i;
      for (std::size_t i = 1; i < seq.size(); i++)
        if (at[seq[i - 1]] > at[seq[i]]) return false;
      return true;
    };
    for (int t = 0; t < 500; t++) {
      const Permutation a = heuresis::variation::random_permutation(n, st);
      Permutation b = a;
      for (int k = 0; k < 3; k++) b = t % 2 ? heuresis::variation::insertion_mutation(b, st) : heuresis::variation::swap_mutation(b, st);
      const auto c1 = hq::precedence_preserving(a, b, st), c2 = hq::edge_recombination(a, b, st);
      const auto [c3, c3b] = heuresis::variation::cycle_crossover(a, b, st);
      const auto [c4, c4b] = heuresis::variation::partially_mapped_crossover(a, b, st);
      const auto c5 = hq::order_based(a, b, st), c6 = hq::position_based(a, b, st), c7 = hq::non_wrapping_order(a, b, st);
      for (const auto* c : {&c1, &c2, &c3, &c4, &c5, &c6, &c7}) perms = perms && heuresis::variation::is_permutation(*c) && c->size() == n;
      ppx = ppx && hq::keeps_shared_precedences(a, b, c1);
      erx = erx && hq::keeps_shared_edges(a, b, c2);
      cx = cx && hq::keeps_shared_positions(a, b, c3);
      pmx = pmx && hq::keeps_shared_positions(a, b, c4);
      Permutation moved5, moved6;
      for (std::size_t i = 0; i < n; i++) {
        if (c5[i] != a[i]) moved5.push_back(c5[i]);
        if (c6[i] != a[i]) moved6.push_back(c6[i]);
      }
      ox2 = ox2 && in_order_of(moved5, b);
      pos = pos && in_order_of(moved6, b);
      bool segment = false;
      for (std::size_t l = 0; l < n && !segment; l++)
        for (std::size_t r = l; r < n && !segment; r++) {
          bool ok = true;
          for (std::size_t i = l; i <= r && ok; i++) ok = c7[i] == b[i];
          Permutation around;
          for (std::size_t i = 0; i < n; i++)
            if (i < l || i > r) around.push_back(c7[i]);
          segment = ok && in_order_of(around, a);
        }
      nwox = nwox && segment;
    }
    s.require("every crossover gives a permutation", perms);
    s.require("PPX keeps every precedence both parents share", ppx);
    s.require("edge recombination keeps every edge both parents share", erx);
    s.require("cycle crossover and PMX keep every position both parents share", cx && pmx);
    s.require("OX2 and POS move items only in the second parent's order", ox2 && pos);
    s.require("NWOX places a segment of the second parent and keeps the first's order around it", nwox);
  }

  // ---- Starkweather's effect ----
  {
    auto st = stream(4);
    const std::size_t n = 40;
    std::vector<double> demand(n), weight(n), px(n), py(n);
    double total = 0;
    for (std::size_t i = 0; i < n; i++) {
      demand[i] = 1 + static_cast<double>(st.below(5)), weight[i] = 1 + static_cast<double>(st.below(10));
      px[i] = st.uniform(), py[i] = st.uniform();
      total += demand[i];
    }
    const double stock = 0.6 * total;
    // Orders filled from a stock that depletes: weighted completion of the filled, a penalty for the unfilled.
    auto depletion = [&](const Permutation& p) {
      double used = 0, cost = 0;
      for (auto i : p) {
        if (used + demand[i] <= stock) used += demand[i], cost += weight[i] * used;
        else cost += weight[i] * 2 * total;
      }
      return cost;
    };
    // Setups between consecutive items: an open path among points.
    auto setups = [&](const Permutation& p) {
      double cost = 0;
      for (std::size_t k = 0; k + 1 < n; k++) cost += std::hypot(px[p[k]] - px[p[k + 1]], py[p[k]] - py[p[k + 1]]);
      return cost;
    };
    const auto probe_d = hq::feature_probe(n, depletion, st), probe_s = hq::feature_probe(n, setups, st);
    char buf[160];
    std::snprintf(buf, sizeof buf, "precedence %.2f, adjacency %.2f, position %.2f", probe_d.precedence, probe_d.adjacency, probe_d.position);
    s.require("the feature probe finds a depletion system's structure in precedence", probe_d.strongest() == hq::Feature::precedence, buf);
    std::snprintf(buf, sizeof buf, "precedence %.2f, adjacency %.2f, position %.2f", probe_s.precedence, probe_s.adjacency, probe_s.position);
    s.require("and a system of setups' structure in adjacency", probe_s.strongest() == hq::Feature::adjacency, buf);
    auto mean_best = [&](const std::function<double(const Permutation&)>& f, hq::Feature feature) {
      double sum = 0;
      for (std::uint64_t seed = 1; seed <= 6; seed++) {
        he::Strategy strategy;
        strategy.mu = 40, strategy.lambda = 40, strategy.generations = 100, strategy.from_archive = 0;
        he::Evolution<hs::Value> e(hq::order_operators(n, {feature}), strategy, {}, seed);
        he::evolve(e, [&](const hs::Value& g) {
          Permutation p(g.ints.begin(), g.ints.end());
          return heuresis::search::scalar(f(p));
        });
        sum += e.state().best->evaluation.objectives[0];
      }
      return sum / 6;
    };
    const double dp = mean_best(depletion, hq::Feature::precedence), da = mean_best(depletion, hq::Feature::adjacency);
    const double sp = mean_best(setups, hq::Feature::precedence), sa = mean_best(setups, hq::Feature::adjacency);
    std::snprintf(buf, sizeof buf, "mean best %.1f against %.1f", dp, da);
    s.require("on depletion, operators that keep precedence beat edge recombination (Starkweather et al. 1991)", dp < da, buf);
    std::snprintf(buf, sizeof buf, "mean best %.3f against %.3f", sa, sp);
    s.require("on setups, edge recombination beats operators that keep precedence", sa < sp, buf);
  }

  // ---- Mallows's laws ----
  {
    auto st = stream(5);
    const std::size_t n = 4;
    Permutation centre = heuresis::variation::random_permutation(n, st);
    std::vector<Permutation> all;
    Permutation p(n);
    std::iota(p.begin(), p.end(), 0u);
    do all.push_back(p);
    while (std::next_permutation(p.begin(), p.end()));
    for (auto metric : {hq::Metric::kendall, hq::Metric::cayley, hq::Metric::hamming}) {
      const hq::Mallows law(centre, 0.5, metric);
      const char* name = metric == hq::Metric::kendall ? "Kendall's" : metric == hq::Metric::cayley ? "Cayley's" : "Hamming's";
      double total = 0, mean = 0;
      for (const auto& q : all) total += law.probability(q), mean += law.probability(q) * law.distance_to_centre(q);
      s.check(std::string("Mallows under ") + name + " metric: probabilities sum to one", total - 1, 1e-12);
      s.check(std::string("Mallows under ") + name + " metric: the expected distance as enumerated", law.expected_distance() - mean, 1e-12);
      std::map<Permutation, std::size_t> count;
      const std::size_t draws = 24000;
      for (std::size_t i = 0; i < draws; i++) count[law.sample(st)]++;
      double chi2 = 0;
      for (const auto& q : all) {
        const double e = law.probability(q) * static_cast<double>(draws);
        chi2 += (static_cast<double>(count[q]) - e) * (static_cast<double>(count[q]) - e) / e;
      }
      s.check(std::string("Mallows under ") + name + " metric: samples by χ² against the law, standardised", (chi2 - 23) / std::sqrt(46.0), 4);
    }
    const Permutation c8 = heuresis::variation::random_permutation(8, st);
    const hq::Mallows truth(c8, 0.6);
    std::vector<Permutation> sample;
    for (int i = 0; i < 5000; i++) sample.push_back(truth.sample(st));
    const auto fit = hq::fit_mallows(sample);
    s.require("a Mallows law fitted to 5000 of its samples recovers its centre", fit.centre() == c8);
    s.check("and its dispersion", fit.phi() - 0.6, 0.03);
  }

  // ---- words with fixed letter counts ----
  {
    auto words = std::make_shared<hq::WordSpace>(std::vector<std::size_t>{3, 2, 4});
    auto st = stream(6);
    const auto laws = hs::check_laws(*words, st);
    s.require("words with letter counts (3, 2, 4) hold every law of spaces (" + std::to_string(laws.applied()) + " applied)", laws.all_hold(), laws.failures());
    hq::WordSpace small({2, 1, 2});
    const auto all = small.elements(1000);
    // The least number of adjacent exchanges, by breadth-first search from each word.
    bool bfs = true;
    double mean = 0;
    for (const auto& u : all) {
      std::map<std::vector<std::int64_t>, int> dist{{u.ints, 0}};
      std::deque<std::vector<std::int64_t>> q{u.ints};
      while (!q.empty()) {
        const auto x = q.front();
        q.pop_front();
        for (std::size_t i = 0; i + 1 < x.size(); i++) {
          auto y = x;
          std::swap(y[i], y[i + 1]);
          if (!dist.count(y)) dist[y] = dist[x] + 1, q.push_back(y);
        }
      }
      for (const auto& v : all) {
        bfs = bfs && small.distance(u, v) == dist[v.ints];
        mean += small.distance(u, v);
      }
    }
    mean /= static_cast<double>(all.size() * all.size());
    s.require("a word distance is the least number of adjacent exchanges, on all " + std::to_string(all.size() * all.size()) + " pairs", bfs);
    s.check("the words' unit is the mean distance over every pair", small.unit() - mean, 1e-12);

    // A picking sequence: three agents pick twice each among six items by additive utilities; maximise the least utility.
    auto picks = std::make_shared<hq::WordSpace>(std::vector<std::size_t>{2, 2, 2});
    std::vector<std::vector<double>> utility(3, std::vector<double>(6));
    for (auto& row : utility)
      for (double& v : row) v = st.uniform();
    auto least = [&](const hs::Value& w) {
      std::vector<char> taken(6, 0);
      std::vector<double> got(3, 0);
      for (auto agent : w.ints) {
        std::size_t best = 6;
        for (std::size_t i = 0; i < 6; i++)
          if (!taken[i] && (best == 6 || utility[static_cast<std::size_t>(agent)][i] > utility[static_cast<std::size_t>(agent)][best])) best = i;
        taken[best] = 1, got[static_cast<std::size_t>(agent)] += utility[static_cast<std::size_t>(agent)][best];
      }
      return *std::min_element(got.begin(), got.end());
    };
    double optimum = 0;
    for (const auto& w : picks->elements(1000)) optimum = std::max(optimum, least(w));
    he::Strategy strategy;
    strategy.mu = 12, strategy.lambda = 12, strategy.generations = 40, strategy.from_archive = 0;
    he::Evolution<hs::Value> e(hs::operators(picks), strategy, {}, 9);
    he::evolve(e, [&](const hs::Value& w) { return heuresis::search::scalar(-least(w)); });
    s.check("an evolution of a picking sequence finds the brute-force optimum of the least utility", -e.state().best->evaluation.objectives[0] - optimum, 1e-12);
  }

  // ---- the adaptivity gap of stochastic knapsacks ----
  {
    auto st = stream(7);
    double worst = 0;
    bool ordered = true;
    for (int t = 0; t < 50; t++) {
      const std::size_t k = 6;
      std::vector<std::array<int, 2>> size(k);
      std::vector<double> value(k);
      for (std::size_t i = 0; i < k; i++) size[i] = {1 + static_cast<int>(st.below(4)), 1 + static_cast<int>(st.below(8))}, value[i] = st.uniform(0.1, 1);
      const int capacity = 6 + static_cast<int>(st.below(6));
      // Scenarios: every combination of outcomes, each of probability 2⁻ᵏ.
      std::vector<std::vector<int>> scenarios;
      for (std::size_t mask = 0; mask < (1u << k); mask++) {
        std::vector<int> sz(k);
        for (std::size_t i = 0; i < k; i++) sz[i] = size[i][(mask >> i) & 1];
        scenarios.push_back(sz);
      }
      // Insert in order until an item overflows; it and the rest count nothing.
      std::function<double(const Word&, const std::vector<int>&)> fill = [&](const Word& w, const std::vector<int>& sz) {
        int used = 0;
        double v = 0;
        for (auto i : w) {
          if (used + sz[i] > capacity) break;
          used += sz[i], v += value[i];
        }
        return v;
      };
      const auto bracket = hq::adaptivity_bracket<std::vector<int>>(k, scenarios, fill);
      // The best adaptive policy, by dynamic programming over the room left and the items left.
      std::map<std::pair<int, unsigned>, double> memo;
      std::function<double(int, unsigned)> policy = [&](int room, unsigned left) -> double {
        auto key = std::make_pair(room, left);
        if (auto it = memo.find(key); it != memo.end()) return it->second;
        double best = 0;
        for (std::size_t i = 0; i < k; i++)
          if (left >> i & 1) {
            double v = 0;
            for (int o = 0; o < 2; o++)
              if (size[i][o] <= room) v += 0.5 * (value[i] + policy(room - size[i][o], left & ~(1u << i)));
            best = std::max(best, v);
          }
        return memo[key] = best;
      };
      const double adaptive = policy(capacity, (1u << k) - 1);
      ordered = ordered && bracket.sequence <= adaptive + 1e-12 && adaptive <= bracket.clairvoyant + 1e-12;
      worst = std::max(worst, adaptive / bracket.sequence);
    }
    s.require("on 50 stochastic knapsacks: best sequence ≤ best adaptive policy ≤ clairvoyance", ordered);
    s.check("and the adaptivity gap is at most 4 (Dean, Goemans and Vondrák 2008): the largest found, less 4, if positive", std::max(0.0, worst - 4), 0);
  }

  // ---- refusals ----
  {
    auto refuses = [&](const std::string& what, const std::function<void()>& f) {
      bool r = false;
      try {
        f();
      } catch (const hs::Refusal&) {
        r = true;
      }
      s.require(what + " is refused", r);
    };
    hq::GrantSystem<int> none;
    none.grants = 2;
    refuses("a grant system without its action", [&] { hq::serial(none, Word{0, 1}); });
    hq::GrantSystem<int> sys;
    sys.grants = 2;
    sys.apply = [](std::size_t, const int& x) -> std::optional<int> { return x + 1; };
    sys.advance = [](const int& x) { return x; };
    refuses("a word with a letter outside the grants", [&] { hq::serial(sys, Word{0, 5}); });
    refuses("a separator among the grants", [&] { hq::stepped(sys, Word{0, 1}, 2, std::size_t{1}); });
    refuses("a Mallows law with a dispersion above one", [] { (void)hq::Mallows(Permutation{0, 1, 2}, 1.5); });
    hq::WordSpace w({1, 1});
    refuses("a word with the wrong letter counts", [&] { w.distance(hs::integer_value({0, 0}), hs::integer_value({0, 1})); });
  }
  (void)correlation_free;
  return s.done();
}
