// Checks of heuresis analysis against small examples computed by hand: runtimes and anytime values read from
// progress; the expected running time; the distribution of runtimes over runs and targets; Clopper and Pearson's
// interval against its closed forms and published values; quantiles of anytime curves; performance and data
// profiles; the permutation test's p-values; runs on two problems of different optima pooled by their gaps to f*; and
// a runtime read from a real run.

#include <heuresis/analysis/analysis.hpp>
#include <heuresis/benchmarks/benchmarks.hpp>
#include <heuresis/search/search.hpp>

#include <calx/verification/check.hpp>

#include <cmath>

using namespace heuresis::analysis;
using heuresis::search::Draws;
using heuresis::search::Progress;

namespace {
Run make_run(std::vector<std::pair<double, double>> steps, double total) {
  Run r;
  for (auto [e, v] : steps) r.progress.push_back(Progress{e, e, v, 0});
  r.evaluations = r.cost = total;
  return r;
}
}  // namespace

int main() {
  calx::verification::Suite s;
  const double inf = INFINITY;

  // Three runs: improvements (evaluations, value) and the evaluations spent in all.
  const std::vector<Run> runs{make_run({{1, 9}, {4, 5}, {10, 1}}, 100), make_run({{1, 8}, {20, 3}}, 50), make_run({{2, 7}, {30, 0.5}}, 80)};

  s.require("runtime: the first improvement at or below the target", runtime(runs[0], 5) == 4.0 && runtime(runs[0], 1) == 10.0 && !runtime(runs[1], 1).has_value());
  s.require("best_at: the best value after a budget", best_at(runs[0], 3) == 9 && best_at(runs[0], 10) == 1 && std::isinf(best_at(runs[2], 1)));

  {
    const auto e = expected_running_time(runs, 1);
    s.check("ERT to f ≤ 1: (10 + 50 + 30)/2 = 45", e.value - 45, 1e-15);
    s.require("ERT counts its successes", e.successes == 2 && e.runs == 3);
    s.require("ERT without a success is +∞", std::isinf(expected_running_time(runs, 0.1).value));
  }

  {
    const std::vector<double> targets{5, 1}, budgets{1, 4, 10, 20, 30, 1000};
    // Runtimes of the six pairs: run 0: 4, 10; run 1: 20, ∞; run 2: 30, 30.
    const auto F = runtime_distribution(runs, targets, budgets);
    const std::vector<double> expected{0, 1.0 / 6, 2.0 / 6, 3.0 / 6, 5.0 / 6, 5.0 / 6};
    double dev = 0;
    for (std::size_t i = 0; i < F.size(); i++) dev = std::max(dev, std::fabs(F[i] - expected[i]));
    s.check("the empirical distribution of runtimes over (run, target) pairs", dev, 1e-15);
  }

  // Two problems of different optima, f* = 5 and f* = −3, pooled by their gaps: targets are precisions Δf.
  {
    const Run a = make_run({{2, 15}, {8, 6}, {20, 5.001}}, 30), b = make_run({{1, 7}, {5, -2}, {12, -2.95}}, 40);
    const std::vector<Run> gaps{relative_to(a, 5), relative_to(b, -3)}, raw{a, b};
    const std::vector<double> precisions{1, 0.01}, budgets{1, 5, 8, 12, 20, 100};
    // Runtimes of the four pairs: a reaches Δf ≤ 1 at 8 and ≤ 0.01 at 20; b reaches ≤ 1 at 5 and never ≤ 0.01.
    const auto F = runtime_distribution(gaps, precisions, budgets), R = runtime_distribution(raw, precisions, budgets);
    const std::vector<double> expected{0, 0.25, 0.5, 0.5, 0.75, 0.75};
    double dev = 0;
    for (std::size_t i = 0; i < F.size(); i++) dev = std::max(dev, std::fabs(F[i] - expected[i]));
    s.check("runs relative to f*: the pooled distribution of runtimes to precisions 1 and 0.01", dev, 1e-15);
    s.require("… which the values themselves, not shifted, misread (b \"reaches\" both at 5, a neither)", R[1] == 0.5 && R.back() == 0.5);
    s.check("… and ERT to Δf ≤ 1 over both problems: (8 + 5)/2 = 6.5", expected_running_time(gaps, 1).value - 6.5, 1e-15);
  }

  {
    const auto z = clopper_pearson(0, 10);
    s.check("Clopper–Pearson, 0 of 10: upper = 1 − 0.025^{1/10}", z.upper - (1 - std::pow(0.025, 0.1)), 1e-10);
    const auto all = clopper_pearson(10, 10);
    s.check("Clopper–Pearson, 10 of 10: lower = 0.025^{1/10}", all.lower - std::pow(0.025, 0.1), 1e-10);
    const auto h = clopper_pearson(5, 10);
    s.check("Clopper–Pearson, 5 of 10: lower 0.187086", h.lower - 0.1870860, 1e-6);
    s.check("Clopper–Pearson, 5 of 10: upper 0.812914", h.upper - 0.8129140, 1e-6);
  }

  {
    const std::vector<double> budgets{1, 10, 40};
    const auto med = anytime_quantile(runs, budgets, 0.5);
    s.require("the median anytime curve: 9, 7, 1 at 1, 10, 40 evaluations", med[0] == 9 && med[1] == 7 && med[2] == 1);
    const auto q75 = anytime_quantile(runs, std::vector<double>{1}, 0.75);
    s.require("an anytime quantile that reaches a run with no value yet is +∞", std::isinf(q75[0]));
  }

  {
    // Four problems, two solvers.
    const std::vector<std::vector<double>> cost{{1, 2}, {3, 3}, {inf, 4}, {2, 1}};
    const std::vector<double> taus{1, 2, 100};
    const auto rho = performance_profile(cost, taus);
    const std::vector<std::vector<double>> expected{{0.5, 0.75, 0.75}, {0.75, 1, 1}};
    double dev = 0;
    for (int a = 0; a < 2; a++)
      for (int k = 0; k < 3; k++) dev = std::max(dev, std::fabs(rho[a][k] - expected[a][k]));
    s.check("performance profiles of two solvers on four problems", dev, 1e-15);

    const std::vector<std::size_t> dims{1, 2, 1, 3};
    const std::vector<double> kappas{0.5, 1, 2};
    const auto d = data_profile(cost, dims, kappas);
    // t/(n + 1): solver 0: 0.5, 1, ∞, 0.5; solver 1: 1, 1, 2, 0.25.
    const std::vector<std::vector<double>> de{{0.5, 0.75, 0.75}, {0.25, 0.75, 1}};
    double dd = 0;
    for (int a = 0; a < 2; a++)
      for (int k = 0; k < 3; k++) dd = std::max(dd, std::fabs(d[a][k] - de[a][k]));
    s.check("data profiles of two solvers on four problems", dd, 1e-15);
    s.require("Moré and Wild's test: f₀ − f ≥ (1 − τ)(f₀ − f_L) first at 10 evaluations (f₀ = 9, f_L = 0.5, τ = 0.1)", evaluations_to_reduce(runs[0], 9, 0.5, 0.1) == 10.0);
  }

  {
    std::vector<double> a, b;
    for (int i = 1; i <= 10; i++) { a.push_back(i + 5); b.push_back(i); }
    const auto shifted = permutation_test(a, b, 2000, Draws(1));
    s.require("a permutation test tells apart a shift of five (p = " + std::to_string(shifted.p_value) + ")", shifted.difference == 5 && shifted.p_value < 0.01);
    const auto same = permutation_test(b, b, 500, Draws(2));
    s.check("identical samples: p = (B + 1)/(B + 1) = 1", same.p_value - 1, 1e-15);
    s.require("the p-value is never below 1/(B + 1)", shifted.p_value >= 1.0 / 2001);
  }

  {
    using heuresis::variation::Bits;
    heuresis::search::RandomSearch<Bits> rs([](calx::random::Stream& r) { return heuresis::variation::random_bits(8, r); }, Draws(3));
    heuresis::search::Record<Bits> record;
    const auto o = heuresis::search::run(rs, [](const Bits& x) { return heuresis::benchmarks::onemax(x); }, heuresis::search::Budget{100000, INFINITY, 0}, heuresis::search::Serial{}, heuresis::search::unlimited, &record);
    const Run r = run_of(o);
    std::size_t first = 0;
    while (first < record.told.size() && record.told[first].evaluation.value() > 0) first++;
    s.require("a real run's runtime to its target is the position of the first evaluation that reached it", runtime(r, 0) == static_cast<double>(first + 1) && runtime(r, 0) <= o.evaluations);
  }

  return s.done();
}
