// Runtimes: how many evaluations a search needs to reach a target, and how that is summarised over runs.
//
// A run is read from its outcome's progress (search/run.hpp): the evaluations (and cost) at which the best feasible
// value improved, and what the run spent in all. Its runtime to a target f_t is the evaluations at the first
// improvement with value ≤ f_t, or none if it never reached it. Over runs and targets:
//
// - the expected running time to a target, ERT = (Σ over runs of the evaluations spent until success, or in all when
//   unsuccessful) / (the number of successes): the expected evaluations of a strategy that restarts independent runs
//   until one succeeds (Price 1997, "Differential evolution vs. the functions of the 2nd ICEO", IEEE ICEC, 153–157;
//   Auger and Hansen 2005, "Performance evaluation of an advanced local search evolutionary algorithm", IEEE CEC,
//   1777–1784), +∞ without a success;
// - the empirical distribution of runtimes over pairs (run, target): for each budget b, the fraction of pairs whose
//   runtime is at most b, the anytime measure of the COCO platform (Hansen, Auger, Ros, Mersmann, Tušar and Brockhoff
//   2021, "COCO: a platform for comparing continuous optimizers in a black-box setting", Optim. Methods Softw. 36,
//   114–144), a nondecreasing step function whose area is an aggregate over targets;
// - success rates with Clopper and Pearson's exact interval, [B(α/2; k, n − k + 1), B(1 − α/2; k + 1, n − k)] with B
//   the beta quantile from tyche (Clopper and Pearson 1934, "The use of confidence or fiducial limits illustrated in the
//   case of the binomial", Biometrika 26, 404–413);
// - anytime curves: the best value after b evaluations, and its quantiles over runs (tyche's quantiles, type 7).

#pragma once

#include <heuresis/search/run.hpp>

#include <tyche/descriptive/summary.hpp>
#include <tyche/probability/distributions.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

namespace heuresis::analysis {

struct Run {
  std::vector<heuresis::search::Progress> progress;
  double evaluations = 0, cost = 0;
};

template <class G>
Run run_of(const heuresis::search::Outcome<G>& o) {
  return {o.progress, o.evaluations, o.cost};
}

// The evaluations at the first improvement reaching the target.
inline std::optional<double> runtime(const Run& r, double target) {
  for (const auto& p : r.progress)
    if (p.value <= target) return p.evaluations;
  return std::nullopt;
}

// The best value after `budget` evaluations; +∞ before the first feasible evaluation.
inline double best_at(const Run& r, double budget) {
  double best = std::numeric_limits<double>::infinity();
  for (const auto& p : r.progress) {
    if (p.evaluations > budget) break;
    best = p.value;
  }
  return best;
}

struct Ert {
  double value = std::numeric_limits<double>::infinity();
  double evaluations = 0;  // spent over all runs until success or in all
  std::size_t successes = 0, runs = 0;
};

inline Ert expected_running_time(std::span<const Run> runs, double target) {
  Ert e;
  e.runs = runs.size();
  for (const auto& r : runs) {
    const auto t = runtime(r, target);
    if (t) {
      e.successes++;
      e.evaluations += *t;
    } else {
      e.evaluations += r.evaluations;
    }
  }
  if (e.successes) e.value = e.evaluations / static_cast<double>(e.successes);
  return e;
}

// For each budget, the fraction of (run, target) pairs whose runtime is at most the budget.
inline std::vector<double> runtime_distribution(std::span<const Run> runs, std::span<const double> targets, std::span<const double> budgets) {
  std::vector<double> times;
  for (const auto& r : runs)
    for (double t : targets) times.push_back(runtime(r, t).value_or(std::numeric_limits<double>::infinity()));
  std::sort(times.begin(), times.end());
  std::vector<double> out;
  out.reserve(budgets.size());
  const double n = static_cast<double>(times.size());
  for (double b : budgets) out.push_back(n == 0 ? 0 : static_cast<double>(std::upper_bound(times.begin(), times.end(), b) - times.begin()) / n);
  return out;
}

struct Proportion {
  double estimate = 0, lower = 0, upper = 1;
};

// k successes of n, with Clopper and Pearson's exact interval at the given confidence.
inline Proportion clopper_pearson(std::size_t k, std::size_t n, double confidence = 0.95) {
  if (n == 0 || k > n || !(confidence > 0 && confidence < 1)) throw std::invalid_argument("heuresis::analysis::clopper_pearson: 0 ≤ k ≤ n, n ≥ 1, confidence in (0, 1)");
  const double alpha = 1 - confidence, K = static_cast<double>(k), N = static_cast<double>(n);
  Proportion p;
  p.estimate = K / N;
  p.lower = k == 0 ? 0 : tyche::probability::Beta{K, N - K + 1}.quantile(alpha / 2);
  p.upper = k == n ? 1 : tyche::probability::Beta{K + 1, N - K}.quantile(1 - alpha / 2);
  return p;
}

// The q-quantile over runs of the best value after each budget (+∞ counts as larger than every value).
inline std::vector<double> anytime_quantile(std::span<const Run> runs, std::span<const double> budgets, double q) {
  std::vector<double> out;
  for (double b : budgets) {
    std::vector<double> v;
    for (const auto& r : runs) v.push_back(best_at(r, b));
    std::sort(v.begin(), v.end());
    if (std::isinf(v.back())) {  // tyche's quantiles need finite values: read the order statistic directly
      const double h = (static_cast<double>(v.size()) - 1) * q;
      const std::size_t lo = static_cast<std::size_t>(std::floor(h)), hi = static_cast<std::size_t>(std::ceil(h));
      out.push_back(std::isinf(v[hi]) ? v[hi] : v[lo] + (h - static_cast<double>(lo)) * (v[hi] - v[lo]));
    } else {
      out.push_back(tyche::descriptive::quantile_sorted(v, q, 7));
    }
  }
  return out;
}

}  // namespace heuresis::analysis
