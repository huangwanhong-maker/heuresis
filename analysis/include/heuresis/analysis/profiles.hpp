// Comparing algorithms over a set of problems: performance and data profiles, and a permutation test.
//
// Given the cost t_{p,s} of solver s on problem p (evaluations to a convergence test; +∞ when it failed):
// - the performance profile of s is ρ_s(τ) = |{p : t_{p,s} ≤ τ min_σ t_{p,σ}}| / |P|, the fraction of problems on which s
//   is within a factor τ of the best solver; ρ_s(1) is how often it is best, ρ_s(∞) how often it succeeds (Dolan and Moré
//   2002, "Benchmarking optimization software with performance profiles", Math. Program. 91, 201–213);
// - the data profile is d_s(κ) = |{p : t_{p,s}/(n_p + 1) ≤ κ}| / |P|, the fraction solved within κ simplex gradients'
//   worth of evaluations, n_p the problem's dimension (Moré and Wild 2009, "Benchmarking derivative-free optimization
//   algorithms", SIAM J. Optim. 20, 172–191), whose convergence test is f(x₀) − f(x) ≥ (1 − τ)(f(x₀) − f_L), f_L the
//   least value any solver found.
// Two samples of results (final values, runtimes) are compared by a permutation test of the difference of their means:
// the labels are permuted B times from keyed streams, and the p-value (b + 1)/(B + 1) of tyche's testing is exact and
// never zero. A rank test (Mann–Whitney) belongs to tyche and is not written here.

#pragma once

#include <heuresis/analysis/runtimes.hpp>
#include <heuresis/search/draws.hpp>

#include <tyche/testing/permutation.hpp>

#include <cmath>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

namespace heuresis::analysis {

// cost[p][s]: problems by rows, solvers by columns. Returns profile[s][k] = ρ_s(taus[k]).
inline std::vector<std::vector<double>> performance_profile(const std::vector<std::vector<double>>& cost, std::span<const double> taus) {
  if (cost.empty()) return {};
  const std::size_t S = cost[0].size();
  std::vector<std::vector<double>> rho(S, std::vector<double>(taus.size(), 0.0));
  for (const auto& row : cost) {
    if (row.size() != S) throw std::invalid_argument("heuresis::analysis::performance_profile: one cost per solver on every problem");
    double best = std::numeric_limits<double>::infinity();
    for (double t : row) best = std::min(best, t);
    for (std::size_t s = 0; s < S; s++)
      for (std::size_t k = 0; k < taus.size(); k++)
        if (std::isfinite(row[s]) && row[s] <= taus[k] * best) rho[s][k] += 1.0 / static_cast<double>(cost.size());
  }
  return rho;
}

// cost[p][s] and the problems' dimensions; returns profile[s][k] = d_s(kappas[k]).
inline std::vector<std::vector<double>> data_profile(const std::vector<std::vector<double>>& cost, std::span<const std::size_t> dimensions, std::span<const double> kappas) {
  if (cost.size() != dimensions.size()) throw std::invalid_argument("heuresis::analysis::data_profile: one dimension per problem");
  if (cost.empty()) return {};
  const std::size_t S = cost[0].size();
  std::vector<std::vector<double>> d(S, std::vector<double>(kappas.size(), 0.0));
  for (std::size_t p = 0; p < cost.size(); p++)
    for (std::size_t s = 0; s < S; s++)
      for (std::size_t k = 0; k < kappas.size(); k++)
        if (cost[p][s] / static_cast<double>(dimensions[p] + 1) <= kappas[k]) d[s][k] += 1.0 / static_cast<double>(cost.size());
  return d;
}

// The evaluations at which a run first met Moré and Wild's test f(x₀) − f ≥ (1 − τ)(f(x₀) − f_L).
inline std::optional<double> evaluations_to_reduce(const Run& r, double f0, double fL, double tau) {
  return runtime(r, f0 - (1 - tau) * (f0 - fL));
}

struct Comparison {
  double difference = 0;  // mean(a) − mean(b)
  double p_value = 1;     // two-sided
  std::size_t permutations = 0;
};

inline Comparison permutation_test(std::span<const double> a, std::span<const double> b, std::size_t permutations, const heuresis::search::Draws& draws) {
  if (a.empty() || b.empty() || permutations == 0) throw std::invalid_argument("heuresis::analysis::permutation_test: two nonempty samples and at least one permutation");
  std::vector<double> all(a.begin(), a.end());
  all.insert(all.end(), b.begin(), b.end());
  const std::size_t na = a.size(), n = all.size();
  auto difference = [&](const std::vector<double>& v) {
    double sa = 0, sb = 0;
    for (std::size_t i = 0; i < n; i++) (i < na ? sa : sb) += v[i];
    return sa / static_cast<double>(na) - sb / static_cast<double>(n - na);
  };
  Comparison c;
  c.difference = difference(all);
  c.permutations = permutations;
  const double observed = std::fabs(c.difference), slack = 1e-12 * (1 + observed);
  std::size_t extreme = 0;
  for (std::size_t k = 0; k < permutations; k++) {
    calx::random::Stream s = draws(heuresis::search::tag("analysis.permutation.shuffle"), 0, k);
    std::vector<double> v = all;
    for (std::size_t i = n; i > 1; i--) std::swap(v[i - 1], v[s.below(i)]);
    if (std::fabs(difference(v)) >= observed - slack) extreme++;
  }
  c.p_value = tyche::testing::permutation_p_value(extreme, permutations);
  return c;
}

}  // namespace heuresis::analysis
