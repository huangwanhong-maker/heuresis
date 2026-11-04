// Niching: keeping several optima in one population.
//
// - Fitness sharing (Goldberg and Richardson 1987, "Genetic algorithms with sharing for multimodal function
//   optimization", ICGA 2, 41–49): each individual's fitness, to be maximised and nonnegative, is divided by its
//   niche count mᵢ = Σⱼ sh(dᵢⱼ), sh(d) = 1 − (d/σ)^α for d < σ and 0 beyond. Individuals then share the payoff of
//   their peak; at equilibrium, when the peaks are farther apart than σ and the individuals of a peak coincide,
//   the shared fitnesses are equal, so a peak of fitness fₖ holds a share of the population proportional to fₖ
//   (their §3, the analogy with animals sharing a resource).
// - Clearing (Pétrowski 1996, "A clearing procedure as a niching method for genetic algorithms", IEEE ICEC,
//   798–803): taking individuals best first, each that is not yet cleared becomes a niche's dominant and keeps
//   its place; of the others within σ of it, the best κ − 1 keep theirs and the rest are cleared. Each niche keeps
//   exactly min(κ, its size) winners, the best of it.

#pragma once

#include <heuresis/search/evaluation.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

namespace heuresis::diversity {

using Distance = std::function<double(std::size_t, std::size_t)>;

inline double sharing_function(double d, double sigma, double alpha = 1) { return d < sigma ? 1 - std::pow(d / sigma, alpha) : 0.0; }

// The shared fitnesses of nonnegative fitnesses (to be maximised).
inline std::vector<double> shared_fitness(std::span<const double> fitness, const Distance& distance, double sigma, double alpha = 1) {
  if (!(sigma > 0)) throw std::invalid_argument("heuresis::diversity::shared_fitness: σ must be positive");
  const std::size_t n = fitness.size();
  std::vector<double> out(n);
  for (std::size_t i = 0; i < n; i++) {
    if (fitness[i] < 0) throw std::invalid_argument("heuresis::diversity::shared_fitness: a negative fitness");
    double m = 0;
    for (std::size_t j = 0; j < n; j++) m += sharing_function(distance(i, j), sigma, alpha);
    out[i] = fitness[i] / m;
  }
  return out;
}

// Which individuals survive clearing, given values to be minimised (ties to the earlier index).
inline std::vector<char> clearing(std::span<const double> values, const Distance& distance, double sigma, std::size_t kappa) {
  if (kappa == 0) throw std::invalid_argument("heuresis::diversity::clearing: κ must be at least 1");
  const std::size_t n = values.size();
  std::vector<std::size_t> o(n);
  std::iota(o.begin(), o.end(), std::size_t{0});
  std::stable_sort(o.begin(), o.end(), [&](std::size_t a, std::size_t b) { return values[a] < values[b]; });
  std::vector<char> cleared(n, 0), winner(n, 0);
  for (std::size_t a = 0; a < n; a++) {
    const std::size_t i = o[a];
    if (cleared[i]) continue;
    winner[i] = 1;
    std::size_t winners = 1;
    for (std::size_t b = a + 1; b < n; b++) {
      const std::size_t j = o[b];
      if (cleared[j] || winner[j] || distance(i, j) >= sigma) continue;
      if (winners < kappa) { winner[j] = 1; winners++; }
      else cleared[j] = 1;
    }
  }
  return winner;
}

}  // namespace heuresis::diversity
