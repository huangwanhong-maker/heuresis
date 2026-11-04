// Directions in objective space: Das and Dennis's simplex lattice, and the scalarisations of decomposition.
//
// - The simplex lattice of p divisions in m objectives is the set of weights w = k/p with k ∈ ℕᵐ, Σ kᵢ = p; it has
//   C(p + m − 1, m − 1) points spread evenly over the standard simplex (Das and Dennis 1998, "Normal-boundary
//   intersection", SIAM J. Optim. 8, 631–657, §3). NSGA-III takes them as reference directions, MOEA/D as the
//   weights of its subproblems.
// - A decomposition turns the front into a family of single-objective problems, one per weight (Zhang and Li 2007,
//   "MOEA/D: a multiobjective evolutionary algorithm based on decomposition", IEEE Trans. Evol. Comput. 11,
//   712–731, §II):
//     Tchebycheff:   g(f | w, z*) = maxᵢ wᵢ |fᵢ − zᵢ*|, with a zero weight replaced by 10⁻⁶;
//     PBI:           g(f | w, z*) = d₁ + θ d₂, d₁ = (f − z*)·w/‖w‖, d₂ = ‖f − z* − d₁ w/‖w‖‖, θ = 5 by default;
//     weighted sum:  g(f | w) = Σ wᵢ fᵢ, which reaches only the convex part of a front.
//   Every Pareto optimal point minimises the Tchebycheff function of some weight (Miettinen 1999, Nonlinear
//   Multiobjective Optimization, Theorem 3.4.5), which is why it is the default.

#pragma once

#include <heuresis/multiobjective/dominance.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <span>
#include <stdexcept>
#include <vector>

namespace heuresis::multiobjective {

// The simplex lattice of p divisions in m objectives, in lexicographic order of k.
inline std::vector<Point> das_dennis(std::size_t m, std::size_t p) {
  if (m == 0 || p == 0) throw std::invalid_argument("heuresis::multiobjective::das_dennis: m and p must be positive");
  std::vector<Point> out;
  std::vector<std::size_t> k(m, 0);
  std::function<void(std::size_t, std::size_t)> fill = [&](std::size_t i, std::size_t left) {
    if (i == m - 1) {
      k[i] = left;
      Point w(m);
      for (std::size_t j = 0; j < m; j++) w[j] = static_cast<double>(k[j]) / static_cast<double>(p);
      out.push_back(std::move(w));
      return;
    }
    for (std::size_t v = 0; v <= left; v++) {
      k[i] = v;
      fill(i + 1, left - v);
    }
  };
  fill(0, p);
  return out;
}

enum class Scalarisation : std::uint8_t { tchebycheff, pbi, weighted_sum };

inline double scalarise(std::span<const double> f, std::span<const double> w, std::span<const double> ideal, Scalarisation how, double theta = 5) {
  const std::size_t m = f.size();
  switch (how) {
    case Scalarisation::tchebycheff: {
      double g = -INFINITY;
      for (std::size_t i = 0; i < m; i++) g = std::max(g, std::max(w[i], 1e-6) * std::fabs(f[i] - ideal[i]));
      return g;
    }
    case Scalarisation::pbi: {
      double norm = 0;
      for (double v : w) norm += v * v;
      norm = std::sqrt(norm);
      double d1 = 0;
      for (std::size_t i = 0; i < m; i++) d1 += (f[i] - ideal[i]) * w[i] / norm;
      double d2 = 0;
      for (std::size_t i = 0; i < m; i++) {
        const double e = f[i] - ideal[i] - d1 * w[i] / norm;
        d2 += e * e;
      }
      return d1 + theta * std::sqrt(d2);
    }
    case Scalarisation::weighted_sum: {
      double g = 0;
      for (std::size_t i = 0; i < m; i++) g += w[i] * f[i];
      return g;
    }
  }
  return INFINITY;
}

}  // namespace heuresis::multiobjective
