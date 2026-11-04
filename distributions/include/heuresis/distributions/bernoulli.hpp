// Products of Bernoulli laws on bit strings, and the algorithms that move them.
//
// P_p(x) = Πᵢ pᵢ^{xᵢ} (1 − pᵢ)^{1 − xᵢ}: an exponential family whose sufficient statistic is x itself and whose
// expectation parameters are p. Its score in p is (xᵢ − pᵢ)/(pᵢ(1 − pᵢ)), its Fisher information diag(1/(pᵢ(1 − pᵢ))),
// so the natural gradient is x − p and every IGO step is linear in p (family.hpp).
//
// Rules, each a function of the generation ranked best first:
// - UMDA: p ← the frequencies of the μ best, the IGO step with δt = 1 and truncation weights (Mühlenbein and Paaß
//   1996, "From recombination of genes to the estimation of distributions I. Binary parameters", PPSN IV, 178–187);
//   clamped to the borders [1/n, 1 − 1/n], without which a frequency that reaches 0 or 1 can never move again (Witt
//   2019, "Upper bounds on the running time of the univariate marginal distribution algorithm on OneMax",
//   Algorithmica 81, 632–667);
// - PBIL: p ← (1 − α) p + α · (the mean of the μ best), the IGO step with δt = α (Baluja 1994, "Population-based
//   incremental learning", Carnegie Mellon University CMU-CS-94-163);
// - the compact GA: two samples, the winner w and the loser l; pᵢ ← pᵢ + (wᵢ − lᵢ)/K on the borders, K the
//   hypothetical population size (Harik, Lobo and Goldberg 1999, "The compact genetic algorithm", IEEE Trans. Evol.
//   Comput. 3, 287–297). It is the IGO step with λ = 2, the signed selection function W = 1 on [0, ½) and −1 on
//   (½, 1] (weights ±½) and δt = 2/K (Ollivier et al. 2017): a step of 1/K along the difference of the two samples.

#pragma once

#include <heuresis/distributions/family.hpp>
#include <heuresis/variation/binary.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace heuresis::distributions {

using heuresis::variation::Bits;

struct ProductBernoulli {
  using Point = Bits;
  using Parameters = std::vector<double>;  // pᵢ = P(xᵢ = 1)

  Point sample(const Parameters& p, calx::random::Stream& s) const {
    Bits x(p.size());
    for (std::size_t i = 0; i < p.size(); i++) x[i] = s.uniform() < p[i] ? 1 : 0;
    return x;
  }
  double log_density(const Parameters& p, const Point& x) const {
    double l = 0;
    for (std::size_t i = 0; i < p.size(); i++) l += x[i] ? std::log(p[i]) : std::log1p(-p[i]);
    return l;
  }
  std::vector<double> statistic(const Point& x) const { return std::vector<double>(x.begin(), x.end()); }
  std::vector<double> expectation(const Parameters& p) const { return p; }
  Parameters from_expectation(std::vector<double> eta) const { return eta; }
  std::vector<double> coordinates(const Parameters& p) const { return p; }
  Parameters from_coordinates(std::vector<double> c) const { return c; }
  std::vector<double> score(const Parameters& p, const Point& x) const {
    std::vector<double> g(p.size());
    for (std::size_t i = 0; i < p.size(); i++) g[i] = (x[i] - p[i]) / (p[i] * (1 - p[i]));
    return g;
  }
  calx::linalg::Matrix fisher(const Parameters& p) const {
    calx::linalg::Matrix I(p.size());
    for (std::size_t i = 0; i < p.size(); i++) I(i, i) = 1 / (p[i] * (1 - p[i]));
    return I;
  }
  static Parameters uniform(std::size_t n) { return Parameters(n, 0.5); }
};

inline void clamp_to_borders(std::vector<double>& p, double lo, double hi) {
  for (auto& v : p) v = std::clamp(v, lo, hi);
}

namespace detail {
inline std::vector<double> mean_of_best(std::span<const Bits> ranked, std::size_t mu, std::size_t n) {
  std::vector<double> m(n, 0.0);
  for (std::size_t k = 0; k < mu; k++)
    for (std::size_t i = 0; i < n; i++) m[i] += ranked[k][i];
  for (auto& v : m) v /= static_cast<double>(mu);
  return m;
}
}  // namespace detail

struct Umda {
  std::size_t mu;
  bool borders = true;
  std::vector<double> operator()(const ProductBernoulli&, const std::vector<double>& p, std::span<const Bits> ranked, std::span<const Evaluation>) const {
    if (mu == 0 || mu > ranked.size()) throw std::invalid_argument("heuresis::distributions::Umda: 0 < μ ≤ λ");
    std::vector<double> q = detail::mean_of_best(ranked, mu, p.size());
    if (borders) clamp_to_borders(q, 1.0 / static_cast<double>(p.size()), 1 - 1.0 / static_cast<double>(p.size()));
    return q;
  }
};

struct Pbil {
  std::size_t mu = 1;
  double rate = 0.1;
  bool borders = true;
  std::vector<double> operator()(const ProductBernoulli&, const std::vector<double>& p, std::span<const Bits> ranked, std::span<const Evaluation>) const {
    if (mu == 0 || mu > ranked.size()) throw std::invalid_argument("heuresis::distributions::Pbil: 0 < μ ≤ λ");
    const std::vector<double> m = detail::mean_of_best(ranked, mu, p.size());
    std::vector<double> q(p.size());
    for (std::size_t i = 0; i < p.size(); i++) q[i] = (1 - rate) * p[i] + rate * m[i];
    if (borders) clamp_to_borders(q, 1.0 / static_cast<double>(p.size()), 1 - 1.0 / static_cast<double>(p.size()));
    return q;
  }
};

// The compact GA's rule; the generation must have two points.
struct Compact {
  double K;
  std::vector<double> operator()(const ProductBernoulli&, const std::vector<double>& p, std::span<const Bits> ranked, std::span<const Evaluation>) const {
    if (ranked.size() != 2) throw std::invalid_argument("heuresis::distributions::Compact: a generation of two");
    const double n = static_cast<double>(p.size());
    std::vector<double> q = p;
    for (std::size_t i = 0; i < p.size(); i++) q[i] += (static_cast<double>(ranked[0][i]) - static_cast<double>(ranked[1][i])) / K;
    clamp_to_borders(q, 1 / n, 1 - 1 / n);
    return q;
  }
};

}  // namespace heuresis::distributions
