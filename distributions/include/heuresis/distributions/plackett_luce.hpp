// The Plackett–Luce law on permutations, and its estimation.
//
// Items i = 0..n−1 carry weights wᵢ > 0. A ranking π (π(0) the first) is drawn by choosing the first item with
// probability proportional to its weight, the next among the rest likewise, and so on:
//
//   P_w(π) = Π_{k<n} w_{π(k)} / Σ_{j≥k} w_{π(j)}   (Luce 1959; Plackett 1975, "The analysis of permutations", Appl.
//   Statist. 24, 193–202).
//
// Equivalently, π orders the items by decreasing log wᵢ + Gᵢ with Gᵢ independent standard Gumbel variables (Yellott
// 1977, "The relationship between Luce's choice axiom, Thurstone's theory of comparative judgment, and the double
// exponential distribution", J. Math. Psychol. 15, 109–144); the checks hold the two samplers to one law. As a
// search distribution on Sₙ it is the model of Ceberio, Mendiburu and Lozano (2013, "The Plackett–Luce ranking model
// on permutation-based optimization problems", IEEE CEC, 494–501): the weights of the best permutations are fitted
// by maximum likelihood, by Hunter's minorise–maximise iteration (Hunter 2004, "MM algorithms for generalized
// Bradley–Terry models", Ann. Statist. 32, 384–406),
//
//   wᵢ ← cᵢ / Σ_j ω_j Σ_{s : i ∈ R_js, s < n−1} 1/Σ_{t∈R_js} w_t,
//
// R_js the items ranked s-th or later in ranking j, ω_j its weight, cᵢ = Σ_j ω_j [i is not last in j]; each
// iteration increases the likelihood, which the checks verify. The weights are normalised to sum 1 (the law is
// invariant under scaling them). The law has collapsed when its most probable ranking, the items by decreasing
// weight, has probability at least 1 − ε.

#pragma once

#include <heuresis/distributions/family.hpp>
#include <heuresis/variation/permutation.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace heuresis::distributions {

using heuresis::variation::Permutation;

struct PlackettLuce {
  using Point = Permutation;
  using Parameters = std::vector<double>;  // positive weights

  Point sample(const Parameters& w, calx::random::Stream& s) const {
    const std::size_t n = w.size();
    std::vector<std::uint32_t> rest(n);
    std::iota(rest.begin(), rest.end(), 0u);
    double total = 0;
    for (double v : w) total += v;
    Permutation p;
    p.reserve(n);
    while (!rest.empty()) {
      double u = s.uniform() * total, c = 0;
      std::size_t k = rest.size() - 1;
      for (std::size_t j = 0; j + 1 < rest.size(); j++)
        if (u < (c += w[rest[j]])) { k = j; break; }
      p.push_back(rest[k]);
      total -= w[rest[k]];
      rest.erase(rest.begin() + static_cast<std::ptrdiff_t>(k));
    }
    return p;
  }
  // The Gumbel sampler: the items by decreasing log wᵢ + Gᵢ, G = −log(−log U).
  Point sample_by_gumbel(const Parameters& w, calx::random::Stream& s) const {
    const std::size_t n = w.size();
    std::vector<double> key(n);
    for (std::size_t i = 0; i < n; i++) key[i] = std::log(w[i]) - std::log(-std::log(s.uniform()));
    Permutation p = heuresis::variation::identity_permutation(n);
    std::sort(p.begin(), p.end(), [&](std::uint32_t a, std::uint32_t b) { return key[a] > key[b]; });
    return p;
  }
  double log_density(const Parameters& w, const Point& p) const {
    double rest = 0;
    for (auto i : p) rest += w[i];
    double l = 0;
    for (std::size_t k = 0; k + 1 < p.size(); k++) {
      l += std::log(w[p[k]]) - std::log(rest);
      rest -= w[p[k]];
    }
    return l;
  }
  static Parameters uniform(std::size_t n) { return Parameters(n, 1.0 / static_cast<double>(n)); }
};

// Iterations of Hunter's MM from `w`, for rankings with weights `omega` (empty: equal); returns normalised weights.
// An item never ranked before last would go to zero; `floor` keeps every weight at least that fraction of 1/n.
inline std::vector<double> plackett_luce_fit(std::span<const Permutation> rankings, std::span<const double> omega, std::vector<double> w, int iterations, double floor = 0) {
  const std::size_t n = w.size();
  for (int it = 0; it < iterations; it++) {
    std::vector<double> c(n, 0.0), d(n, 0.0);
    for (std::size_t j = 0; j < rankings.size(); j++) {
      const double om = omega.empty() ? 1.0 : omega[j];
      if (om == 0) continue;
      const Permutation& r = rankings[j];
      // Suffix sums of the weights: S_s = Σ_{t≥s} w_{r(t)}.
      std::vector<double> S(n + 1, 0.0);
      for (std::size_t s = n; s-- > 0;) S[s] = S[s + 1] + w[r[s]];
      double acc = 0;  // Σ_{s ≤ position} 1/S_s over the stages s < n − 1
      for (std::size_t s = 0; s < n; s++) {
        if (s + 1 < n) acc += 1 / S[s];
        d[r[s]] += om * acc;
        if (s + 1 < n) c[r[s]] += om;
      }
    }
    double total = 0;
    for (std::size_t i = 0; i < n; i++) total += (w[i] = d[i] > 0 ? c[i] / d[i] : w[i]);
    for (auto& v : w) v /= total;
    if (floor > 0) {
      for (auto& v : w) v = std::max(v, floor / static_cast<double>(n));
      total = 0;
      for (double v : w) total += v;
      for (auto& v : w) v /= total;
    }
  }
  return w;
}

// The estimation-of-distribution rule: fit the μ best by MM from the current weights, then smooth.
// The probability of the most probable ranking, Π_k w_(k)/Σ_{j≥k} w_(j) with the weights in decreasing order; the law
// has collapsed when it is at least 1 − ε.
inline double mode_probability(const std::vector<double>& w) {
  std::vector<double> v(w);
  std::sort(v.begin(), v.end(), std::greater<double>());
  double rest = 0, p = 1;
  for (double x : v) rest += x;
  for (double x : v) {
    p *= x / rest;
    rest -= x;
  }
  return p;
}
inline bool collapsed(const PlackettLuce&, const std::vector<double>& w, double epsilon) { return mode_probability(w) >= 1 - epsilon; }

struct PlackettLuceRule {
  std::size_t mu;
  int iterations = 20;
  double smoothing = 0.5;  // w ← α·fit + (1 − α)·w
  double floor = 0.01;
  std::vector<double> operator()(const PlackettLuce&, const std::vector<double>& w, std::span<const Permutation> ranked, std::span<const Evaluation>) const {
    if (mu == 0 || mu > ranked.size()) throw std::invalid_argument("heuresis::distributions::PlackettLuceRule: 0 < μ ≤ λ");
    const std::vector<double> fit = plackett_luce_fit(ranked.subspan(0, mu), {}, w, iterations, floor);
    std::vector<double> q(w.size());
    double total = 0;
    for (std::size_t i = 0; i < w.size(); i++) total += (q[i] = smoothing * fit[i] + (1 - smoothing) * w[i]);
    for (auto& v : q) v /= total;
    return q;
  }
};

}  // namespace heuresis::distributions
