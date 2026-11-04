// Products of categorical laws on vectors of choices.
//
// Variable i takes one of kᵢ values with probabilities pᵢ = (pᵢ₀, …, pᵢ,kᵢ₋₁). The product is an exponential family
// whose sufficient statistic is the indicator of each value but the last of each variable, and whose expectation
// parameters are those probabilities; its IGO steps are therefore the frequency updates of the Bernoulli case,
// value by value (family.hpp). UMDA's rule fits the frequencies of the μ best and keeps every probability above the
// border 1/((kᵢ − 1) n), the multi-valued borders of Ben Jedidia, Doerr and Krejca (2023, "Estimation-of-
// distribution algorithms for multi-valued decision variables", GECCO, 230–238), which reduce to [1/n, 1 − 1/n]
// for kᵢ = 2. The law has collapsed when every variable takes one value with probability at least 1 − ε.

#pragma once

#include <heuresis/distributions/family.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace heuresis::distributions {

struct ProductCategorical {
  using Point = std::vector<std::uint32_t>;
  using Parameters = std::vector<std::vector<double>>;

  Point sample(const Parameters& p, calx::random::Stream& s) const {
    Point x(p.size());
    for (std::size_t i = 0; i < p.size(); i++) {
      double u = s.uniform(), c = 0;
      std::uint32_t v = static_cast<std::uint32_t>(p[i].size() - 1);
      for (std::size_t j = 0; j + 1 < p[i].size(); j++)
        if (u < (c += p[i][j])) { v = static_cast<std::uint32_t>(j); break; }
      x[i] = v;
    }
    return x;
  }
  double log_density(const Parameters& p, const Point& x) const {
    double l = 0;
    for (std::size_t i = 0; i < p.size(); i++) l += std::log(p[i][x[i]]);
    return l;
  }
  std::vector<double> statistic(const Point& x) const {
    std::vector<double> t;
    for (std::size_t i = 0; i < x.size(); i++)
      for (std::size_t j = 0; j + 1 < sizes[i]; j++) t.push_back(x[i] == j ? 1.0 : 0.0);
    return t;
  }
  std::vector<double> expectation(const Parameters& p) const {
    std::vector<double> e;
    for (const auto& q : p)
      for (std::size_t j = 0; j + 1 < q.size(); j++) e.push_back(q[j]);
    return e;
  }
  Parameters from_expectation(std::vector<double> eta) const {
    Parameters p(sizes.size());
    std::size_t k = 0;
    for (std::size_t i = 0; i < sizes.size(); i++) {
      p[i].resize(sizes[i]);
      double rest = 1;
      for (std::size_t j = 0; j + 1 < sizes[i]; j++) rest -= (p[i][j] = eta[k++]);
      p[i][sizes[i] - 1] = rest;
    }
    return p;
  }
  Parameters uniform() const {
    Parameters p(sizes.size());
    for (std::size_t i = 0; i < sizes.size(); i++) p[i].assign(sizes[i], 1.0 / static_cast<double>(sizes[i]));
    return p;
  }

  std::vector<std::size_t> sizes;  // kᵢ ≥ 2
};

// Each probability raised to at least `lo`, the others scaled so that the variable's law sums to 1.
inline void clamp_categorical(std::vector<double>& q, double lo) {
  std::vector<char> fixed(q.size(), 0);
  for (bool changed = true; changed;) {
    changed = false;
    double free_mass = 0, fixed_mass = 0;
    for (std::size_t j = 0; j < q.size(); j++) (fixed[j] ? fixed_mass : free_mass) += fixed[j] ? lo : q[j];
    const double scale = free_mass > 0 ? (1 - fixed_mass) / free_mass : 0;
    for (std::size_t j = 0; j < q.size(); j++) {
      if (fixed[j]) { q[j] = lo; continue; }
      q[j] *= scale;
      if (q[j] < lo) { fixed[j] = 1; changed = true; }
    }
  }
}

// The law has collapsed: every variable takes one value with probability at least 1 − ε.
inline bool collapsed(const ProductCategorical&, const ProductCategorical::Parameters& p, double epsilon) {
  return std::all_of(p.begin(), p.end(), [epsilon](const std::vector<double>& q) { return !q.empty() && *std::max_element(q.begin(), q.end()) >= 1 - epsilon; });
}

struct CategoricalUmda {
  std::size_t mu;
  bool borders = true;
  ProductCategorical::Parameters operator()(const ProductCategorical& f, const ProductCategorical::Parameters& p, std::span<const ProductCategorical::Point> ranked, std::span<const Evaluation>) const {
    if (mu == 0 || mu > ranked.size()) throw std::invalid_argument("heuresis::distributions::CategoricalUmda: 0 < μ ≤ λ");
    ProductCategorical::Parameters q(p.size());
    for (std::size_t i = 0; i < p.size(); i++) {
      q[i].assign(f.sizes[i], 0.0);
      for (std::size_t k = 0; k < mu; k++) q[i][ranked[k][i]] += 1.0 / static_cast<double>(mu);
      if (borders) clamp_categorical(q[i], 1.0 / (static_cast<double>(f.sizes[i] - 1) * static_cast<double>(p.size())));
    }
    return q;
  }
};

}  // namespace heuresis::distributions
