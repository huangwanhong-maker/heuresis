// Search distributions: families of laws P_θ on a space, the devices that sample them, and the information-
// geometric update that moves θ towards what was good.
//
// A family is a map θ ↦ P_θ from a parameter space Θ to the laws on a space X. The library keeps three things
// apart (decision 0003 §2.8):
//
//   the family     — Θ, the law's density, its geometry (score and Fisher information) and, for an exponential
//                    family, its sufficient statistic T and expectation parameters η = E_θ[T];
//   the sampler    — the device that draws x ~ P_θ. The reference sampler is the family's own `sample`, on the CPU,
//                    from a keyed stream: it defines the law. Any other device (an accelerator, a quantum device, an
//                    annealer, a probabilistic-bit machine) is a sampler of a law it defines, tested against the
//                    reference;
//   the rule       — how θ moves after a generation is evaluated.
//
// The information-geometric update (Ollivier, Arnold, Auger and Hansen 2017, "Information-geometric optimization
// algorithms: a unifying picture via invariance principles", JMLR 18(18), 1–65). A generation x₁, …, x_λ ~ P_θ is
// ranked best first; rank i receives a weight ŵᵢ = W((i − ½)/λ)/λ from a nonincreasing selection function W on
// [0, 1], tied points sharing the mean of their ranks' weights, so that the update depends on the objective only
// through the order it induces and is invariant under every strictly increasing transformation of it. The IGO step
// of size δt is
//
//   θ ← θ + δt Σᵢ ŵᵢ ∇̃_θ ln P_θ(xᵢ),   ∇̃ = I(θ)⁻¹ ∇,
//
// I(θ) the Fisher information. The natural gradient ∇̃ does not depend on how Θ is parametrised (to first order in
// δt), which is the invariance that makes the step a property of the family, not of its coordinates. Two closed
// forms follow (Ollivier et al. §2–§4):
//   - in the expectation parameters η of an exponential family, ∇̃_η ln P(x) = T(x) − η, so the step is
//     η ← η + δt Σ ŵᵢ (T(xᵢ) − η); with Σ ŵᵢ = 1 and δt = 1 it is the maximum-likelihood fit to the weighted
//     sample ("IGO-ML"), which for truncation weights is UMDA on bits and the cross-entropy method on Gaussians;
//   - for Bernoulli products, η = p and the step is PBIL's; for Gaussians in the coordinates (m, C) the step is
//     m ← m + δt Σ ŵᵢ (xᵢ − m), C ← C + δt Σ ŵᵢ ((xᵢ − m)(xᵢ − m)ᵀ − C), the rank-μ update of CMA-ES with equal
//     learning rates (gaussian.hpp).
// `igo_step` computes the step generically, from a family's score and Fisher information by a Cholesky solve; the
// checks hold it equal to the closed forms, which is a check of the theory, not only of the code.

#pragma once

#include <heuresis/search/evaluation.hpp>

#include <calx/linalg/cholesky.hpp>
#include <calx/linalg/dense.hpp>
#include <calx/random/stream.hpp>

#include <cmath>
#include <concepts>
#include <functional>
#include <span>
#include <stdexcept>
#include <vector>

namespace heuresis::distributions {

using heuresis::search::Evaluation;

template <class F>
concept Family = requires(const F& f, const typename F::Parameters& theta, const typename F::Point& x, calx::random::Stream& s) {
  typename F::Point;
  typename F::Parameters;
  { f.sample(theta, s) } -> std::convertible_to<typename F::Point>;
  { f.log_density(theta, x) } -> std::convertible_to<double>;
};

// An exponential family stated in its expectation parameters.
template <class F>
concept ExponentialFamily = Family<F> && requires(const F& f, const typename F::Parameters& theta, const typename F::Point& x, std::vector<double> eta) {
  { f.statistic(x) } -> std::convertible_to<std::vector<double>>;
  { f.expectation(theta) } -> std::convertible_to<std::vector<double>>;
  { f.from_expectation(eta) } -> std::convertible_to<typename F::Parameters>;
};

// A family with coordinates, a score and a Fisher information in them.
template <class F>
concept Riemannian = Family<F> && requires(const F& f, const typename F::Parameters& theta, const typename F::Point& x, std::vector<double> c) {
  { f.coordinates(theta) } -> std::convertible_to<std::vector<double>>;
  { f.from_coordinates(c) } -> std::convertible_to<typename F::Parameters>;
  { f.score(theta, x) } -> std::convertible_to<std::vector<double>>;
  { f.fisher(theta) } -> std::convertible_to<calx::linalg::Matrix>;
};

// A device that draws from the family's law.
template <class S, class F>
concept Sampler = Family<F> && requires(S& s, const F& f, const typename F::Parameters& theta, calx::random::Stream& r) {
  { s(f, theta, r) } -> std::convertible_to<typename F::Point>;
};

// The reference sampler: the family's own law, from the keyed stream.
struct Reference {
  template <Family F>
  typename F::Point operator()(const F& f, const typename F::Parameters& theta, calx::random::Stream& r) const { return f.sample(theta, r); }
};

// ŵᵢ = W((i − ½)/λ)/λ for the ranks i = 1..λ.
inline std::vector<double> quantile_weights(std::size_t lambda, const std::function<double(double)>& W) {
  std::vector<double> w(lambda);
  for (std::size_t i = 0; i < lambda; i++) w[i] = W((static_cast<double>(i) + 0.5) / static_cast<double>(lambda)) / static_cast<double>(lambda);
  return w;
}

// Truncation: 1/μ for the μ best, 0 for the rest (W = (λ/μ) 1{q ≤ μ/λ}).
inline std::vector<double> truncation_weights(std::size_t lambda, std::size_t mu) {
  if (mu == 0 || mu > lambda) throw std::invalid_argument("heuresis::distributions::truncation_weights: 0 < μ ≤ λ");
  std::vector<double> w(lambda, 0.0);
  for (std::size_t i = 0; i < mu; i++) w[i] = 1.0 / static_cast<double>(mu);
  return w;
}

// CMA-ES's weights: ln(μ + ½) − ln i for the μ best, normalised to sum 1 (Hansen 2016, "The CMA evolution strategy:
// a tutorial", arXiv:1604.00772, with μ = ⌊λ/2⌋ by default).
inline std::vector<double> logarithmic_weights(std::size_t lambda, std::size_t mu = 0) {
  if (mu == 0) mu = lambda / 2;
  if (mu == 0 || mu > lambda) throw std::invalid_argument("heuresis::distributions::logarithmic_weights: 0 < μ ≤ λ");
  std::vector<double> w(lambda, 0.0);
  double t = 0;
  for (std::size_t i = 0; i < mu; i++) t += (w[i] = std::log(static_cast<double>(mu) + 0.5) - std::log(static_cast<double>(i + 1)));
  for (auto& v : w) v /= t;
  return w;
}

// The weights of points ranked best first, each group of tied evaluations sharing the mean of its ranks' weights.
inline std::vector<double> tied_weights(std::span<const Evaluation> ranked, std::span<const double> weights) {
  if (ranked.size() != weights.size()) throw std::invalid_argument("heuresis::distributions::tied_weights: one weight per point");
  std::vector<double> w(weights.begin(), weights.end());
  for (std::size_t i = 0; i < ranked.size();) {
    std::size_t j = i + 1;
    while (j < ranked.size() && !heuresis::search::better(ranked[i], ranked[j]) && !heuresis::search::better(ranked[j], ranked[i])) j++;
    if (j - i > 1) {
      double m = 0;
      for (std::size_t k = i; k < j; k++) m += weights[k];
      m /= static_cast<double>(j - i);
      for (std::size_t k = i; k < j; k++) w[k] = m;
    }
    i = j;
  }
  return w;
}

// The IGO step in expectation parameters: η ← η + δt Σ ŵᵢ (T(xᵢ) − η).
template <ExponentialFamily F>
typename F::Parameters expectation_step(const F& f, const typename F::Parameters& theta, std::span<const typename F::Point> points, std::span<const double> weights, double dt) {
  std::vector<double> eta = f.expectation(theta);
  std::vector<double> step(eta.size(), 0.0);
  for (std::size_t i = 0; i < points.size(); i++) {
    if (weights[i] == 0) continue;
    const std::vector<double> t = f.statistic(points[i]);
    for (std::size_t k = 0; k < eta.size(); k++) step[k] += weights[i] * (t[k] - eta[k]);
  }
  for (std::size_t k = 0; k < eta.size(); k++) eta[k] += dt * step[k];
  return f.from_expectation(std::move(eta));
}

// The IGO step from the score and the Fisher information: c ← c + δt I(θ)⁻¹ Σ ŵᵢ ∇ ln P_θ(xᵢ).
template <Riemannian F>
typename F::Parameters igo_step(const F& f, const typename F::Parameters& theta, std::span<const typename F::Point> points, std::span<const double> weights, double dt) {
  std::vector<double> c = f.coordinates(theta);
  std::vector<double> g(c.size(), 0.0);
  for (std::size_t i = 0; i < points.size(); i++) {
    if (weights[i] == 0) continue;
    const std::vector<double> s = f.score(theta, points[i]);
    for (std::size_t k = 0; k < c.size(); k++) g[k] += weights[i] * s[k];
  }
  const auto factor = calx::linalg::cholesky(f.fisher(theta));
  if (!factor.positive_definite) throw std::domain_error("heuresis::distributions::igo_step: the Fisher information is not positive definite");
  const std::vector<double> natural = calx::linalg::cholesky_solve(factor, g);
  for (std::size_t k = 0; k < c.size(); k++) c[k] += dt * natural[k];
  return f.from_coordinates(std::move(c));
}

}  // namespace heuresis::distributions
