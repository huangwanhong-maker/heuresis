// Gaussian-process surrogates: a prior over objectives, conditioned on the evaluations made.
//
// An expensive objective (a simulation that takes hours, an experiment) is modelled as a draw f from a Gaussian
// process with constant mean μ₀ and a stationary covariance k; the evaluations yᵢ = f(xᵢ) + εᵢ, εᵢ ~ N(0, σₙ²), are
// observations of linear functionals of f, and the posterior of f(x) at any x is Gaussian with the mean and variance of
// tyche's Gaussian conditioning (tyche/spatial/conditioning.hpp: Rasmussen and Williams 2006, Gaussian Processes for
// Machine Learning, eqs. 2.23–2.24), extended one evaluation at a time in O(n²) and with the log marginal likelihood
// log p(y) read from its factor. Nothing here writes that algebra again; the surrogate supplies the covariances.
//
// Kernels, r the distance scaled by the length scales ℓ (one per coordinate, automatic relevance determination, or one
// for all):
//   squared exponential  k = σ_f² exp(−r²/2),
//   Matérn ν = 5/2        k = σ_f² (1 + √5 r + 5r²/3) exp(−√5 r)   (Rasmussen and Williams 2006, §4.2),
// the second giving twice-differentiable sample paths, the usual choice for optimisation (Snoek, Larochelle and Adams
// 2012, "Practical Bayesian optimization of machine learning algorithms", NeurIPS).
//
// The hyperparameters (log σ_f², log ℓ, log σₙ²) are chosen to maximise log p(y), by calx's Nelder–Mead from a start
// (type-II maximum likelihood, Rasmussen and Williams 2006, ch. 5), within bounds that keep the covariance well conditioned.

#pragma once

#include <tyche/spatial/conditioning.hpp>

#include <calx/optimisation/nelder_mead.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace heuresis::models {

enum class KernelKind : std::uint8_t { squared_exponential, matern52 };

struct Kernel {
  KernelKind kind = KernelKind::matern52;
  double variance = 1;               // σ_f²
  std::vector<double> lengthscales;  // ℓ, one per coordinate or one for all

  double scaled_distance(std::span<const double> a, std::span<const double> b) const {
    double s = 0;
    for (std::size_t i = 0; i < a.size(); i++) {
      const double l = lengthscales.size() == 1 ? lengthscales[0] : lengthscales[i];
      const double d = (a[i] - b[i]) / l;
      s += d * d;
    }
    return std::sqrt(s);
  }
  double operator()(std::span<const double> a, std::span<const double> b) const {
    const double r = scaled_distance(a, b);
    if (kind == KernelKind::squared_exponential) return variance * std::exp(-0.5 * r * r);
    const double t = std::sqrt(5.0) * r;
    return variance * (1 + t + t * t / 3) * std::exp(-t);
  }
};

class GaussianProcess {
 public:
  struct Posterior {
    double mean = 0, variance = 0;
    double sd() const { return std::sqrt(variance); }
  };

  GaussianProcess(Kernel kernel, double noise, double mean = 0) : kernel_(std::move(kernel)), noise_(noise), mean_(mean) {
    if (!(noise >= 0)) throw std::invalid_argument("heuresis::models::GaussianProcess: a noise variance of zero or more");
  }

  void observe(std::span<const double> x, double y) {
    std::vector<double> k(points_.size());
    for (std::size_t i = 0; i < points_.size(); i++) k[i] = kernel_(points_[i], x);
    conditioning_.observe(k, kernel_(x, x), noise_, y - mean_);
    points_.emplace_back(x.begin(), x.end());
  }

  // The posterior of the latent f(x) (without the noise).
  Posterior predict(std::span<const double> x) const {
    std::vector<double> k(points_.size());
    for (std::size_t i = 0; i < points_.size(); i++) k[i] = kernel_(points_[i], x);
    const auto p = conditioning_.predict(k, mean_, kernel_(x, x));
    return {p.mean, p.variance};
  }

  double log_likelihood() const { return conditioning_.log_likelihood(); }
  std::size_t size() const { return points_.size(); }
  const Kernel& kernel() const { return kernel_; }
  double noise() const { return noise_; }
  double mean() const { return mean_; }

 private:
  Kernel kernel_;
  double noise_, mean_;
  std::vector<std::vector<double>> points_;
  tyche::spatial::GaussianConditioning conditioning_;
};

struct HyperparameterBounds {
  double log_variance_lo = std::log(1e-2), log_variance_hi = std::log(1e2);
  double log_lengthscale_lo = std::log(1e-3), log_lengthscale_hi = std::log(1e2);
  double log_noise_lo = std::log(1e-8), log_noise_hi = std::log(1.0);
};

struct Fit {
  Kernel kernel;
  double noise = 0;
  double log_likelihood = -std::numeric_limits<double>::infinity();
};

// The log marginal likelihood of (x, y) under a kernel and noise, or −∞ where the covariance is not positive definite.
inline double log_marginal_likelihood(std::span<const std::vector<double>> x, std::span<const double> y, const Kernel& kernel, double noise, double mean = 0) {
  GaussianProcess gp(kernel, noise, mean);
  try {
    for (std::size_t i = 0; i < x.size(); i++) gp.observe(x[i], y[i]);
  } catch (const std::domain_error&) {
    return -std::numeric_limits<double>::infinity();
  }
  return gp.log_likelihood();
}

// Type-II maximum likelihood from a start, by Nelder–Mead on the logarithms of the hyperparameters.
inline Fit fit_hyperparameters(std::span<const std::vector<double>> x, std::span<const double> y, const Kernel& start, double start_noise, double mean = 0, const HyperparameterBounds& bounds = {}, int max_evaluations = 200) {
  const std::size_t L = start.lengthscales.size();
  std::vector<double> t0{std::log(start.variance)};
  for (double l : start.lengthscales) t0.push_back(std::log(l));
  t0.push_back(std::log(std::max(start_noise, std::exp(bounds.log_noise_lo))));
  auto unpack = [&](std::span<const double> t, Kernel& k, double& noise) {
    k = start;
    k.variance = std::exp(t[0]);
    for (std::size_t i = 0; i < L; i++) k.lengthscales[i] = std::exp(t[1 + i]);
    noise = std::exp(t[1 + L]);
  };
  auto inside = [&](std::span<const double> t) {
    if (t[0] < bounds.log_variance_lo || t[0] > bounds.log_variance_hi) return false;
    for (std::size_t i = 0; i < L; i++)
      if (t[1 + i] < bounds.log_lengthscale_lo || t[1 + i] > bounds.log_lengthscale_hi) return false;
    return t[1 + L] >= bounds.log_noise_lo && t[1 + L] <= bounds.log_noise_hi;
  };
  auto objective = [&](std::span<const double> t) {
    if (!inside(t)) return std::numeric_limits<double>::infinity();
    Kernel k;
    double noise;
    unpack(t, k, noise);
    const double l = log_marginal_likelihood(x, y, k, noise, mean);
    return std::isfinite(l) ? -l : std::numeric_limits<double>::infinity();
  };
  for (std::size_t i = 0; i < t0.size(); i++) {  // a start inside the bounds
    const double lo = i == 0 ? bounds.log_variance_lo : i == 1 + L ? bounds.log_noise_lo : bounds.log_lengthscale_lo;
    const double hi = i == 0 ? bounds.log_variance_hi : i == 1 + L ? bounds.log_noise_hi : bounds.log_lengthscale_hi;
    t0[i] = std::clamp(t0[i], lo, hi);
  }
  calx::optimisation::NelderMeadOptions options;
  options.max_evaluations = max_evaluations;
  options.initial_step = {0.5};
  options.x_tolerance = 1e-4;
  options.f_tolerance = 1e-6;
  const auto r = calx::optimisation::nelder_mead(objective, t0, options);
  Fit fit;
  const auto& best = std::isfinite(r.value) && r.value <= objective(t0) ? r.x : t0;
  unpack(best, fit.kernel, fit.noise);
  fit.log_likelihood = -objective(best);
  return fit;
}

}  // namespace heuresis::models
