// Acquisition functions: where a surrogate says the next evaluation is worth most.
//
// For minimisation, with posterior N(μ(x), σ(x)²), the best value seen y*, and z = (y* − μ − ξ)/σ:
//   expected improvement      EI = E[max(y* − ξ − f(x), 0)] = (y* − μ − ξ)Φ(z) + σφ(z)   (Močkus, Tiešis and Žilinskas
//                             1978; Jones, Schonlau and Welch 1998, "Efficient global optimization of expensive black-
//                             box functions", J. Global Optim. 13, 455–492);
//   probability of improvement PI = Φ(z) (Kushner 1964);
//   lower confidence bound     LCB = μ − κσ, minimised (Srinivas, Krause, Kakade and Seeger 2010, "Gaussian process
//                             optimization in the bandit setting", ICML, in its minimising form).
// Each is returned as a value to maximise (−LCB for the last). With σ = 0 the first two degenerate to their limits.
// An acquisition is maximised over the box by many keyed uniform samples, the best few refined by Nelder–Mead on the
// clipped point: multistart local search, adequate in the few dimensions where Gaussian processes serve.

#pragma once

#include <heuresis/models/process.hpp>
#include <heuresis/variation/real.hpp>

#include <tyche/probability/distributions.hpp>

#include <calx/optimisation/nelder_mead.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

namespace heuresis::models {

enum class AcquisitionKind : std::uint8_t { expected_improvement, probability_of_improvement, lower_confidence_bound };

inline double expected_improvement(double mu, double sd, double best, double xi = 0) {
  const double d = best - mu - xi;
  if (!(sd > 0)) return std::max(d, 0.0);
  const tyche::probability::Normal standard{0, 1};
  const double z = d / sd;
  return d * standard.cdf(z) + sd * standard.pdf(z);
}

inline double probability_of_improvement(double mu, double sd, double best, double xi = 0) {
  const double d = best - mu - xi;
  if (!(sd > 0)) return d > 0 ? 1 : 0;
  return tyche::probability::Normal{0, 1}.cdf(d / sd);
}

inline double lower_confidence_bound(double mu, double sd, double kappa = 2) { return mu - kappa * sd; }

struct Acquisition {
  AcquisitionKind kind = AcquisitionKind::expected_improvement;
  double xi = 0;     // EI's and PI's margin
  double kappa = 2;  // LCB's weight
  // The value to maximise at a posterior, given the best value seen.
  double operator()(const GaussianProcess::Posterior& p, double best) const {
    switch (kind) {
      case AcquisitionKind::expected_improvement: return expected_improvement(p.mean, p.sd(), best, xi);
      case AcquisitionKind::probability_of_improvement: return probability_of_improvement(p.mean, p.sd(), best, xi);
      case AcquisitionKind::lower_confidence_bound: return -lower_confidence_bound(p.mean, p.sd(), kappa);
    }
    return 0;
  }
};

struct Maximum {
  std::vector<double> x;
  double value = -INFINITY;
};

// The maximum of a function over a box: `samples` uniform points from the stream, the best `starts` (and any `seeds`)
// refined by Nelder–Mead on the clipped point.
inline Maximum maximise_over_box(const std::function<double(std::span<const double>)>& g, const heuresis::variation::Box& box, calx::random::Stream& s, std::size_t samples = 2000, std::size_t starts = 5, const std::vector<std::vector<double>>& seeds = {}, int evaluations_each = 150) {
  struct Point {
    std::vector<double> x;
    double v;
  };
  std::vector<Point> pool;
  pool.reserve(samples + seeds.size());
  for (std::size_t k = 0; k < samples; k++) {
    auto x = heuresis::variation::uniform_point(box, s);
    const double v = g(x);
    pool.push_back({std::move(x), v});
  }
  std::partial_sort(pool.begin(), pool.begin() + static_cast<std::ptrdiff_t>(std::min(starts, pool.size())), pool.end(), [](const Point& a, const Point& b) { return a.v > b.v; });
  pool.resize(std::min(starts, pool.size()));
  for (const auto& x : seeds) pool.push_back({x, g(x)});
  auto clip = [&](std::span<const double> x) {
    std::vector<double> y(x.begin(), x.end());
    for (std::size_t i = 0; i < y.size(); i++) y[i] = std::clamp(y[i], box.lower[i], box.upper[i]);
    return y;
  };
  Maximum best;
  for (const auto& p : pool) {
    calx::optimisation::NelderMeadOptions o;
    o.max_evaluations = evaluations_each;
    std::vector<double> step(box.size());
    for (std::size_t i = 0; i < step.size(); i++) step[i] = 0.05 * box.width(i);
    o.initial_step = step;
    o.x_tolerance = 1e-9;
    o.f_tolerance = 1e-14;
    const auto r = calx::optimisation::nelder_mead([&](std::span<const double> x) { return -g(clip(x)); }, p.x, o);
    const auto x = clip(r.x);
    const double v = g(x);
    if (v > best.value) best = {x, v};
    if (p.v > best.value) best = {p.x, p.v};
  }
  return best;
}

}  // namespace heuresis::models
