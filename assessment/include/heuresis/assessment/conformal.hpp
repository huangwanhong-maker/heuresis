// Surrogates whose intervals are calibrated, and screening on them that keeps every candidate that could be best
// (decision 0011; the survey of evolutionary allocation, Part VII §6.1, item 3, and Part V).
//
// SPLIT CONFORMAL INTERVALS (Vovk, Gammerman and Shafer 2005, Algorithmic Learning in a Random World; Lei, G'Sell, Rinaldo,
// Tibshirani and Wasserman 2018, "Distribution-free predictive inference for regression", J. Am. Stat. Assoc. 113,
// 1094–1111). A predictor fitted on runs of the true evaluator answers a centre ŷ(x) and a scale σ̂(x) (1 for the plain form;
// a Gaussian process's predictive deviation for the normalised form, whose widths adapt). On a calibration part of n further
// runs its scores are s_i = |y_i − ŷ(x_i)| / σ̂(x_i); with q̂ the ⌈(n + 1)(1 − α)⌉-th smallest, the interval ŷ(x) ± q̂ σ̂(x)
// covers a new run's true value with probability at least 1 − α whenever the calibration runs and the new one are
// exchangeable, and at most 1 − α + 1/(n + 1) when the scores have no ties (their Theorem 2.2). When ⌈(n + 1)(1 − α)⌉ > n
// the interval is the whole line, and says so.
//
// *Caveat.* The coverage is marginal, over new points drawn as the calibration points were. Points a search chooses are not
// drawn so: on them the coverage is whatever it is, and the screen below reports the coverage it achieved on every point the
// true evaluator assessed.
//
// SCREENING. Among candidates (to be minimised), one whose interval's lower end lies above the least upper end cannot be the
// minimiser if every interval covers; the rest are assessed by the true evaluator, and the least true value is chosen. So the
// true minimiser among the candidates is assessed, and chosen, whenever the intervals cover it and the one whose upper end is
// least: nothing is chosen on the surrogate's word, and what the surrogate saved is the candidates it screened out.
#pragma once

#include <heuresis/assessment/record.hpp>
#include <heuresis/models/process.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace heuresis::assessment {

// A centre and a scale at a point.
struct Prediction {
  double centre = 0, scale = 1;
};
using Predictor = std::function<Prediction(std::span<const double>)>;

struct Interval {
  double centre = 0, lower = 0, upper = 0;
  bool whole_line = false;
};

class Conformal {
 public:
  Conformal() = default;
  Conformal(Predictor p, double alpha, double q, std::size_t n) : predict_(std::move(p)), alpha_(alpha), q_(q), n_(n) {}
  Interval operator()(std::span<const double> x) const {
    const Prediction p = predict_(x);
    if (!std::isfinite(q_)) return {p.centre, -std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(), true};
    const double h = q_ * std::max(p.scale, 1e-300);
    return {p.centre, p.centre - h, p.centre + h, false};
  }
  double alpha() const { return alpha_; }
  double quantile() const { return q_; }
  std::size_t calibration() const { return n_; }
  // The guaranteed band of coverage, [1 − α, 1 − α + 1/(n + 1)].
  std::pair<double, double> guarantee() const { return {1 - alpha_, std::min(1.0, 1 - alpha_ + 1.0 / static_cast<double>(n_ + 1))}; }

 private:
  Predictor predict_;
  double alpha_ = 0.1, q_ = std::numeric_limits<double>::infinity();
  std::size_t n_ = 0;
};

// Calibrates a predictor on runs it was not fitted on.
inline Conformal calibrate(Predictor p, const std::vector<std::vector<double>>& x, const std::vector<double>& y, double alpha) {
  if (!p) throw Refusal("a conformal interval needs a predictor");
  if (x.size() != y.size() || x.empty()) throw Refusal("a true value for each calibration point, and at least one");
  if (!(alpha > 0 && alpha < 1)) throw Refusal("a level α strictly between 0 and 1");
  const std::size_t n = x.size();
  std::vector<double> s(n);
  for (std::size_t i = 0; i < n; i++) {
    const Prediction q = p(x[i]);
    if (!(q.scale > 0) || !std::isfinite(q.scale)) throw Refusal("a predictor's scale must be positive and finite");
    s[i] = std::fabs(y[i] - q.centre) / q.scale;
  }
  std::sort(s.begin(), s.end());
  const std::size_t k = static_cast<std::size_t>(std::ceil((static_cast<double>(n) + 1) * (1 - alpha) - 1e-12));
  const double q = k > n ? std::numeric_limits<double>::infinity() : s[k - 1];
  return Conformal(std::move(p), alpha, q, n);
}

// A Gaussian process fitted to runs (hyperparameters by marginal likelihood, the mean the runs' mean), as a predictor whose
// scale is its predictive deviation of a new run (normalised) or 1 (plain).
inline Predictor gaussian_process(const std::vector<std::vector<double>>& x, const std::vector<double>& y, bool normalised = true) {
  if (x.size() != y.size() || x.size() < 3) throw Refusal("a Gaussian process surrogate needs at least three runs, each with its value");
  double mean = 0;
  for (double v : y) mean += v / static_cast<double>(y.size());
  double var = 0;
  for (double v : y) var += (v - mean) * (v - mean) / static_cast<double>(y.size());
  models::Kernel k;
  k.kind = models::KernelKind::matern52;
  k.variance = var > 0 ? var : 1;
  k.lengthscales.assign(x[0].size(), 1.0);
  models::HyperparameterBounds bounds;
  bounds.log_variance_lo = std::log(std::max(var, 1e-12) * 1e-3), bounds.log_variance_hi = std::log(std::max(var, 1e-12) * 1e3);
  bounds.log_noise_lo = std::log(std::max(var, 1e-12) * 1e-8), bounds.log_noise_hi = std::log(std::max(var, 1e-12));
  const auto fit = models::fit_hyperparameters(x, y, k, std::max(var, 1e-12) * 1e-2, mean, bounds);
  auto gp = std::make_shared<models::GaussianProcess>(fit.kernel, fit.noise, mean);
  for (std::size_t i = 0; i < x.size(); i++) gp->observe(x[i], y[i]);
  const double noise = fit.noise;
  return [gp, noise, normalised](std::span<const double> z) {
    const auto p = gp->predict(z);
    return Prediction{p.mean, normalised ? std::sqrt(p.variance + noise) : 1.0};
  };
}

struct Screen {
  std::vector<std::size_t> assessed, screened;  // candidates assessed by the true evaluator, and screened out
  std::size_t best = 0;                         // the least true value among the assessed
  double best_value = std::numeric_limits<double>::infinity();
  std::size_t covered = 0;                      // assessed candidates whose true value lay within their interval
  double coverage() const { return assessed.empty() ? 0 : static_cast<double>(covered) / static_cast<double>(assessed.size()); }
  std::vector<Interval> intervals;              // every candidate's
};

// Candidates screened by a calibrated surrogate, the rest assessed by `truth` (to be minimised).
inline Screen screen(const std::vector<std::vector<double>>& candidates, const Conformal& c, const std::function<double(std::size_t)>& truth) {
  if (candidates.empty()) throw Refusal("a screen needs candidates");
  Screen S;
  double least_upper = std::numeric_limits<double>::infinity();
  for (const auto& x : candidates) {
    S.intervals.push_back(c(x));
    least_upper = std::min(least_upper, S.intervals.back().upper);
  }
  for (std::size_t i = 0; i < candidates.size(); i++) {
    if (S.intervals[i].lower > least_upper) {
      S.screened.push_back(i);
      continue;
    }
    S.assessed.push_back(i);
    const double v = truth(i);
    S.covered += v >= S.intervals[i].lower && v <= S.intervals[i].upper;
    if (v < S.best_value) S.best_value = v, S.best = i;
  }
  return S;
}

}  // namespace heuresis::assessment
