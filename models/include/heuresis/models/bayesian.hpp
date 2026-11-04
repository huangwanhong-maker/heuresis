// Bayesian optimisation: the next evaluation where a Gaussian-process surrogate expects most from it.
//
// For objectives that cost hours an evaluation (a simulation, an experiment), the cost of choosing well is negligible:
// the surrogate is refitted after every evaluation, and the next point maximises an acquisition (Jones, Schonlau and
// Welch 1998; Shahriari, Swersky, Wang, Adams and de Freitas 2016, "Taking the human out of the loop: a review of
// Bayesian optimization", Proc. IEEE 104, 148–175). The box is mapped onto the unit cube and the values standardised
// (mean 0, standard deviation 1) before the surrogate sees them, so that one prior on the hyperparameters serves every
// problem.
//
// As an ask–tell algorithm it is asynchronous: an initial design of uniform points (keyed by their index) is offered
// at once; afterwards each ask offers at most `breadth` points chosen from the current state (decision 0003 §2.2).
// Points asked but not yet told — the rest of a batch, evaluations still running on other workers — are filled in by a
// lie: the kriging believer gives each the surrogate's mean there, the constant liar the best value seen (Ginsbourger,
// Le Riche and Carraro 2010, "Kriging is well-suited to parallelize optimization", in Computational Intelligence in
// Expensive Optimization Problems, Springer, 131–162). The lies make a batch's points spread out without a
// multi-point acquisition. Its proposals depend on what has been told, and when; a record of arrivals replays it.
//
// Constraints and failures are not modelled: an evaluation that failed or is infeasible enters the surrogate with the
// worst value seen, so that the search learns to avoid it (a classifier of feasibility is an open question).

#pragma once

#include <heuresis/models/acquisition.hpp>
#include <heuresis/models/process.hpp>
#include <heuresis/search/draws.hpp>
#include <heuresis/search/protocol.hpp>
#include <heuresis/variation/real.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <optional>
#include <vector>

namespace heuresis::models {

using heuresis::variation::Box;
using heuresis::variation::Real;

enum class Lie : std::uint8_t { kriging_believer, constant_liar };

struct BayesianOptions {
  std::size_t initial = 0;       // the initial design; 0: 2d + 2; at least 2
  std::size_t breadth = 1;       // points offered per ask after the initial design
  Acquisition acquisition{};
  Lie lie = Lie::kriging_believer;
  KernelKind kernel = KernelKind::matern52;
  bool relevance = true;         // one length scale per coordinate
  std::size_t samples = 2000;    // uniform points for the acquisition's maximisation
  std::size_t starts = 5;        // of which this many are refined
  int fit_evaluations = 200;     // Nelder–Mead's budget for the hyperparameters
};

class BayesianOptimisation {
 public:
  using Genotype = Real;

  BayesianOptimisation(Box box, BayesianOptions options = {}, heuresis::search::Draws draws = heuresis::search::Draws{}) : box_(std::move(box)), options_(options), draws_(draws) {
    box_.validate();
    for (std::size_t i = 0; i < box_.size(); i++)
      if (!std::isfinite(box_.width(i)) || !(box_.width(i) > 0)) throw std::invalid_argument("heuresis::models::BayesianOptimisation: a bounded box of positive widths");
    if (options_.initial == 0) options_.initial = 2 * box_.size() + 2;
    options_.initial = std::max<std::size_t>(options_.initial, 2);  // a surrogate needs two points
    if (options_.breadth == 0) throw std::invalid_argument("heuresis::models::BayesianOptimisation: breadth ≥ 1");
    kernel_.kind = options_.kernel;
    kernel_.lengthscales.assign(options_.relevance ? box_.size() : 1, 0.3);
  }

  std::vector<heuresis::search::Candidate<Real>> ask(std::size_t limit) {
    std::vector<heuresis::search::Candidate<Real>> out;
    while (designed_ < options_.initial && out.size() < limit) {
      calx::random::Stream s = draws_(design_tag, 0, designed_);
      out.push_back(issue(heuresis::variation::uniform_point(Box::cube(box_.size(), 0, 1), s), design_tag));
      designed_++;
    }
    if (!out.empty() || told_.size() < 2) return out;
    const std::size_t k = std::min(limit, options_.breadth);
    if (k == 0) return out;
    refit();
    GaussianProcess gp = surrogate();
    for (const auto& [ticket, u] : pending_) gp.observe(u, lie(gp, u));
    for (std::size_t j = 0; j < k; j++) {
      calx::random::Stream s = draws_(propose_tag, proposals_, j);
      const double best = best_standardised();
      auto acq = [&](std::span<const double> u) { return options_.acquisition(gp.predict(u), best); };
      std::vector<std::vector<double>> seeds;
      if (incumbent_) seeds.push_back(*incumbent_);
      const auto m = maximise_over_box(acq, Box::cube(box_.size(), 0, 1), s, options_.samples, options_.starts, seeds);
      gp.observe(m.x, lie(gp, m.x));
      out.push_back(issue(m.x, propose_tag));
    }
    proposals_++;
    return out;
  }

  void tell(std::span<const heuresis::search::Told> told) {
    for (const auto& t : told) {
      const auto it = pending_.find(t.ticket);
      if (it == pending_.end()) throw std::invalid_argument("heuresis::models::BayesianOptimisation: a ticket that is not outstanding");
      const bool usable = t.evaluation.feasible();
      told_.push_back({it->second, usable ? t.evaluation.value() : std::numeric_limits<double>::quiet_NaN()});
      if (usable && (!best_ || t.evaluation.value() < *best_)) { best_ = t.evaluation.value(); incumbent_ = it->second; }
      pending_.erase(it);
    }
  }

  bool stopped() const { return false; }

  // The surrogate on the evaluations told (in the unit cube, standardised values).
  GaussianProcess surrogate() const {
    GaussianProcess gp(kernel_, noise_, 0);
    const auto [mu, sd] = standardisation();
    const double worst = worst_value();
    for (const auto& [u, y] : told_) gp.observe(u, ((std::isnan(y) ? worst : y) - mu) / sd);
    return gp;
  }
  // A point of the unit cube in the problem's box, and back.
  Real to_box(std::span<const double> u) const {
    Real x(u.size());
    for (std::size_t i = 0; i < x.size(); i++) x[i] = box_.lower[i] + u[i] * box_.width(i);
    return x;
  }
  const Kernel& kernel() const { return kernel_; }
  double noise() const { return noise_; }

  static constexpr std::uint32_t design_tag = heuresis::search::tag("models.bayes.design");
  static constexpr std::uint32_t propose_tag = heuresis::search::tag("models.bayes.propose");

 private:
  heuresis::search::Candidate<Real> issue(const std::vector<double>& u, std::uint32_t operation) {
    const auto t = heuresis::search::make_ticket(draws_.island(), serial_++);
    pending_[t] = u;
    return {t, to_box(u), heuresis::search::Origin{operation, {}}};
  }
  std::pair<double, double> standardisation() const {
    double s = 0, s2 = 0, n = 0;
    for (const auto& [u, y] : told_)
      if (!std::isnan(y)) { s += y; s2 += y * y; n++; }
    if (n == 0) return {0, 1};
    const double mu = s / n, var = n > 1 ? (s2 - n * mu * mu) / (n - 1) : 0;
    return {mu, var > 0 ? std::sqrt(var) : 1};
  }
  double worst_value() const {
    double w = -std::numeric_limits<double>::infinity();
    for (const auto& [u, y] : told_)
      if (!std::isnan(y)) w = std::max(w, y);
    return std::isfinite(w) ? w : 0;
  }
  double best_standardised() const {
    const auto [mu, sd] = standardisation();
    return best_ ? (*best_ - mu) / sd : 0;
  }
  double lie(const GaussianProcess& gp, std::span<const double> u) const {
    return options_.lie == Lie::kriging_believer ? gp.predict(u).mean : best_standardised();
  }
  void refit() {
    std::vector<std::vector<double>> x;
    std::vector<double> y;
    const auto [mu, sd] = standardisation();
    const double worst = worst_value();
    for (const auto& [u, v] : told_) {
      x.push_back(u);
      y.push_back(((std::isnan(v) ? worst : v) - mu) / sd);
    }
    const Fit f = fit_hyperparameters(x, y, kernel_, noise_, 0, HyperparameterBounds{}, options_.fit_evaluations);
    if (std::isfinite(f.log_likelihood)) { kernel_ = f.kernel; noise_ = f.noise; }
  }

  Box box_;
  BayesianOptions options_;
  heuresis::search::Draws draws_;
  Kernel kernel_;
  double noise_ = 1e-4;
  std::map<heuresis::search::Ticket, std::vector<double>> pending_;  // in the unit cube
  std::vector<std::pair<std::vector<double>, double>> told_;         // (u, y), y NaN for a failure
  std::optional<double> best_;
  std::optional<std::vector<double>> incumbent_;
  std::size_t designed_ = 0;
  std::uint64_t serial_ = 0, proposals_ = 0;
};

}  // namespace heuresis::models
