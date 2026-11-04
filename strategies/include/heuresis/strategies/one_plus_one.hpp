// The (1+1) evolution strategy with the one-fifth success rule.
//
// One parent x and one offspring y = x + σ N(0, I) per step; y replaces x unless x is better (plus selection,
// ties to the offspring, so a plateau can be crossed). The step size follows Rechenberg's one-fifth rule
// (Rechenberg 1973, Evolutionsstrategie, Frommann-Holzboog): enlarge σ after a success, shrink it after a
// failure, so that about one step in five succeeds, the rate near which progress on the sphere is greatest.
// Implemented in the smooth form of Kern, Müller, Hansen, Büche, Ocenasek and Koumoutsakos (2004, "Learning
// probability distributions in continuous evolutionary algorithms — a comparative review", Natural Computing 3,
// 77–112, §3.1):
//
//   σ ← σ exp((1/d)(𝟙[success] − p_target)/(1 − p_target)),  p_target = 1/5,  d = 1 + n/2,
//
// so that σ is stationary exactly when the success rate is p_target. On the sphere its convergence is linear:
// log f decreases in proportion to the number of evaluations (Jägersküpper 2007, "Algorithmic analysis of a
// basic evolutionary algorithm for continuous optimization", Theor. Comput. Sci. 379, 329–347, for the
// one-fifth rule; Akimoto, Auger and Glasmachers 2018, "Drift theory in continuous search spaces: expected
// hitting time of the (1+1)-ES with 1/5 success rule", GECCO, for the smooth form).
//
// It is generational with one candidate per generation: the first is the initial point, then one offspring at a
// time. It stops once σ has fallen below min_sigma, by default 10⁻¹² σ₀: twelve decades below its start a step
// changes the point by little more than rounding, and an algorithm that never stops spends the rest of any budget
// where it settled. min_sigma = 0 never stops.

#pragma once

#include <heuresis/search/draws.hpp>
#include <heuresis/search/protocol.hpp>
#include <heuresis/variation/real.hpp>

#include <cmath>
#include <optional>
#include <stdexcept>

namespace heuresis::strategies {

struct OnePlusOneOptions {
  double target_success = 0.2;
  double damping = 0;  // 0: 1 + n/2
  std::optional<heuresis::variation::Box> bounds;
  heuresis::variation::Repair repair = heuresis::variation::Repair::reflect;
  std::optional<double> min_sigma;  // stopped once σ falls below it; by default 10⁻¹² σ₀ (0: never)
};

class OnePlusOneEs {
 public:
  using Genotype = heuresis::variation::Real;
  static constexpr std::uint32_t purpose = heuresis::search::tag("strategies.one_plus_one.mutate");

  OnePlusOneEs(Genotype x0, double sigma, OnePlusOneOptions options = {}, heuresis::search::Draws draws = heuresis::search::Draws{})
      : options_(std::move(options)), draws_(draws), sigma_(sigma) {
    if (x0.empty() || !(sigma > 0)) throw std::invalid_argument("heuresis::strategies::OnePlusOneEs: an empty point or a step size that is not positive");
    if (options_.damping <= 0) options_.damping = 1 + static_cast<double>(x0.size()) / 2;
    if (!options_.min_sigma) options_.min_sigma = 1e-12 * sigma;
    start_ = std::move(x0);
  }

  std::vector<heuresis::search::Candidate<Genotype>> ask(std::size_t limit) {
    if (!batch_.active() && !stopped()) {
      Genotype y;
      heuresis::search::Origin origin{purpose, {}};
      if (!parent_) {
        y = start_;
      } else {
        calx::random::Stream s = draws_(purpose, step_, 0);
        y = heuresis::variation::gaussian_mutation(parent_->genotype, sigma_, s);
        if (options_.bounds) heuresis::variation::repair(y, *options_.bounds, options_.repair, s);
        origin.parents = {parent_->ticket};
      }
      batch_.open({{heuresis::search::make_ticket(draws_.island(), step_), std::move(y), std::move(origin)}});
    }
    return batch_.take(limit);
  }

  void tell(std::span<const heuresis::search::Told> told) {
    for (const auto& t : told)
      if (batch_.accept(t)) {
        auto child = batch_.individuals()[0];
        batch_.close();
        step_++;
        if (!parent_) { parent_ = std::move(child); continue; }
        const bool success = !heuresis::search::better(parent_->evaluation, child.evaluation);
        sigma_ *= std::exp(((success ? 1.0 : 0.0) - options_.target_success) / (options_.damping * (1 - options_.target_success)));
        successes_ += success;
        if (success) parent_ = std::move(child);
      }
  }

  bool stopped() const { return !batch_.active() && sigma_ < *options_.min_sigma; }
  double sigma() const { return sigma_; }
  double min_sigma() const { return *options_.min_sigma; }
  const std::optional<heuresis::search::Individual<Genotype>>& parent() const { return parent_; }
  std::uint64_t successes() const { return successes_; }
  std::uint64_t steps() const { return step_; }

 private:
  OnePlusOneOptions options_;
  heuresis::search::Draws draws_;
  double sigma_;
  Genotype start_;
  std::optional<heuresis::search::Individual<Genotype>> parent_;
  heuresis::search::Batch<Genotype> batch_;
  std::uint64_t step_ = 0, successes_ = 0;
};

}  // namespace heuresis::strategies
