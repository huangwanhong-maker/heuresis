// Temperatures: the schedules of annealing, open-loop and adaptive.
//
// A schedule gives the temperature T_k of step k and may adapt it to the acceptances it observes. Open-loop:
// - constant, T_k = T, under which the Metropolis chain samples the Boltzmann law at T;
// - geometric, T_k = T₀ α^{⌊k/L⌋}, L steps at each level (Kirkpatrick, Gelatt and Vecchi 1983, "Optimization by
//   simulated annealing", Science 220, 671–680);
// - logarithmic, T_k = c / ln(k + k₀). On a finite space with a symmetric, irreducible neighbourhood, annealing
//   converges in probability to the set of global minima if and only if c ≥ d*, the greatest depth of a local
//   minimum that is not global (the energy one must climb from it to reach a lower state) (Hajek 1988, "Cooling
//   schedules for optimal annealing", Math. Oper. Res. 13, 311–329, Theorem 1). It is the only schedule here with
//   such a guarantee, and it is slow.
// Adaptive:
// - the modified Lam schedule: the temperature is multiplied or divided by 0.999 at each step so that the
//   acceptance rate, averaged as r ← (499 r + accepted)/500, follows a target that falls from 1 to 0.44 over the
//   first 15% of the steps (0.44 + 0.56 · 560^{−f/0.15}, f the fraction done), stays at 0.44 until 65%, and falls
//   as 0.44 · 440^{−(f − 0.65)/0.35} after (Lam and Delosme 1988; Swartz 1993; the constants as Boyan 1998,
//   Learning Evaluation Functions for Global Optimization, PhD thesis, Carnegie Mellon University, states them).

#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace heuresis::annealing {

class Schedule {
 public:
  static Schedule constant(double t) {
    Schedule s(Kind::constant);
    s.t0_ = s.t_ = check(t);
    return s;
  }
  static Schedule geometric(double t0, double alpha, std::uint64_t steps_per_level = 1) {
    if (!(alpha > 0 && alpha <= 1) || steps_per_level == 0) throw std::invalid_argument("heuresis::annealing::Schedule::geometric: 0 < α ≤ 1 and at least one step per level");
    Schedule s(Kind::geometric);
    s.t0_ = s.t_ = check(t0);
    s.alpha_ = alpha;
    s.level_ = steps_per_level;
    return s;
  }
  static Schedule logarithmic(double c, double offset = 2) {
    if (!(offset > 1)) throw std::invalid_argument("heuresis::annealing::Schedule::logarithmic: the offset must exceed 1");
    Schedule s(Kind::logarithmic);
    s.t0_ = check(c);
    s.offset_ = offset;
    s.t_ = c / std::log(offset);
    return s;
  }
  static Schedule modified_lam(double t0, std::uint64_t steps) {
    if (steps == 0) throw std::invalid_argument("heuresis::annealing::Schedule::modified_lam: at least one step");
    Schedule s(Kind::lam);
    s.t0_ = s.t_ = check(t0);
    s.total_ = steps;
    return s;
  }

  double temperature() const { return t_; }
  std::uint64_t step() const { return k_; }
  double acceptance() const { return rate_; }

  // The Lam schedule's target acceptance at the fraction f of the run.
  static double lam_target(double f) {
    if (f < 0.15) return 0.44 + 0.56 * std::pow(560.0, -f / 0.15);
    if (f < 0.65) return 0.44;
    return 0.44 * std::pow(440.0, -(f - 0.65) / 0.35);
  }

  // Moves to the next step, having observed whether this one's proposal was accepted.
  void advance(bool accepted) {
    rate_ = (499 * rate_ + (accepted ? 1.0 : 0.0)) / 500;
    k_++;
    switch (kind_) {
      case Kind::constant: break;
      case Kind::geometric: t_ = t0_ * std::pow(alpha_, static_cast<double>(k_ / level_)); break;
      case Kind::logarithmic: t_ = t0_ / std::log(static_cast<double>(k_) + offset_); break;
      case Kind::lam: t_ = rate_ > lam_target(static_cast<double>(k_) / static_cast<double>(total_)) ? t_ * 0.999 : t_ / 0.999; break;
    }
  }

 private:
  enum class Kind : std::uint8_t { constant, geometric, logarithmic, lam };
  explicit Schedule(Kind k) : kind_(k) {}
  static double check(double t) {
    if (!(t > 0) || !std::isfinite(t)) throw std::invalid_argument("heuresis::annealing::Schedule: a temperature must be positive and finite");
    return t;
  }

  Kind kind_;
  double t0_ = 1, t_ = 1, alpha_ = 1, offset_ = 2, rate_ = 0.5;
  std::uint64_t level_ = 1, total_ = 1, k_ = 0;
};

}  // namespace heuresis::annealing
