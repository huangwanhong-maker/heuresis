// The Metropolis rule, and the energy an evaluation gives it.
//
// A proposal y from the current state x, drawn by a symmetric kernel (q(x, y) = q(y, x)), is accepted with
// probability min(1, exp(−(E(y) − E(x))/T)) (Metropolis, Rosenbluth, Rosenbluth, Teller and Teller 1953, "Equation
// of state calculations by fast computing machines", J. Chem. Phys. 21, 1087–1092). The chain then satisfies
// detailed balance with respect to the Boltzmann law π_T(x) ∝ exp(−E(x)/T), which is its stationary law when the
// kernel is irreducible on a finite space; aperiodicity follows wherever a proposal can be rejected. With an
// asymmetric kernel the ratio q(y, x)/q(x, y) enters the rule (Hastings 1970), which these kernels do not need.
//
// The energy is the evaluation's first objective. A failure has energy +∞; so does an infeasible point unless a
// finite penalty coefficient μ is given, E = f + μ · violation: the Metropolis rule needs a number, and which one is
// the caller's choice, stated. A proposal of infinite energy is never accepted from a finite state; from an
// infinite one every proposal is, so a chain can leave the infeasible region.

#pragma once

#include <heuresis/search/evaluation.hpp>

#include <cmath>
#include <limits>

namespace heuresis::annealing {

inline constexpr double infinite_penalty = std::numeric_limits<double>::infinity();

inline double energy(const search::Evaluation& e, double penalty = infinite_penalty) {
  if (e.failed()) return std::numeric_limits<double>::infinity();
  const double v = e.violation();
  if (v == 0) return e.value();
  return std::isfinite(penalty) ? e.value() + penalty * v : std::numeric_limits<double>::infinity();
}

// Whether to move from energy `current` to `proposed` at temperature t, given a uniform u on (0, 1).
inline bool metropolis(double current, double proposed, double t, double u) {
  if (proposed <= current) return true;
  if (!std::isfinite(proposed)) return false;
  if (!std::isfinite(current)) return true;
  return u < std::exp(-(proposed - current) / t);
}

}  // namespace heuresis::annealing
