// What an evaluation reports, and the order in which evaluations are compared.
//
// A problem maps a point x of a space X to an evaluation. Every objective is minimised (a quantity to
// maximise is negated where it is stated); a constraint value g_j(x) is satisfied when g_j(x) ≤ 0, and its
// violation is max(0, g_j(x)); descriptors place x in a space of behaviours, for the search for diversity
// (diversity/); the cost is what the evaluation spent, in the unit the caller's budget counts (one per
// evaluation by default, or seconds, or core-hours). An evaluation may fail: a simulation that diverged, a
// point outside the model's domain, a device that did not answer. A failure says why, and is worse than any
// evaluation that did not fail; it is never answered with a made-up value.
//
// The order on single-objective evaluations is Deb's feasibility rule (Deb 2000, "An efficient constraint
// handling method for genetic algorithms", Comput. Methods Appl. Mech. Engrg. 186, 311–338, §3): a feasible
// point is better than an infeasible one, two feasible points are compared by their objective, and two
// infeasible ones by their total violation. It needs no penalty coefficients, and on a problem without
// constraints it is the order of the objective. `better` is a strict weak order; ties are left to the caller,
// who breaks them by ticket so that every result is reproducible. Orders on several objectives are in
// multiobjective/.

#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace heuresis::search {

enum class Status : std::uint8_t { ok, failed };

struct Evaluation {
  std::vector<double> objectives;   // minimised
  std::vector<double> constraints;  // satisfied when ≤ 0
  std::vector<double> descriptors;  // a behaviour, for diversity
  double cost = 1;                  // in the unit of the budget
  Status status = Status::ok;
  std::string reason;               // why it failed

  // Failed when the evaluator said so, gave no objective, or gave one that is not a number.
  bool failed() const {
    if (status == Status::failed || objectives.empty()) return true;
    for (double f : objectives)
      if (std::isnan(f)) return true;
    return false;
  }
  // Σ max(0, g_j); a constraint that is not a number is violated without bound.
  double violation() const {
    double v = 0;
    for (double g : constraints) {
      if (std::isnan(g)) return std::numeric_limits<double>::infinity();
      if (g > 0) v += g;
    }
    return v;
  }
  bool feasible() const { return !failed() && violation() == 0; }
  // The first objective, or +∞ for a failure.
  double value() const { return failed() ? std::numeric_limits<double>::infinity() : objectives[0]; }
};

inline Evaluation scalar(double f, double cost = 1) {
  Evaluation e;
  e.objectives = {f};
  e.cost = cost;
  return e;
}

inline Evaluation evaluation(std::vector<double> objectives, std::vector<double> constraints = {}, std::vector<double> descriptors = {}, double cost = 1) {
  Evaluation e;
  e.objectives = std::move(objectives);
  e.constraints = std::move(constraints);
  e.descriptors = std::move(descriptors);
  e.cost = cost;
  return e;
}

inline Evaluation failure(std::string reason, double cost = 1) {
  Evaluation e;
  e.status = Status::failed;
  e.reason = std::move(reason);
  e.cost = cost;
  return e;
}

// a is better than b by Deb's feasibility rule on the first objective; failures last.
inline bool better(const Evaluation& a, const Evaluation& b) {
  const bool fa = a.failed(), fb = b.failed();
  if (fa || fb) return !fa && fb;
  const double va = a.violation(), vb = b.violation();
  if (va != vb) return va < vb;
  return a.objectives[0] < b.objectives[0];
}

}  // namespace heuresis::search
