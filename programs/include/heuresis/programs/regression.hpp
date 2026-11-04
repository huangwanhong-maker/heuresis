// Symbolic regression: a term scored by how well it predicts the targets of a set of cases.
//
// The evaluation's objective is the mean squared error over the cases; its descriptors are the absolute errors
// case by case, the term's behaviour on the data, which lexicase selection reads (selection/lexicase.hpp) and the
// search for diversity can use. A term whose value is not finite on some case (a division by zero, the logarithm
// of a negative number) is a failure with that reason, never a large made-up error.

#pragma once

#include <heuresis/programs/trees.hpp>
#include <heuresis/search/evaluation.hpp>

#include <cmath>
#include <string>
#include <vector>

namespace heuresis::programs {

struct Cases {
  std::vector<std::vector<double>> inputs;  // one assignment of the variables per case
  std::vector<double> targets;
};

// |t(xᵢ) − yᵢ| for every case; NaN where the value is not finite.
inline std::vector<double> case_errors(const Signature& sig, const Tree& t, const Cases& c) {
  std::vector<double> e(c.targets.size());
  for (std::size_t i = 0; i < e.size(); i++) {
    const double v = interpret(sig, t, c.inputs[i]);
    e[i] = std::isfinite(v) ? std::fabs(v - c.targets[i]) : NAN;
  }
  return e;
}

inline search::Evaluation regression(const Signature& sig, const Tree& t, const Cases& c) {
  auto e = case_errors(sig, t, c);
  double mse = 0;
  for (std::size_t i = 0; i < e.size(); i++) {
    if (std::isnan(e[i])) return search::failure("not finite on case " + std::to_string(i));
    mse += e[i] * e[i];
  }
  mse /= static_cast<double>(e.empty() ? 1 : e.size());
  return search::evaluation({mse}, {}, std::move(e));
}

}  // namespace heuresis::programs
