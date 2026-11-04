// Lexicase selection: parents chosen case by case, so that specialists survive.
//
// When an individual is scored on many cases (the data points of a regression, the tests of a program, the
// scenarios of a design), aggregating the errors into one number hides who solves what. Lexicase selection
// shuffles the cases and filters the population through them in that order, keeping at each case only the
// individuals best on it, until one remains or the cases are exhausted (Spector 2012, "Assessment of problem
// modality by differential performance of lexicase selection in genetic programming", GECCO companion,
// 401–408; Helmuth, Spector and Matheson 2015, IEEE Trans. Evol. Comput. 19, 630–643).
//
// ε-lexicase keeps, at each case, every individual within ε_c of the best, with ε_c the median absolute
// deviation of the population's errors on that case, which suits continuous errors (La Cava, Spector and
// Danai 2016, "Epsilon-lexicase selection for regression", GECCO, 741–748, "semi-dynamic" form with the
// population's ε).
//
// errors[i][c] is individual i's error on case c (to be minimised); ties among the survivors of every case are
// broken uniformly from the stream.

#pragma once

#include <calx/random/stream.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace heuresis::selection {

namespace detail {
inline double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  const std::size_t n = v.size();
  return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}
}  // namespace detail

// The median absolute deviation of each case's errors over the population.
inline std::vector<double> case_epsilons(const std::vector<std::vector<double>>& errors) {
  if (errors.empty()) return {};
  const std::size_t cases = errors[0].size();
  std::vector<double> eps(cases);
  for (std::size_t c = 0; c < cases; c++) {
    std::vector<double> col(errors.size());
    for (std::size_t i = 0; i < errors.size(); i++) col[i] = errors[i][c];
    const double m = detail::median(col);
    for (auto& v : col) v = std::fabs(v - m);
    eps[c] = detail::median(col);
  }
  return eps;
}

// One parent by lexicase selection; with epsilons, by ε-lexicase.
inline std::size_t lexicase(const std::vector<std::vector<double>>& errors, calx::random::Stream& s, const std::vector<double>& epsilons = {}) {
  if (errors.empty()) throw std::invalid_argument("heuresis::selection::lexicase: an empty population");
  const std::size_t cases = errors[0].size();
  std::vector<std::size_t> pool(errors.size()), order(cases);
  for (std::size_t i = 0; i < pool.size(); i++) pool[i] = i;
  for (std::size_t c = 0; c < cases; c++) order[c] = c;
  for (std::size_t i = cases; i > 1; i--) std::swap(order[i - 1], order[s.below(i)]);
  for (std::size_t k = 0; k < cases && pool.size() > 1; k++) {
    const std::size_t c = order[k];
    double best = INFINITY;
    for (auto i : pool) best = std::min(best, errors[i][c]);
    const double eps = epsilons.empty() ? 0 : epsilons[c];
    std::vector<std::size_t> next;
    for (auto i : pool)
      if (errors[i][c] <= best + eps) next.push_back(i);
    pool = std::move(next);
  }
  return pool[s.below(pool.size())];
}

}  // namespace heuresis::selection
