// Crowding, and the indicators that measure an approximation A of a front against a reference set Z.
//
// - Crowding distance (Deb et al. 2002, NSGA-II, §III.B): within a front, the sum over objectives of the gap
//   between a point's two neighbours along that objective, divided by the front's range in it; the extremes of
//   each objective get +∞. It is NSGA-II's secondary order: within a front, the less crowded is preferred.
// - GD and IGD: the mean Euclidean distance from each point of A to its nearest point of Z, and from each point
//   of Z to its nearest point of A (Van Veldhuizen and Lamont 2000; Coello Coello and Reyes Sierra 2004; the means,
//   p = 1, as Ishibuchi, Masuda, Tanigaki and Nojima 2015 recommend).
// - IGD⁺: as IGD with the distance d⁺(z, a) = ‖max(a − z, 0)‖, which counts only where a is worse than z; it is
//   weakly Pareto compliant, unlike IGD (Ishibuchi et al. 2015, "Modified distance calculation in generational
//   distance and inverted generational distance", EMO, LNCS 9019, 110–125, Theorem 1).
// - The additive ε-indicator I_ε+(A, B) = max_{b∈B} min_{a∈A} max_i (aᵢ − bᵢ): the least ε by which A must be
//   translated to weakly dominate B (Zitzler et al. 2003, §IV.B). I_ε+(A, B) ≤ 0 exactly when A weakly dominates B.
// - Spread Δ in two objectives (Deb et al. 2002, eq. 1): (d_f + d_l + Σ|dᵢ − d̄|)/(d_f + d_l + (N − 1)d̄), with dᵢ
//   the gaps between consecutive points of A sorted by f₁, d̄ their mean, and d_f, d_l the distances from the front's
//   extremes to A's; 0 for an evenly spread set that reaches the extremes.

#pragma once

#include <heuresis/multiobjective/dominance.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

namespace heuresis::multiobjective {

// The crowding distance of each member of a front (indices into points), in the order given.
inline std::vector<double> crowding_distance(std::span<const Point> points, std::span<const std::size_t> front) {
  const std::size_t n = front.size();
  std::vector<double> d(n, 0.0);
  if (n == 0) return d;
  if (n <= 2) {
    std::fill(d.begin(), d.end(), std::numeric_limits<double>::infinity());
    return d;
  }
  const std::size_t m = points[front[0]].size();
  std::vector<std::size_t> o(n);
  for (std::size_t k = 0; k < m; k++) {
    std::iota(o.begin(), o.end(), std::size_t{0});
    std::stable_sort(o.begin(), o.end(), [&](std::size_t a, std::size_t b) { return points[front[a]][k] < points[front[b]][k]; });
    const double lo = points[front[o.front()]][k], hi = points[front[o.back()]][k];
    d[o.front()] = d[o.back()] = std::numeric_limits<double>::infinity();
    if (!(hi > lo)) continue;
    for (std::size_t j = 1; j + 1 < n; j++) d[o[j]] += (points[front[o[j + 1]]][k] - points[front[o[j - 1]]][k]) / (hi - lo);
  }
  return d;
}

namespace detail {
inline double distance(std::span<const double> a, std::span<const double> b) {
  double s = 0;
  for (std::size_t i = 0; i < a.size(); i++) s += (a[i] - b[i]) * (a[i] - b[i]);
  return std::sqrt(s);
}
inline double distance_plus(std::span<const double> z, std::span<const double> a) {
  double s = 0;
  for (std::size_t i = 0; i < a.size(); i++) {
    const double d = std::max(a[i] - z[i], 0.0);
    s += d * d;
  }
  return std::sqrt(s);
}
template <class D>
double mean_nearest(std::span<const Point> from, std::span<const Point> to, D&& dist) {
  if (from.empty() || to.empty()) throw std::invalid_argument("heuresis::multiobjective: an empty set");
  double s = 0;
  for (const auto& f : from) {
    double best = std::numeric_limits<double>::infinity();
    for (const auto& t : to) best = std::min(best, dist(f, t));
    s += best;
  }
  return s / static_cast<double>(from.size());
}
}  // namespace detail

inline double generational_distance(std::span<const Point> approximation, std::span<const Point> reference) {
  return detail::mean_nearest(approximation, reference, [](const Point& a, const Point& z) { return detail::distance(a, z); });
}

inline double inverted_generational_distance(std::span<const Point> approximation, std::span<const Point> reference) {
  return detail::mean_nearest(reference, approximation, [](const Point& z, const Point& a) { return detail::distance(z, a); });
}

inline double inverted_generational_distance_plus(std::span<const Point> approximation, std::span<const Point> reference) {
  return detail::mean_nearest(reference, approximation, [](const Point& z, const Point& a) { return detail::distance_plus(z, a); });
}

inline double additive_epsilon(std::span<const Point> a, std::span<const Point> b) {
  if (a.empty() || b.empty()) throw std::invalid_argument("heuresis::multiobjective::additive_epsilon: an empty set");
  double worst = -std::numeric_limits<double>::infinity();
  for (const auto& y : b) {
    double best = std::numeric_limits<double>::infinity();
    for (const auto& x : a) {
      double e = -std::numeric_limits<double>::infinity();
      for (std::size_t i = 0; i < x.size(); i++) e = std::max(e, x[i] - y[i]);
      best = std::min(best, e);
    }
    worst = std::max(worst, best);
  }
  return worst;
}

// Deb's Δ for a two-objective approximation, given the extremes of the true front.
inline double spread(std::span<const Point> approximation, std::span<const double> first_extreme, std::span<const double> last_extreme) {
  std::vector<Point> p(approximation.begin(), approximation.end());
  if (p.size() < 2) throw std::invalid_argument("heuresis::multiobjective::spread: at least two points");
  std::sort(p.begin(), p.end(), [](const Point& a, const Point& b) { return a[0] < b[0]; });
  std::vector<double> gaps;
  for (std::size_t i = 0; i + 1 < p.size(); i++) gaps.push_back(detail::distance(p[i], p[i + 1]));
  const double mean = std::accumulate(gaps.begin(), gaps.end(), 0.0) / static_cast<double>(gaps.size());
  const double df = detail::distance(first_extreme, p.front()), dl = detail::distance(last_extreme, p.back());
  double dev = 0;
  for (double g : gaps) dev += std::fabs(g - mean);
  return (df + dl + dev) / (df + dl + static_cast<double>(gaps.size()) * mean);
}

}  // namespace heuresis::multiobjective
