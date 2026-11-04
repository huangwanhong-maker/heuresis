// The hypervolume indicator: the Lebesgue measure of what a set of points dominates, up to a reference point.
//
// For a finite set A ⊂ ℝᵐ and a reference point r, HV(A; r) = λ(⋃_{a ∈ A, a ≺ r} [a, r]), the measure of the union of
// the boxes between each point and r (Zitzler and Thiele 1998, "Multiobjective optimization using evolutionary
// algorithms — a comparative case study", PPSN V, 292–301). It is the only known unary indicator that is strictly
// Pareto compliant: if A dominates B (every b ∈ B is weakly dominated by some a ∈ A, and A ≠ B as sets of what they
// dominate), then HV(A) > HV(B) (Zitzler, Thiele, Laumanns, Fonseca and Grunert da Fonseca 2003, "Performance
// assessment of multiobjective optimizers: an analysis and review", IEEE Trans. Evol. Comput. 7, 117–132, §V;
// Fleischer 2003). Maximising it therefore drives a set towards the front; its maximisers spread along the front
// with a density set by the front's slope (Auger, Bader, Brockhoff and Zitzler 2009, FOGA).
//
// Only points strictly better than r in every objective contribute; the others are ignored.
//
// Algorithms, each the exact measure (they are checked against each other and against inclusion–exclusion):
// - two objectives: a sweep in f₁ over the staircase, O(n log n);
// - three objectives: a sweep in f₃ keeping the staircase of the (f₁, f₂) projections in a balanced tree and its
//   area updated on each insertion, O(n log n) (Beume, Fonseca, López-Ibáñez, Paquete and Vahrenhold 2009, "On the
//   complexity of computing the hypervolume indicator", IEEE Trans. Evol. Comput. 13, 1075–1082, §IV);
// - any number: WFG, HV(A) = Σ_k [λ([a_k, r]) − HV(nds(limit(A, k)))], limit(A, k) = {a_k ∨ a_j : j > k}, ∨ the
//   componentwise maximum (While, Bradstreet and Barone 2012, "A fast way of calculating exact hypervolumes",
//   IEEE Trans. Evol. Comput. 16, 86–95), with the three- and two-objective sweeps as its base;
// - inclusion–exclusion over the subsets, Σ_{∅≠S⊆A} (−1)^{|S|+1} λ([∨S, r]), O(2ⁿ m): the definition, for checks.
//
// The exclusive contribution of a ∈ A is HV(A) − HV(A ∖ {a}): in two objectives the rectangle between a and its
// neighbours on the staircase; in general the difference, computed so.

#pragma once

#include <heuresis/multiobjective/dominance.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <map>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

namespace heuresis::multiobjective {

namespace detail {

inline void check_reference(std::span<const Point> points, std::span<const double> r) {
  for (const auto& p : points)
    if (p.size() != r.size()) throw std::invalid_argument("heuresis::multiobjective::hypervolume: a point and the reference of different dimensions");
}

// The points strictly better than r in every objective.
inline std::vector<Point> inside(std::span<const Point> points, std::span<const double> r) {
  std::vector<Point> out;
  for (const auto& p : points) {
    bool in = true;
    for (std::size_t i = 0; i < r.size() && in; i++) in = p[i] < r[i];
    if (in) out.push_back(p);
  }
  return out;
}

inline double box(std::span<const double> a, std::span<const double> r) {
  double v = 1;
  for (std::size_t i = 0; i < r.size(); i++) v *= std::max(0.0, r[i] - a[i]);
  return v;
}

inline double sweep2(std::vector<Point> p, std::span<const double> r) {
  std::sort(p.begin(), p.end(), [](const Point& a, const Point& b) { return a[0] != b[0] ? a[0] < b[0] : a[1] < b[1]; });
  double hv = 0, y = r[1];
  for (const auto& q : p)
    if (q[1] < y) {
      hv += (r[0] - q[0]) * (y - q[1]);
      y = q[1];
    }
  return hv;
}

// The staircase of points in the plane under a reference (rx, ry), and the area it dominates.
class Staircase {
 public:
  Staircase(double rx, double ry) : rx_(rx), ry_(ry) {}
  // Adds (px, py); returns the area it adds.
  double insert(double px, double py) {
    auto it = steps_.upper_bound(px);
    if (it != steps_.begin() && std::prev(it)->second <= py) return 0;  // weakly dominated
    it = steps_.lower_bound(px);
    double cur_x = px, cur_h = it != steps_.begin() ? std::prev(it)->second : ry_, added = 0;
    while (it != steps_.end() && it->second >= py) {
      added += (it->first - cur_x) * (cur_h - py);
      cur_x = it->first;
      cur_h = it->second;
      it = steps_.erase(it);
    }
    const double next_x = it != steps_.end() ? it->first : rx_;
    added += (next_x - cur_x) * (cur_h - py);
    steps_.emplace(px, py);
    area_ += added;
    return added;
  }
  double area() const { return area_; }

 private:
  double rx_, ry_, area_ = 0;
  std::map<double, double> steps_;  // x ascending, y strictly descending
};

inline double sweep3(std::vector<Point> p, std::span<const double> r) {
  std::sort(p.begin(), p.end(), [](const Point& a, const Point& b) { return a[2] < b[2]; });
  Staircase s(r[0], r[1]);
  double hv = 0;
  for (std::size_t i = 0; i < p.size(); i++) {
    s.insert(p[i][0], p[i][1]);
    const double z_next = i + 1 < p.size() ? p[i + 1][2] : r[2];
    hv += s.area() * (z_next - p[i][2]);
  }
  return hv;
}

inline std::vector<Point> nondominated_points(const std::vector<Point>& p) {
  std::vector<Point> out;
  for (std::size_t i = 0; i < p.size(); i++) {
    bool keep = true;
    for (std::size_t j = 0; j < p.size() && keep; j++) {
      if (j == i) continue;
      // drop i if j dominates it, or j equals it and comes first
      keep = !(dominates(p[j], p[i]) || (j < i && p[j] == p[i]));
    }
    if (keep) out.push_back(p[i]);
  }
  return out;
}

inline double wfg(std::vector<Point> p, std::span<const double> r) {
  if (p.empty()) return 0;
  const std::size_t m = r.size();
  if (m == 1) {
    double lo = r[0];
    for (const auto& q : p) lo = std::min(lo, q[0]);
    return r[0] - lo;
  }
  if (m == 2) return sweep2(std::move(p), r);
  if (m == 3) return sweep3(std::move(p), r);
  std::sort(p.begin(), p.end(), [m](const Point& a, const Point& b) { return a[m - 1] > b[m - 1]; });
  double hv = 0;
  for (std::size_t k = 0; k < p.size(); k++) {
    std::vector<Point> limit;
    limit.reserve(p.size() - k - 1);
    for (std::size_t j = k + 1; j < p.size(); j++) {
      Point w(m);
      for (std::size_t i = 0; i < m; i++) w[i] = std::max(p[k][i], p[j][i]);
      limit.push_back(std::move(w));
    }
    hv += box(p[k], r) - wfg(nondominated_points(limit), r);
  }
  return hv;
}

}  // namespace detail

// HV in two objectives by the sweep.
inline double hypervolume_2d(std::span<const Point> points, std::span<const double> r) {
  if (r.size() != 2) throw std::invalid_argument("heuresis::multiobjective::hypervolume_2d: two objectives");
  detail::check_reference(points, r);
  return detail::sweep2(detail::inside(points, r), r);
}

// HV in three objectives by Beume et al.'s sweep.
inline double hypervolume_3d(std::span<const Point> points, std::span<const double> r) {
  if (r.size() != 3) throw std::invalid_argument("heuresis::multiobjective::hypervolume_3d: three objectives");
  detail::check_reference(points, r);
  return detail::sweep3(detail::inside(points, r), r);
}

// HV in any number of objectives by WFG.
inline double hypervolume_wfg(std::span<const Point> points, std::span<const double> r) {
  detail::check_reference(points, r);
  const auto in = detail::inside(points, r);
  return detail::wfg(detail::nondominated_points(in), r);
}

// HV by inclusion–exclusion over the subsets; at most 24 points.
inline double hypervolume_inclusion_exclusion(std::span<const Point> points, std::span<const double> r) {
  detail::check_reference(points, r);
  const std::size_t n = points.size(), m = r.size();
  if (n > 24) throw std::invalid_argument("heuresis::multiobjective::hypervolume_inclusion_exclusion: at most 24 points");
  double hv = 0;
  Point join(m);
  for (std::uint32_t mask = 1; mask < (1u << n); mask++) {
    std::fill(join.begin(), join.end(), -INFINITY);
    int size = 0;
    for (std::size_t k = 0; k < n; k++)
      if (mask >> k & 1u) {
        size++;
        for (std::size_t i = 0; i < m; i++) join[i] = std::max(join[i], points[k][i]);
      }
    hv += (size % 2 ? 1 : -1) * detail::box(join, r);
  }
  return hv;
}

// HV by the fastest exact method for the dimension.
inline double hypervolume(std::span<const Point> points, std::span<const double> r) {
  if (r.size() == 2) return hypervolume_2d(points, r);
  if (r.size() == 3) return hypervolume_3d(points, r);
  return hypervolume_wfg(points, r);
}

// The exclusive contribution of each point: HV(A) − HV(A ∖ {a}); zero for a point dominated, or equal to another.
inline std::vector<double> contributions(std::span<const Point> points, std::span<const double> r) {
  detail::check_reference(points, r);
  const std::size_t n = points.size();
  std::vector<double> c(n, 0.0);
  if (r.size() == 2) {
    // The staircase of distinct non-dominated points inside r, x ascending; each contributes the rectangle up to
    // its neighbours.
    std::vector<std::size_t> idx;
    for (std::size_t i = 0; i < n; i++)
      if (points[i][0] < r[0] && points[i][1] < r[1]) idx.push_back(i);
    std::sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) { return points[a][0] != points[b][0] ? points[a][0] < points[b][0] : points[a][1] < points[b][1]; });
    std::vector<std::size_t> stair;
    std::vector<char> twin;
    double y = r[1];
    for (std::size_t k = 0; k < idx.size(); k++) {
      const auto& p = points[idx[k]];
      if (p[1] < y) {
        stair.push_back(idx[k]);
        twin.push_back(0);
        y = p[1];
      } else if (!stair.empty() && points[stair.back()] == p) {
        twin.back() = 1;
      }
    }
    for (std::size_t j = 0; j < stair.size(); j++) {
      if (twin[j]) continue;
      const double right = j + 1 < stair.size() ? points[stair[j + 1]][0] : r[0];
      const double up = j > 0 ? points[stair[j - 1]][1] : r[1];
      c[stair[j]] = (right - points[stair[j]][0]) * (up - points[stair[j]][1]);
    }
    return c;
  }
  const double total = hypervolume(points, r);
  std::vector<Point> rest(points.begin(), points.end());
  for (std::size_t i = 0; i < n; i++) {
    Point keep = rest[i];
    rest.erase(rest.begin() + static_cast<std::ptrdiff_t>(i));
    c[i] = std::max(0.0, total - hypervolume(rest, r));
    rest.insert(rest.begin() + static_cast<std::ptrdiff_t>(i), std::move(keep));
  }
  return c;
}

}  // namespace heuresis::multiobjective
