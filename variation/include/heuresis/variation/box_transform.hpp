// A smooth map of ℝⁿ onto a box, so that a search on ℝⁿ can search a box.
//
// Each coordinate is mapped by Hansen's linear–quadratic transformation, as pycma's
// BoxConstraintsLinQuadTransformation has it (Hansen, pycma, cma/transformations.py; Hansen 2009, "Benchmarking a
// BI-population CMA-ES on the BBOB-2009 function testbed", §3, for its use): the identity on [l + a_l, u − a_u],
// quadratic beyond, reaching the bound with slope 0 at l − a_l and u + a_u, then mirrored, and periodic with
// period 2(u − l + a_l + a_u). The margins are a = min((u − l)/2, 0.05 max(|bound|, 1)). The map is C¹ and
// onto, so a bound is reached at an interior point of the genotype space: an optimum on the boundary of the box is
// an ordinary optimum of the transformed problem, where repairing a sample instead (reflection, clipping) lets the
// step size grow without bound against the wall. The inverse maps a point of the box to its preimage in
// [l − a_l, u + a_u].

#pragma once

#include <heuresis/variation/real.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

namespace heuresis::variation {

class BoxTransform {
 public:
  explicit BoxTransform(Box box) : box_(std::move(box)) {
    box_.validate();
    const std::size_t n = box_.size();
    al_.resize(n);
    au_.resize(n);
    for (std::size_t i = 0; i < n; i++) {
      const double l = box_.lower[i], u = box_.upper[i];
      al_[i] = std::isfinite(l) ? std::min((u - l) / 2, 0.05 * std::max(std::fabs(l), 1.0)) : 1;
      au_[i] = std::isfinite(u) ? std::min((u - l) / 2, 0.05 * std::max(std::fabs(u), 1.0)) : 1;
    }
  }

  double forward(double y, std::size_t i) const {
    const double l = box_.lower[i], u = box_.upper[i], al = al_[i], au = au_[i];
    if (std::isfinite(l) && std::isfinite(u) && (y < l - 2 * al - (u - l) / 2 || y > u + 2 * au + (u - l) / 2)) {
      const double r = 2 * (u - l + al + au), s = l - 2 * al - (u - l) / 2;
      y -= r * std::floor((y - s) / r);
    }
    if (y > u + au) y -= 2 * (y - u - au);
    if (y < l - al) y += 2 * (l - al - y);
    if (y < l + al) y = l + (y - (l - al)) * (y - (l - al)) / (4 * al);
    if (y > u - au) y = u - (y - (u + au)) * (y - (u + au)) / (4 * au);
    return y;
  }

  heuresis::variation::Real forward(std::span<const double> y) const {
    heuresis::variation::Real x(y.size());
    for (std::size_t i = 0; i < y.size(); i++) x[i] = forward(y[i], i);
    return x;
  }

  // The preimage in [l − a_l, u + a_u] of a point of the box.
  double inverse(double x, std::size_t i) const {
    const double l = box_.lower[i], u = box_.upper[i], al = al_[i], au = au_[i];
    x = std::clamp(x, l, u);
    if (x < l + al) return l - al + 2 * std::sqrt(al * (x - l));
    if (x > u - au) return u + au - 2 * std::sqrt(au * (u - x));
    return x;
  }

  heuresis::variation::Real inverse(std::span<const double> x) const {
    heuresis::variation::Real y(x.size());
    for (std::size_t i = 0; i < x.size(); i++) y[i] = inverse(x[i], i);
    return y;
  }

  const Box& box() const { return box_; }

 private:
  Box box_;
  std::vector<double> al_, au_;
};

}  // namespace heuresis::variation
