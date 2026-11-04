// Real vectors in a box, and their variation.
//
// The space is a box B = Π [lᵢ, uᵢ] ⊂ ℝⁿ (unbounded coordinates have infinite bounds). A point that leaves the
// box is brought back by a repair, which is part of the operator and changes its distribution near the walls:
// clipping puts mass on the walls, reflection folds it back, wrapping treats the box as a torus, resampling
// redraws the offending coordinates (Kukkonen and Lampinen 2006; Biedrzycki, Arabas and Jagodziński 2019,
// "Bound constraints handling in differential evolution: an experimental study", Swarm Evol. Comput. 50).
//
// Mutations (one parent, one child):
// - Gaussian: xᵢ + σᵢ N(0, 1) with probability `rate` per coordinate (Rechenberg 1973; Schwefel 1981);
// - Cauchy: xᵢ + s C(0, 1), heavier tails (Yao, Liu and Lin 1999, "Evolutionary programming made faster",
//   IEEE Trans. Evol. Comput. 3, 82–102);
// - polynomial, bounded, with distribution index η_m (Deb and Goyal 1996, "A combined genetic adaptive search
//   (GeneAS) for engineering design", Comput. Sci. Informatics 26, 30–45; the bounded form of Deb's NSGA-II
//   code, as pymoo has it).
// Crossovers (two parents):
// - simulated binary crossover, bounded, with distribution index η_c (Deb and Agrawal 1995, "Simulated binary
//   crossover for continuous search space", Complex Systems 9, 115–148; the bounded form of Deb's NSGA-II
//   code): each coordinate crossed with probability ½, the two children exchanged with probability ½;
// - BLX-α: each coordinate uniform on the parents' interval widened by α on both sides (Eshelman and Schaffer
//   1993, "Real-coded genetic algorithms and interval-schemata", FOGA 2, 187–202);
// - weighted intermediate recombination of any number of parents (Beyer and Schwefel 2002, "Evolution
//   strategies: a comprehensive introduction", Natural Computing 1, 3–52, §4.1.3);
// - iso+line: x + σ_iso N(0, I) + σ_line N(0, 1)(y − x), the directional variation of quality-diversity
//   (Vassiliades and Mouret 2018, "Discovering the elite hypervolume by leveraging interspecies correlation",
//   GECCO, eq. 3).

#pragma once

#include <calx/random/stream.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace heuresis::variation {

using Real = std::vector<double>;

struct Box {
  std::vector<double> lower, upper;

  static Box cube(std::size_t n, double lo, double hi) { return {std::vector<double>(n, lo), std::vector<double>(n, hi)}; }
  static Box unbounded(std::size_t n) { return cube(n, -std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()); }
  std::size_t size() const { return lower.size(); }
  bool contains(std::span<const double> x) const {
    if (x.size() != size()) return false;
    for (std::size_t i = 0; i < x.size(); i++)
      if (!(x[i] >= lower[i] && x[i] <= upper[i])) return false;
    return true;
  }
  double width(std::size_t i) const { return upper[i] - lower[i]; }
  void validate() const {
    if (lower.size() != upper.size()) throw std::invalid_argument("heuresis::variation::Box: bounds of different lengths");
    for (std::size_t i = 0; i < size(); i++)
      if (!(lower[i] <= upper[i])) throw std::invalid_argument("heuresis::variation::Box: a lower bound above its upper bound");
  }
};

enum class Repair : std::uint8_t { clip, reflect, wrap, resample };

// Brings x back into the box.
inline void repair(Real& x, const Box& b, Repair how, calx::random::Stream& s) {
  for (std::size_t i = 0; i < x.size(); i++) {
    const double l = b.lower[i], u = b.upper[i], w = u - l;
    if (x[i] >= l && x[i] <= u) continue;
    if (!std::isfinite(w)) {  // a half-open coordinate can only be clipped
      x[i] = std::clamp(x[i], l, u);
      continue;
    }
    switch (how) {
      case Repair::clip: x[i] = std::clamp(x[i], l, u); break;
      case Repair::reflect: {
        double t = std::fmod(x[i] - l, 2 * w);
        if (t < 0) t += 2 * w;
        x[i] = l + (t <= w ? t : 2 * w - t);
        break;
      }
      case Repair::wrap: {
        double t = std::fmod(x[i] - l, w);
        if (t < 0) t += w;
        x[i] = l + t;
        break;
      }
      case Repair::resample: x[i] = s.uniform(l, u); break;
    }
  }
}

// A uniform point of a bounded box.
inline Real uniform_point(const Box& b, calx::random::Stream& s) {
  Real x(b.size());
  for (std::size_t i = 0; i < x.size(); i++) {
    if (!std::isfinite(b.width(i))) throw std::invalid_argument("heuresis::variation::uniform_point: an unbounded coordinate");
    x[i] = s.uniform(b.lower[i], b.upper[i]);
  }
  return x;
}

inline Real gaussian_mutation(const Real& x, std::span<const double> sigma, calx::random::Stream& s, double rate = 1) {
  Real y = x;
  for (std::size_t i = 0; i < y.size(); i++)
    if (rate >= 1 || s.uniform() < rate) y[i] += sigma[sigma.size() == 1 ? 0 : i] * s.normal();
  return y;
}
inline Real gaussian_mutation(const Real& x, double sigma, calx::random::Stream& s, double rate = 1) { return gaussian_mutation(x, std::span<const double>(&sigma, 1), s, rate); }

inline Real cauchy_mutation(const Real& x, double scale, calx::random::Stream& s, double rate = 1) {
  Real y = x;
  for (std::size_t i = 0; i < y.size(); i++)
    if (rate >= 1 || s.uniform() < rate) y[i] += scale * std::tan(std::numbers::pi * (s.uniform() - 0.5));
  return y;
}

// Bounded polynomial mutation; rate defaults to 1/n.
inline Real polynomial_mutation(const Real& x, const Box& b, double eta, calx::random::Stream& s, double rate = -1) {
  if (rate < 0) rate = x.empty() ? 0 : 1.0 / static_cast<double>(x.size());
  Real y = x;
  const double power = 1 / (eta + 1);
  for (std::size_t i = 0; i < y.size(); i++) {
    if (!(s.uniform() < rate)) continue;
    const double l = b.lower[i], u = b.upper[i], w = u - l;
    if (!(w > 0)) continue;
    const double d1 = (y[i] - l) / w, d2 = (u - y[i]) / w, r = s.uniform();
    double dq;
    if (r < 0.5) {
      const double v = 2 * r + (1 - 2 * r) * std::pow(1 - d1, eta + 1);
      dq = std::pow(v, power) - 1;
    } else {
      const double v = 2 * (1 - r) + 2 * (r - 0.5) * std::pow(1 - d2, eta + 1);
      dq = 1 - std::pow(v, power);
    }
    y[i] = std::clamp(y[i] + dq * w, l, u);
  }
  return y;
}

// Bounded simulated binary crossover.
inline std::pair<Real, Real> simulated_binary_crossover(const Real& p1, const Real& p2, const Box& b, double eta, calx::random::Stream& s) {
  if (p1.size() != p2.size()) throw std::invalid_argument("heuresis::variation::simulated_binary_crossover: parents of different lengths");
  std::pair<Real, Real> c{p1, p2};
  for (std::size_t i = 0; i < p1.size(); i++) {
    if (!(s.uniform() <= 0.5)) continue;
    if (std::fabs(p1[i] - p2[i]) <= 1e-14) continue;
    const double y1 = std::min(p1[i], p2[i]), y2 = std::max(p1[i], p2[i]), l = b.lower[i], u = b.upper[i];
    const double r = s.uniform();
    auto betaq = [&](double beta) {
      const double alpha = 2 - std::pow(beta, -(eta + 1));
      return r <= 1 / alpha ? std::pow(r * alpha, 1 / (eta + 1)) : std::pow(1 / (2 - r * alpha), 1 / (eta + 1));
    };
    double c1 = 0.5 * ((y1 + y2) - betaq(1 + 2 * (y1 - l) / (y2 - y1)) * (y2 - y1));
    double c2 = 0.5 * ((y1 + y2) + betaq(1 + 2 * (u - y2) / (y2 - y1)) * (y2 - y1));
    c1 = std::clamp(c1, l, u);
    c2 = std::clamp(c2, l, u);
    if (s.uniform() <= 0.5) std::swap(c1, c2);
    c.first[i] = c1;
    c.second[i] = c2;
  }
  return c;
}

inline std::pair<Real, Real> blend_crossover(const Real& p1, const Real& p2, double alpha, calx::random::Stream& s) {
  if (p1.size() != p2.size()) throw std::invalid_argument("heuresis::variation::blend_crossover: parents of different lengths");
  std::pair<Real, Real> c{p1, p2};
  for (std::size_t i = 0; i < p1.size(); i++) {
    const double lo = std::min(p1[i], p2[i]), hi = std::max(p1[i], p2[i]), d = hi - lo;
    c.first[i] = s.uniform(lo - alpha * d, hi + alpha * d);
    c.second[i] = s.uniform(lo - alpha * d, hi + alpha * d);
  }
  return c;
}

// Σ wᵢ xᵢ / Σ wᵢ over the parents; equal weights when none are given.
inline Real intermediate_recombination(std::span<const Real> parents, std::span<const double> weights = {}) {
  if (parents.empty()) throw std::invalid_argument("heuresis::variation::intermediate_recombination: no parents");
  Real m(parents[0].size(), 0.0);
  double total = 0;
  for (std::size_t k = 0; k < parents.size(); k++) {
    const double w = weights.empty() ? 1.0 : weights[k];
    total += w;
    for (std::size_t i = 0; i < m.size(); i++) m[i] += w * parents[k][i];
  }
  for (auto& v : m) v /= total;
  return m;
}

inline Real iso_line(const Real& x, const Real& y, double sigma_iso, double sigma_line, calx::random::Stream& s) {
  Real c = x;
  const double t = sigma_line * s.normal();
  for (std::size_t i = 0; i < c.size(); i++) c[i] += sigma_iso * s.normal() + t * (y[i] - x[i]);
  return c;
}

}  // namespace heuresis::variation
