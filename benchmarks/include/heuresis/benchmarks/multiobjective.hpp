// Problems of several objectives with known Pareto fronts.
//
// ZDT1–4 and ZDT6 (Zitzler, Deb and Thiele 2000, "Comparison of multiobjective evolutionary algorithms:
// empirical results", Evol. Comput. 8, 173–195) have two objectives; DTLZ1–4 and DTLZ7 (Deb, Thiele, Laumanns
// and Zitzler 2005, "Scalable test problems for evolutionary multiobjective optimization", in Evolutionary
// Multiobjective Optimization, Springer, 105–145) have any number m, with n = m + k − 1 variables. All are
// minimised on [0, 1]ⁿ except ZDT4, whose variables after the first lie in [−5, 5].
//
// Their fronts, and the hypervolumes of the fronts with respect to the reference point (1, …, 1), are known in
// closed form (the theory note, §6):
//   ZDT1, ZDT4: f₂ = 1 − √f₁,                      HV = 2/3;
//   ZDT2:       f₂ = 1 − f₁²,                       HV = 1/3;
//   ZDT6:       f₂ = 1 − f₁², f₁ ∈ [f₁*, 1],          HV = (1 − f₁*³)/3, f₁* = 1 − exp(−4x̂) sin⁶(6πx̂) minimised;
//   DTLZ1:      Σ fᵢ = ½,                            HV = 1 − 2⁻ᵐ/m!;
//   DTLZ2–4:    Σ fᵢ² = 1,                           HV = 1 − π^{m/2}/(2ᵐ Γ(m/2 + 1)),
// the last two being the unit cube less the region below the front, a simplex and the positive part of a ball.
//
// The fronts of ZDT3 and DTLZ7 are disconnected, and are sets of running records (the case study "Fronts in three
// objectives" of The Science Infrastructures derived them so). On g = 1:
// - ZDT3 is the curve f = (t, z(t)), z(t) = 1 − √t − t sin 10πt. The point at t is non-dominated among the curve's
//   points exactly when z(t) < z(s) for every s < t: the front is the set of strict running records of z from below,
//   five segments.
// - DTLZ7 is f = (t₁, …, t_{m−1}, 2m − Σ φ(tᵢ)), φ(t) = t(1 + sin 3πt). Its last objective is separable, so
//   (t₁, …, t_{m−1}) is non-dominated exactly when each tᵢ is a strict running record of φ from above: if a′ ≤ a with
//   φ(a′) ≥ φ(a), not both equal, then a′ in place of a dominates; conversely, a point that dominates makes one of its
//   coordinates such a witness. The front is T*^{m−1}, T* two intervals: 2^{m−1} patches.
// Their hypervolumes, with r = (1, 1) for ZDT3 and r = (1, …, 1, 2m + 1) for DTLZ7, are the measure above the front's
// lower envelope, integrals of the running extremes Ψ(y) = min_{t ≤ y} z(t) and Φ(y) = max_{t ≤ y} φ(t):
//   ZDT3:  HV = ∫₀¹ (1 − Ψ(y)) dy                 = 1.044426…;
//   DTLZ7: HV = 1 + (m − 1) ∫₀¹ Φ(y) dy           = 2.457669… for m = 3,
// computed by the midpoint rule on the running extreme; 10⁶ cells agree with 4 × 10⁶ to 10⁻¹⁰.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace heuresis::benchmarks {

using Objectives = std::vector<double>;

namespace detail {
inline double zdt_g_linear(std::span<const double> x) {
  double s = 0;
  for (std::size_t i = 1; i < x.size(); i++) s += x[i];
  return 1 + 9 * s / static_cast<double>(x.size() - 1);
}
}  // namespace detail

inline Objectives zdt1(std::span<const double> x) {
  const double g = detail::zdt_g_linear(x);
  return {x[0], g * (1 - std::sqrt(x[0] / g))};
}
inline Objectives zdt2(std::span<const double> x) {
  const double g = detail::zdt_g_linear(x), r = x[0] / g;
  return {x[0], g * (1 - r * r)};
}
inline Objectives zdt3(std::span<const double> x) {
  const double g = detail::zdt_g_linear(x), r = x[0] / g;
  return {x[0], g * (1 - std::sqrt(r) - r * std::sin(10 * std::numbers::pi * x[0]))};
}
inline Objectives zdt4(std::span<const double> x) {
  double g = 1 + 10.0 * static_cast<double>(x.size() - 1);
  for (std::size_t i = 1; i < x.size(); i++) g += x[i] * x[i] - 10 * std::cos(4 * std::numbers::pi * x[i]);
  return {x[0], g * (1 - std::sqrt(x[0] / g))};
}
inline Objectives zdt6(std::span<const double> x) {
  const double f1 = 1 - std::exp(-4 * x[0]) * std::pow(std::sin(6 * std::numbers::pi * x[0]), 6);
  double s = 0;
  for (std::size_t i = 1; i < x.size(); i++) s += x[i];
  const double g = 1 + 9 * std::pow(s / static_cast<double>(x.size() - 1), 0.25), r = f1 / g;
  return {f1, g * (1 - r * r)};
}

namespace detail {
// DTLZ1's and DTLZ3's multimodal distance function over the last k variables.
inline double dtlz_g_rastrigin(std::span<const double> xm) {
  double s = static_cast<double>(xm.size());
  for (double v : xm) s += (v - 0.5) * (v - 0.5) - std::cos(20 * std::numbers::pi * (v - 0.5));
  return 100 * s;
}
inline double dtlz_g_sphere(std::span<const double> xm) {
  double s = 0;
  for (double v : xm) s += (v - 0.5) * (v - 0.5);
  return s;
}
inline Objectives dtlz_spherical(std::span<const double> x, std::size_t m, double g, double alpha) {
  Objectives f(m, 1 + g);
  const double h = std::numbers::pi / 2;
  for (std::size_t i = 0; i < m; i++) {
    for (std::size_t j = 0; j + 1 < m - i; j++) f[i] *= std::cos(std::pow(x[j], alpha) * h);
    if (i > 0) f[i] *= std::sin(std::pow(x[m - 1 - i], alpha) * h);
  }
  return f;
}
inline void check_size(std::span<const double> x, std::size_t m, const char* who) {
  if (m < 2 || x.size() < m) throw std::invalid_argument(std::string("heuresis::benchmarks::") + who + ": n ≥ m ≥ 2");
}
}  // namespace detail

inline Objectives dtlz1(std::span<const double> x, std::size_t m) {
  detail::check_size(x, m, "dtlz1");
  const double g = detail::dtlz_g_rastrigin(x.subspan(m - 1));
  Objectives f(m, 0.5 * (1 + g));
  for (std::size_t i = 0; i < m; i++) {
    for (std::size_t j = 0; j + 1 < m - i; j++) f[i] *= x[j];
    if (i > 0) f[i] *= 1 - x[m - 1 - i];
  }
  return f;
}
inline Objectives dtlz2(std::span<const double> x, std::size_t m) {
  detail::check_size(x, m, "dtlz2");
  return detail::dtlz_spherical(x, m, detail::dtlz_g_sphere(x.subspan(m - 1)), 1);
}
inline Objectives dtlz3(std::span<const double> x, std::size_t m) {
  detail::check_size(x, m, "dtlz3");
  return detail::dtlz_spherical(x, m, detail::dtlz_g_rastrigin(x.subspan(m - 1)), 1);
}
inline Objectives dtlz4(std::span<const double> x, std::size_t m, double alpha = 100) {
  detail::check_size(x, m, "dtlz4");
  return detail::dtlz_spherical(x, m, detail::dtlz_g_sphere(x.subspan(m - 1)), alpha);
}
inline Objectives dtlz7(std::span<const double> x, std::size_t m) {
  detail::check_size(x, m, "dtlz7");
  const auto xm = x.subspan(m - 1);
  double s = 0;
  for (double v : xm) s += v;
  const double g = 1 + 9 * s / static_cast<double>(xm.size());
  Objectives f(m);
  double h = static_cast<double>(m);
  for (std::size_t i = 0; i + 1 < m; i++) {
    f[i] = x[i];
    h -= f[i] / (1 + g) * (1 + std::sin(3 * std::numbers::pi * f[i]));
  }
  f[m - 1] = (1 + g) * h;
  return f;
}

// The least f₁ on ZDT6's front, by golden section on [0, 0.2] where it lies, to rounding.
inline double zdt6_least_f1() {
  auto f = [](double x) { return 1 - std::exp(-4 * x) * std::pow(std::sin(6 * std::numbers::pi * x), 6); };
  double a = 0.05, b = 0.12;
  const double r = (std::sqrt(5.0) - 1) / 2;
  for (int i = 0; i < 200; i++) {
    const double c = b - r * (b - a), d = a + r * (b - a);
    (f(c) < f(d) ? b : a) = (f(c) < f(d) ? d : c);
  }
  return f(0.5 * (a + b));
}

// The exact hypervolumes of the fronts with respect to (1, …, 1).
inline double zdt1_hypervolume() { return 2.0 / 3.0; }
inline double zdt2_hypervolume() { return 1.0 / 3.0; }
inline double zdt6_hypervolume() {
  const double f = zdt6_least_f1();
  return (1 - f * f * f) / 3;
}
inline double dtlz1_hypervolume(std::size_t m) { return 1 - std::pow(0.5, static_cast<double>(m)) / std::tgamma(static_cast<double>(m) + 1); }
inline double dtlz2_hypervolume(std::size_t m) {
  const double M = static_cast<double>(m);
  return 1 - std::pow(std::numbers::pi, M / 2) / (std::pow(2.0, M) * std::tgamma(M / 2 + 1));
}

// Points of the fronts: ZDT1's at f₁ = t, ZDT2's, and DTLZ2's on the positive unit sphere at given angles.
inline Objectives zdt1_front(double t) { return {t, 1 - std::sqrt(t)}; }
inline Objectives zdt2_front(double t) { return {t, 1 - t * t}; }

// Running records of a function h on [0, 1]: the fronts of curves whose first objective is t.
enum class Extreme : std::uint8_t { least, greatest };

// The strict running records of h on the grid tᵢ = i/(N − 1): the tᵢ at which h is below (least) or above (greatest)
// its every earlier value on the grid.
template <class H>
std::vector<double> running_records(H&& h, std::size_t points, Extreme extreme) {
  if (points < 2) throw std::invalid_argument("heuresis::benchmarks::running_records: at least two points");
  std::vector<double> out;
  double record = extreme == Extreme::least ? std::numeric_limits<double>::infinity() : -std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < points; i++) {
    const double t = static_cast<double>(i) / static_cast<double>(points - 1), v = h(t);
    if (extreme == Extreme::least ? v < record : v > record) {
      out.push_back(t);
      record = v;
    }
  }
  return out;
}

// ∫₀¹ of the running extreme of h (min or max over [0, y]) by the midpoint rule on `cells` cells, the extreme taken
// over the midpoints so far. The running extreme is continuous and monotone, so the error falls with the cells.
template <class H>
double running_extreme_integral(H&& h, std::size_t cells, Extreme extreme) {
  if (cells == 0) throw std::invalid_argument("heuresis::benchmarks::running_extreme_integral: at least one cell");
  double record = extreme == Extreme::least ? std::numeric_limits<double>::infinity() : -std::numeric_limits<double>::infinity(), s = 0;
  for (std::size_t i = 0; i < cells; i++) {
    const double v = h((static_cast<double>(i) + 0.5) / static_cast<double>(cells));
    record = extreme == Extreme::least ? std::min(record, v) : std::max(record, v);
    s += record;
  }
  return s / static_cast<double>(cells);
}

// ZDT3's curve at g = 1, z(t) = 1 − √t − t sin 10πt, and DTLZ7's φ(t) = t(1 + sin 3πt).
inline double zdt3_curve(double t) { return 1 - std::sqrt(t) - t * std::sin(10 * std::numbers::pi * t); }
inline double dtlz7_phi(double t) { return t * (1 + std::sin(3 * std::numbers::pi * t)); }

// ZDT3's front sampled on the grid of N points: the non-dominated points of the curve there.
inline std::vector<Objectives> zdt3_front(std::size_t points) {
  std::vector<Objectives> out;
  for (double t : running_records(zdt3_curve, points, Extreme::least)) out.push_back({t, zdt3_curve(t)});
  return out;
}

// DTLZ7's front in m objectives sampled on the grid of N points in each of its first m − 1: the product of the
// records of φ, in lexicographic order of the records' indices (the first coordinate slowest).
inline std::vector<Objectives> dtlz7_front(std::size_t points, std::size_t m = 3) {
  if (m < 2) throw std::invalid_argument("heuresis::benchmarks::dtlz7_front: m ≥ 2");
  const std::vector<double> T = running_records(dtlz7_phi, points, Extreme::greatest);
  std::vector<Objectives> out;
  std::vector<std::size_t> at(m - 1, 0);
  for (;;) {
    Objectives f(m);
    double last = 2 * static_cast<double>(m);
    for (std::size_t i = 0; i + 1 < m; i++) {
      f[i] = T[at[i]];
      last -= dtlz7_phi(f[i]);
    }
    f[m - 1] = last;
    out.push_back(std::move(f));
    std::size_t i = m - 1;  // the odometer: the last coordinate turns fastest
    while (i > 0) {
      if (++at[i - 1] < T.size()) break;
      at[--i] = 0;
    }
    if (i == 0) break;
  }
  return out;
}

// The exact hypervolumes of ZDT3's front with respect to (1, 1) and of DTLZ7's with respect to (1, …, 1, 2m + 1).
inline double zdt3_hypervolume(std::size_t cells = 1000000) { return 1 - running_extreme_integral(zdt3_curve, cells, Extreme::least); }
inline double dtlz7_hypervolume(std::size_t m = 3, std::size_t cells = 1000000) {
  if (m < 2) throw std::invalid_argument("heuresis::benchmarks::dtlz7_hypervolume: m ≥ 2");
  return 1 + static_cast<double>(m - 1) * running_extreme_integral(dtlz7_phi, cells, Extreme::greatest);
}

}  // namespace heuresis::benchmarks
