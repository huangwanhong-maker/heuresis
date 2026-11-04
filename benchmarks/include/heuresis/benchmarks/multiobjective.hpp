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

#pragma once

#include <cmath>
#include <numbers>
#include <span>
#include <stdexcept>
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

}  // namespace heuresis::benchmarks
