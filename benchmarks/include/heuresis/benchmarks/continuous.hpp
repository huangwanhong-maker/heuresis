// Functions of real vectors with known minima.
//
// The classical test functions of continuous optimisation, each with its minimum value and a minimiser (Jamil
// and Yang 2013, "A literature survey of benchmark functions for global optimisation problems", Int. J. Math.
// Model. Numer. Optim. 4, 150–194; Surjanovic and Bingham, Virtual Library of Simulation Experiments, Simon Fraser
// University, for the constants of Branin, the six-hump camel and Hartmann's functions). Separable functions
// become non-separable under a rotation; a shift moves the optimum off the centre, where some operators are
// biased (Hansen, Finck, Ros and Auger 2009, "Real-parameter black-box optimization benchmarking 2009:
// noiseless functions definitions", INRIA RR-6829, which defines BBOB's transformations).
//
// | function        | f*                       | x*                                            |
// |-----------------|--------------------------|-----------------------------------------------|
// | sphere          | 0                        | 0                                             |
// | ellipsoid       | 0                        | 0 (condition 10⁶)                             |
// | rosenbrock      | 0                        | (1, …, 1)                                     |
// | rastrigin       | 0                        | 0                                             |
// | ackley          | 0                        | 0                                             |
// | griewank        | 0                        | 0                                             |
// | schwefel        | ≈ 0 (below 10⁻⁸ n)       | 420.968746… in each coordinate                |
// | styblinski_tang | −39.16616570377142 n     | −2.903534027771178 in each coordinate         |
// | levy            | 0                        | (1, …, 1)                                     |
// | branin          | 0.397887357729738        | (−π, 12.275), (π, 2.275), (9.42477796, 2.475) |
// | six_hump_camel  | −1.031628453489877       | ±(0.08984201368301331, −0.7126564032704135)   |
// | hartmann3       | −3.86278214782076        | (0.114614, 0.555649, 0.852547)                |
// | hartmann6       | −3.32236801141551        | (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573) |
//
// Constrained: Rosenbrock on the disc x₁² + x₂² ≤ 2 (f* = 0 at (1, 1), on the boundary), and G06 of Floudas and
// Pardalos 1990, f* = −6961.81387558015 at (14.095, 0.84296078921547957), both constraints active.

#pragma once

#include <heuresis/search/draws.hpp>
#include <heuresis/search/evaluation.hpp>

#include <calx/linalg/qr.hpp>
#include <calx/linalg/rectangular.hpp>

#include <cmath>
#include <functional>
#include <numbers>
#include <optional>
#include <span>
#include <vector>

namespace heuresis::benchmarks {

inline double sphere(std::span<const double> x) {
  double s = 0;
  for (double v : x) s += v * v;
  return s;
}

inline double ellipsoid(std::span<const double> x) {
  const std::size_t n = x.size();
  double s = 0;
  for (std::size_t i = 0; i < n; i++) s += std::pow(10.0, n > 1 ? 6.0 * static_cast<double>(i) / static_cast<double>(n - 1) : 0.0) * x[i] * x[i];
  return s;
}

inline double rosenbrock(std::span<const double> x) {
  double s = 0;
  for (std::size_t i = 0; i + 1 < x.size(); i++) s += 100 * (x[i + 1] - x[i] * x[i]) * (x[i + 1] - x[i] * x[i]) + (1 - x[i]) * (1 - x[i]);
  return s;
}

inline double rastrigin(std::span<const double> x) {
  double s = 10.0 * static_cast<double>(x.size());
  for (double v : x) s += v * v - 10 * std::cos(2 * std::numbers::pi * v);
  return s;
}

inline double ackley(std::span<const double> x) {
  const double n = static_cast<double>(x.size());
  double a = 0, b = 0;
  for (double v : x) { a += v * v; b += std::cos(2 * std::numbers::pi * v); }
  return -20 * std::exp(-0.2 * std::sqrt(a / n)) - std::exp(b / n) + 20 + std::numbers::e;
}

inline double griewank(std::span<const double> x) {
  double s = 0, p = 1;
  for (std::size_t i = 0; i < x.size(); i++) { s += x[i] * x[i] / 4000; p *= std::cos(x[i] / std::sqrt(static_cast<double>(i + 1))); }
  return 1 + s - p;
}

inline double schwefel(std::span<const double> x) {
  double s = 418.9828872724338 * static_cast<double>(x.size());
  for (double v : x) s -= v * std::sin(std::sqrt(std::fabs(v)));
  return s;
}

inline double styblinski_tang(std::span<const double> x) {
  double s = 0;
  for (double v : x) s += v * v * v * v - 16 * v * v + 5 * v;
  return s / 2;
}

inline double levy(std::span<const double> x) {
  const std::size_t n = x.size();
  auto w = [&](std::size_t i) { return 1 + (x[i] - 1) / 4; };
  const double pi = std::numbers::pi;
  double s = std::pow(std::sin(pi * w(0)), 2);
  for (std::size_t i = 0; i + 1 < n; i++) s += (w(i) - 1) * (w(i) - 1) * (1 + 10 * std::pow(std::sin(pi * w(i) + 1), 2));
  s += (w(n - 1) - 1) * (w(n - 1) - 1) * (1 + std::pow(std::sin(2 * pi * w(n - 1)), 2));
  return s;
}

inline double branin(std::span<const double> x) {
  const double pi = std::numbers::pi, b = 5.1 / (4 * pi * pi), c = 5 / pi, t = 1 / (8 * pi);
  const double u = x[1] - b * x[0] * x[0] + c * x[0] - 6;
  return u * u + 10 * (1 - t) * std::cos(x[0]) + 10;
}

inline double six_hump_camel(std::span<const double> x) {
  const double a = x[0], b = x[1];
  return (4 - 2.1 * a * a + a * a * a * a / 3) * a * a + a * b + (-4 + 4 * b * b) * b * b;
}

namespace detail {
inline double hartmann(std::span<const double> x, const double* A, const double* P, std::size_t d) {
  constexpr double alpha[4] = {1.0, 1.2, 3.0, 3.2};
  double s = 0;
  for (std::size_t i = 0; i < 4; i++) {
    double e = 0;
    for (std::size_t j = 0; j < d; j++) e += A[i * d + j] * (x[j] - P[i * d + j]) * (x[j] - P[i * d + j]);
    s -= alpha[i] * std::exp(-e);
  }
  return s;
}
}  // namespace detail

inline double hartmann3(std::span<const double> x) {
  static constexpr double A[12] = {3, 10, 30, 0.1, 10, 35, 3, 10, 30, 0.1, 10, 35};
  static constexpr double P[12] = {0.3689, 0.1170, 0.2673, 0.4699, 0.4387, 0.7470, 0.1091, 0.8732, 0.5547, 0.03815, 0.5743, 0.8828};
  return detail::hartmann(x, A, P, 3);
}

inline double hartmann6(std::span<const double> x) {
  static constexpr double A[24] = {10, 3, 17, 3.5, 1.7, 8, 0.05, 10, 17, 0.1, 8, 14, 3, 3.5, 1.7, 10, 17, 8, 17, 8, 0.05, 10, 0.1, 14};
  static constexpr double P[24] = {0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886, 0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991,
                                   0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650, 0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381};
  return detail::hartmann(x, A, P, 6);
}

// Rosenbrock on the disc x₁² + x₂² ≤ 2.
inline heuresis::search::Evaluation disc_rosenbrock(std::span<const double> x) {
  return heuresis::search::evaluation({rosenbrock(x)}, {x[0] * x[0] + x[1] * x[1] - 2});
}

// G06 (Floudas and Pardalos 1990, problem 4.6; Michalewicz and Schoenauer 1996); bounds 13 ≤ x₁ ≤ 100, 0 ≤ x₂ ≤ 100.
inline heuresis::search::Evaluation g06(std::span<const double> x) {
  const double a = x[0] - 10, b = x[1] - 20;
  return heuresis::search::evaluation({a * a * a + b * b * b}, {-(x[0] - 5) * (x[0] - 5) - (x[1] - 5) * (x[1] - 5) + 100, (x[0] - 6) * (x[0] - 6) + (x[1] - 5) * (x[1] - 5) - 82.81});
}

// A rotation of ℝⁿ drawn from the Haar measure on O(n): the Q of a Gaussian matrix's QR factorisation, its
// columns signed by R's diagonal (Mezzadri 2007, "How to generate random matrices from the classical compact
// groups", Notices AMS 54, 592–604).
class Rotation {
 public:
  Rotation(std::size_t n, std::uint64_t seed, std::uint64_t instance = 0) : n_(n), r_(n * n) {
    calx::random::Stream s = heuresis::search::Draws(seed)(heuresis::search::tag("benchmarks.rotation"), instance, n);
    calx::linalg::Rectangular a(n, n);
    for (auto& v : a.a) v = s.normal();
    const auto f = calx::linalg::qr(a, false);
    const auto q = calx::linalg::thin_q(f);
    const auto r = calx::linalg::r_factor(f);
    for (std::size_t i = 0; i < n; i++)
      for (std::size_t j = 0; j < n; j++) r_[i * n + j] = q(i, j) * (r(j, j) < 0 ? -1 : 1);
  }
  std::vector<double> apply(std::span<const double> x) const {
    std::vector<double> y(n_, 0.0);
    for (std::size_t i = 0; i < n_; i++)
      for (std::size_t j = 0; j < n_; j++) y[i] += r_[i * n_ + j] * x[j];
    return y;
  }
  double operator()(std::size_t i, std::size_t j) const { return r_[i * n_ + j]; }
  std::size_t size() const { return n_; }

 private:
  std::size_t n_;
  std::vector<double> r_;
};

// x ↦ f(R(x − shift)), the shifted and rotated form of f; its minimiser is shift + Rᵀ x*.
inline std::function<double(std::span<const double>)> transformed(std::function<double(std::span<const double>)> f, std::vector<double> shift, std::optional<Rotation> rotation = std::nullopt) {
  return [f = std::move(f), shift = std::move(shift), rotation = std::move(rotation)](std::span<const double> x) {
    std::vector<double> y(x.begin(), x.end());
    if (!shift.empty())
      for (std::size_t i = 0; i < y.size(); i++) y[i] -= shift[i];
    if (rotation) y = rotation->apply(y);
    return f(y);
  };
}

}  // namespace heuresis::benchmarks
