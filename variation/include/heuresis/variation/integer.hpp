// Integer vectors in a box, and their variation.
//
// - Uniform reset: each coordinate redrawn uniformly in its bounds with probability `rate`.
// - Creep: each coordinate moved by ±1 … ±step with probability `rate`, clipped to its bounds.
// - Rudolph's two-sided geometric mutation: each coordinate moved by G₁ − G₂, the difference of two
//   geometric variables (failures before a success of probability p), the discrete distribution of maximal
//   entropy with a given mean absolute step; for a mean step σ of the whole vector in the ℓ¹ norm,
//   p = 1 − (σ/n)/(1 + √(1 + (σ/n)²)) (Rudolph 1994, "An evolutionary algorithm for integer programming",
//   PPSN III, 139–148, §3), so that E|G₁ − G₂| = 2(1 − p)/(1 − (1 − p)²) = σ/n.

#pragma once

#include <calx/random/stream.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace heuresis::variation {

using Integers = std::vector<std::int64_t>;

struct IntegerBox {
  std::vector<std::int64_t> lower, upper;
  static IntegerBox cube(std::size_t n, std::int64_t lo, std::int64_t hi) { return {std::vector<std::int64_t>(n, lo), std::vector<std::int64_t>(n, hi)}; }
  std::size_t size() const { return lower.size(); }
  bool contains(const Integers& x) const {
    if (x.size() != size()) return false;
    for (std::size_t i = 0; i < x.size(); i++)
      if (x[i] < lower[i] || x[i] > upper[i]) return false;
    return true;
  }
};

inline std::int64_t uniform_integer(std::int64_t lo, std::int64_t hi, calx::random::Stream& s) {
  return lo + static_cast<std::int64_t>(s.below(static_cast<std::uint64_t>(hi - lo) + 1));
}

inline Integers uniform_integers(const IntegerBox& b, calx::random::Stream& s) {
  Integers x(b.size());
  for (std::size_t i = 0; i < x.size(); i++) x[i] = uniform_integer(b.lower[i], b.upper[i], s);
  return x;
}

inline Integers reset_mutation(const Integers& x, const IntegerBox& b, double rate, calx::random::Stream& s) {
  Integers y = x;
  for (std::size_t i = 0; i < y.size(); i++)
    if (s.uniform() < rate) y[i] = uniform_integer(b.lower[i], b.upper[i], s);
  return y;
}

inline Integers creep_mutation(const Integers& x, const IntegerBox& b, std::int64_t step, double rate, calx::random::Stream& s) {
  if (step < 1) throw std::invalid_argument("heuresis::variation::creep_mutation: a step below 1");
  Integers y = x;
  for (std::size_t i = 0; i < y.size(); i++) {
    if (!(s.uniform() < rate)) continue;
    const std::int64_t d = 1 + static_cast<std::int64_t>(s.below(static_cast<std::uint64_t>(step)));
    y[i] = std::clamp(y[i] + (s.uniform() < 0.5 ? -d : d), b.lower[i], b.upper[i]);
  }
  return y;
}

// The success probability p of Rudolph's geometric variables for a mean ℓ¹ step sigma over n coordinates.
inline double rudolph_p(double sigma, std::size_t n) {
  const double r = sigma / static_cast<double>(n);
  return 1 - r / (1 + std::sqrt(1 + r * r));
}

inline Integers geometric_mutation(const Integers& x, const IntegerBox& b, double sigma, calx::random::Stream& s) {
  const double p = rudolph_p(sigma, x.size());
  Integers y = x;
  for (std::size_t i = 0; i < y.size(); i++) {
    const auto g1 = static_cast<std::int64_t>(s.geometric(p)), g2 = static_cast<std::int64_t>(s.geometric(p));
    y[i] = std::clamp(y[i] + g1 - g2, b.lower[i], b.upper[i]);
  }
  return y;
}

}  // namespace heuresis::variation
