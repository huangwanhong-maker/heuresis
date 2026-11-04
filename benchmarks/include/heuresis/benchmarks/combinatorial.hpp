// Problems on permutations, and a repertoire problem for the search for diversity.
//
// - The symmetric Euclidean travelling salesman: n points uniform in the unit square, a tour's length (Applegate,
//   Bixby, Chvátal and Cook 2006, The Traveling Salesman Problem: A Computational Study). For n points uniform in
//   the unit square the optimal length grows as β√n with β ≈ 0.7124 (Beardwood, Halton and Hammersley 1959;
//   the constant from Applegate et al. 2006, §2.1).
// - The planar arm (Cully, Clune, Tarapore and Mouret 2015, "Robots that can adapt like animals", Nature 521,
//   503–507, Methods; Vassiliades, Chatzilygeroudis and Mouret 2018, "Using centroidal Voronoi tessellations to
//   scale up the multidimensional archive of phenotypic elites algorithm", IEEE Trans. Evol. Comput. 22,
//   623–630, §IV): d links of length 1/d, angles θᵢ ∈ [−π, π]; the descriptor is the position of its tip mapped
//   into [0, 1]², the objective the variance of the angles (the smoothest arm reaching each place). Every point of
//   the disc of radius 1 is reachable, so a full repertoire covers the disc; it names no machine, only a chain.

#pragma once

#include <heuresis/search/draws.hpp>
#include <heuresis/search/evaluation.hpp>
#include <heuresis/variation/permutation.hpp>

#include <cmath>
#include <numbers>
#include <span>
#include <vector>

namespace heuresis::benchmarks {

class Tour {
 public:
  explicit Tour(std::vector<std::pair<double, double>> points) : points_(std::move(points)) {}
  static Tour random(std::size_t n, std::uint64_t seed, std::uint64_t instance = 0) {
    calx::random::Stream s = heuresis::search::Draws(seed)(heuresis::search::tag("benchmarks.tour"), instance, n);
    std::vector<std::pair<double, double>> p(n);
    for (auto& q : p) q = {s.uniform(), s.uniform()};
    return Tour(std::move(p));
  }
  double distance(std::size_t a, std::size_t b) const { return std::hypot(points_[a].first - points_[b].first, points_[a].second - points_[b].second); }
  double length(const heuresis::variation::Permutation& p) const {
    double L = 0;
    for (std::size_t i = 0; i < p.size(); i++) L += distance(p[i], p[(i + 1) % p.size()]);
    return L;
  }
  double operator()(const heuresis::variation::Permutation& p) const { return length(p); }
  std::size_t size() const { return points_.size(); }
  const std::vector<std::pair<double, double>>& points() const { return points_; }

 private:
  std::vector<std::pair<double, double>> points_;
};

inline heuresis::search::Evaluation planar_arm(std::span<const double> theta) {
  const double d = static_cast<double>(theta.size()), L = 1 / d;
  double x = 0, y = 0, a = 0, mean = 0;
  for (double t : theta) {
    a += t;
    x += L * std::cos(a);
    y += L * std::sin(a);
    mean += t;
  }
  mean /= d;
  double var = 0;
  for (double t : theta) var += (t - mean) * (t - mean);
  return heuresis::search::evaluation({var / d}, {}, {(x + 1) / 2, (y + 1) / 2});
}

}  // namespace heuresis::benchmarks
