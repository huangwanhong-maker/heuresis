// What a random walk sees of a landscape: its autocorrelation and correlation length.
//
// A landscape is a function f on the vertices of a graph, the space with its neighbourhood (Stadler 2002,
// "Fitness landscapes", in Biological Evolution and Statistical Physics, Springer, 183–204). Along a simple
// random walk x₀, x₁, … the values' autocorrelation
//   ρ(s) = Cov(f(x_t), f(x_{t+s})) / Var f
// says how fast f decorrelates with distance (Weinberger 1990, "Correlated and uncorrelated fitness landscapes
// and how to tell the difference", Biol. Cybern. 63, 325–336); its correlation length is ℓ = −1/ln|ρ(1)|. A
// rugged landscape decorrelates in a step, a smooth one slowly. On an elementary landscape (enumerated.hpp)
// ρ(s) = (1 − λ/d)ˢ exactly, with λ the eigenvalue of the graph Laplacian and d the degree (Stadler 1996,
// "Landscapes and their correlation functions", J. Math. Chem. 20, 1–45).
//
// The walk draws its steps from streams keyed by the step, so it is the same wherever it is taken.

#pragma once

#include <heuresis/search/draws.hpp>
#include <heuresis/search/evaluation.hpp>

#include <tyche/descriptive/summary.hpp>

#include <cmath>
#include <span>
#include <type_traits>
#include <vector>

namespace heuresis::landscapes {

namespace detail {
template <class R>
double value_of(const R& r) {
  if constexpr (std::is_convertible_v<R, heuresis::search::Evaluation>) return heuresis::search::Evaluation(r).value();
  else return static_cast<double>(r);
}
}  // namespace detail

// The values of f along `length` points of a walk from start; step(x, stream) is a random neighbour of x.
template <class G, class F, class Step>
std::vector<double> random_walk(G start, F&& f, Step&& step, std::size_t length, heuresis::search::Draws draws = heuresis::search::Draws{}) {
  constexpr std::uint32_t purpose = heuresis::search::tag("landscapes.walk.step");
  std::vector<double> v;
  v.reserve(length);
  G x = std::move(start);
  for (std::size_t t = 0; t < length; t++) {
    v.push_back(detail::value_of(f(x)));
    if (t + 1 == length) break;
    calx::random::Stream s = draws(purpose, t, 0);
    x = step(x, s);
  }
  return v;
}

// Weinberger's estimator of ρ(0), …, ρ(max_lag) from one walk.
inline std::vector<double> autocorrelation(std::span<const double> walk, std::size_t max_lag) {
  const std::size_t T = walk.size();
  if (T < 2 || max_lag >= T) throw std::invalid_argument("heuresis::landscapes::autocorrelation: the walk is too short for the lag");
  const double m = tyche::descriptive::mean(walk), v = tyche::descriptive::variance(walk, 0);
  if (!(v > 0)) throw std::invalid_argument("heuresis::landscapes::autocorrelation: a walk of constant value");
  std::vector<double> rho(max_lag + 1);
  for (std::size_t s = 0; s <= max_lag; s++) {
    double c = 0;
    for (std::size_t t = 0; t + s < T; t++) c += (walk[t] - m) * (walk[t + s] - m);
    rho[s] = c / static_cast<double>(T - s) / v;
  }
  return rho;
}

inline double correlation_length(double rho1) { return -1 / std::log(std::fabs(rho1)); }

// The correlation of values with distances to the nearest global optimum over a sample (Jones and Forrest 1995,
// "Fitness distance correlation as a measure of problem difficulty for genetic algorithms", ICGA 6, 184–192).
// With objectives minimised, r near +1 says the values fall towards the optimum (easy), r ≤ 0 that they mislead
// (deceptive); r is a summary, not a proof of difficulty (Altenberg 1997, "Fitness distance correlation analysis:
// an instructive counterexample", ICGA 7, 57–64).
inline double fitness_distance_correlation(std::span<const double> values, std::span<const double> distances) { return tyche::descriptive::correlation(values, distances); }

template <class G, class F, class D>
double fitness_distance_correlation(std::span<const G> sample, F&& f, D&& distance_to_optimum) {
  std::vector<double> v, d;
  for (const auto& x : sample) {
    v.push_back(detail::value_of(f(x)));
    d.push_back(static_cast<double>(distance_to_optimum(x)));
  }
  return fitness_distance_correlation(v, d);
}

}  // namespace heuresis::landscapes
