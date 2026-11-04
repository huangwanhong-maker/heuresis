// Choosing parents: which individuals of a population breed, and how often.
//
// Every selection here sees a population only through an order (a comparator on indices, by default Deb's rule
// on the evaluations, protocol.hpp) or through weights the caller computes from it. Selection by order alone —
// tournaments, truncation, ranking — is invariant under every strictly increasing transformation of the
// objective, so the search it drives is too (the theory note, §3); selection by weights is not, and says on
// which scale its fitness is read.
//
// - Tournament of size k with replacement: the best of k uniform draws. The individual of rank i (1 = best)
//   of N wins with probability ((N − i + 1)ᵏ − (N − i)ᵏ)/Nᵏ (Goldberg and Deb 1991, "A comparative analysis of
//   selection schemes used in genetic algorithms", FOGA 1, 69–93; Bäck 1994).
// - Truncation: the best μ.
// - Linear ranking with selective pressure s ∈ [1, 2]: P(rank i) = (2 − s)/N + 2(s − 1)(N − i)/(N(N − 1)) (Baker
//   1985, "Adaptive selection methods for genetic algorithms", ICGA 1, 101–111).
// - Exponential ranking with base c ∈ (0, 1): P(rank i) ∝ c^{i − 1} (Blickle and Thiele 1996, "A comparison of
//   selection schemes used in evolutionary algorithms", Evol. Comput. 4, 361–394).
// - Fitness-proportional selection from nonnegative weights: by roulette (independent draws) or by Baker's
//   stochastic universal sampling, one spin with equally spaced pointers, which gives each individual ⌊e⌋ or
//   ⌈e⌉ copies of its expected number e (Baker 1987, "Reducing bias and inefficiency in the selection algorithm",
//   ICGA 2, 14–21).
// - Boltzmann weights exp(−(fᵢ − f_min)/T), the selection of annealing.
// Ties in an order are broken by index, so every selection is a function of its stream.

#pragma once

#include <heuresis/search/protocol.hpp>

#include <calx/random/stream.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

namespace heuresis::selection {

using heuresis::search::Evaluation;

// A strict weak order on the indices of a population: less(i, j) when i is better than j.
using Less = std::function<bool(std::size_t, std::size_t)>;

inline Less by_evaluation(std::span<const Evaluation> e) {
  return [e](std::size_t i, std::size_t j) { return heuresis::search::better(e[i], e[j]); };
}

// The indices ordered best first, ties by index.
inline std::vector<std::size_t> order(std::size_t n, const Less& less) {
  std::vector<std::size_t> o(n);
  std::iota(o.begin(), o.end(), std::size_t{0});
  std::stable_sort(o.begin(), o.end(), [&](std::size_t a, std::size_t b) { return less(a, b); });
  return o;
}

// The winner of a tournament of k drawn with replacement; ties to the earlier index.
inline std::size_t tournament(std::size_t n, std::size_t k, const Less& less, calx::random::Stream& s) {
  if (n == 0 || k == 0) throw std::invalid_argument("heuresis::selection::tournament: an empty population or tournament");
  std::size_t best = s.below(n);
  for (std::size_t t = 1; t < k; t++) {
    const std::size_t c = s.below(n);
    if (less(c, best) || (!less(best, c) && c < best)) best = c;
  }
  return best;
}
inline std::size_t tournament(std::span<const Evaluation> e, std::size_t k, calx::random::Stream& s) { return tournament(e.size(), k, by_evaluation(e), s); }

// The probability that the individual of rank i (1 = best) of n wins a tournament of k with replacement.
inline double tournament_probability(std::size_t n, std::size_t k, std::size_t i) {
  const double N = static_cast<double>(n), a = N - static_cast<double>(i) + 1, K = static_cast<double>(k);
  return (std::pow(a, K) - std::pow(a - 1, K)) / std::pow(N, K);
}

inline std::vector<std::size_t> truncation(std::size_t n, std::size_t mu, const Less& less) {
  auto o = order(n, less);
  o.resize(std::min(mu, n));
  return o;
}
inline std::vector<std::size_t> truncation(std::span<const Evaluation> e, std::size_t mu) { return truncation(e.size(), mu, by_evaluation(e)); }

// The probabilities of the ranks 1..n (best first) under linear ranking with pressure s ∈ [1, 2].
inline std::vector<double> linear_ranking(std::size_t n, double pressure) {
  if (!(pressure >= 1 && pressure <= 2)) throw std::invalid_argument("heuresis::selection::linear_ranking: the pressure must lie in [1, 2]");
  std::vector<double> p(n);
  const double N = static_cast<double>(n);
  for (std::size_t i = 0; i < n; i++) p[i] = n == 1 ? 1 : (2 - pressure) / N + 2 * (pressure - 1) * (N - 1 - static_cast<double>(i)) / (N * (N - 1));
  return p;
}

// The probabilities of the ranks 1..n under exponential ranking with base c ∈ (0, 1).
inline std::vector<double> exponential_ranking(std::size_t n, double c) {
  if (!(c > 0 && c < 1)) throw std::invalid_argument("heuresis::selection::exponential_ranking: the base must lie in (0, 1)");
  std::vector<double> p(n);
  double z = 0;
  for (std::size_t i = 0; i < n; i++) z += (p[i] = std::pow(c, static_cast<double>(i)));
  for (auto& v : p) v /= z;
  return p;
}

// Boltzmann weights of values to be minimised at temperature T.
inline std::vector<double> boltzmann_weights(std::span<const double> f, double temperature) {
  if (!(temperature > 0)) throw std::invalid_argument("heuresis::selection::boltzmann_weights: the temperature must be positive");
  const double lo = *std::min_element(f.begin(), f.end());
  std::vector<double> w(f.size());
  for (std::size_t i = 0; i < f.size(); i++) w[i] = std::exp(-(f[i] - lo) / temperature);
  return w;
}

namespace detail {
inline std::vector<double> cumulative(std::span<const double> w) {
  std::vector<double> c(w.size());
  double t = 0;
  for (std::size_t i = 0; i < w.size(); i++) {
    if (!(w[i] >= 0) || !std::isfinite(w[i])) throw std::invalid_argument("heuresis::selection: weights must be finite and nonnegative");
    c[i] = (t += w[i]);
  }
  if (!(t > 0)) throw std::invalid_argument("heuresis::selection: the weights sum to zero");
  return c;
}
}  // namespace detail

// One draw proportional to the weights.
inline std::size_t roulette(std::span<const double> weights, calx::random::Stream& s) {
  const auto c = detail::cumulative(weights);
  const double u = s.uniform() * c.back();
  return static_cast<std::size_t>(std::upper_bound(c.begin(), c.end(), u) - c.begin());
}

// count draws by stochastic universal sampling, in pointer order.
inline std::vector<std::size_t> universal_sampling(std::span<const double> weights, std::size_t count, calx::random::Stream& s) {
  const auto c = detail::cumulative(weights);
  std::vector<std::size_t> out;
  out.reserve(count);
  const double step = c.back() / static_cast<double>(count), start = s.uniform() * step;
  std::size_t i = 0;
  for (std::size_t k = 0; k < count; k++) {
    const double pointer = start + static_cast<double>(k) * step;
    while (i + 1 < c.size() && c[i] <= pointer) i++;
    out.push_back(i);
  }
  return out;
}

// One rank (0 = best) drawn from rank probabilities, mapped through an order to an individual.
inline std::size_t by_rank(std::span<const std::size_t> ordered, std::span<const double> rank_probability, calx::random::Stream& s) {
  return ordered[roulette(rank_probability, s)];
}

}  // namespace heuresis::selection
