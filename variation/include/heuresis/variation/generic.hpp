// Crossover of any two sequences of equal length, position by position.
//
// A crossover of two parents gives two children, the pair (c₁, c₂): for the operators here c₂ is the complement
// of c₁, taking at each position the parent c₁ did not. Cut points and masks are drawn from the stream only,
// never from the parents' values.
//
// Symmetry (decision 0003 §2.7, the theory note §4). Uniform crossover commutes with every permutation of the
// positions; k-point crossover only with the reversal of positions, since it keeps adjacent positions together
// (it assumes linkage): which operator to use is a statement about which symmetries the problem has. Both
// commute with any relabelling of the values that acts position by position (for bits, x ↦ x ⊕ m).

#pragma once

#include <calx/random/stream.hpp>

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

namespace heuresis::variation {

namespace detail {
template <class T>
void same_length(const std::vector<T>& a, const std::vector<T>& b, const char* who) {
  if (a.size() != b.size()) throw std::invalid_argument(std::string("heuresis::variation::") + who + ": parents of different lengths");
}
}  // namespace detail

// Each position is exchanged with probability `rate` (Syswerda 1989, "Uniform crossover in genetic
// algorithms", ICGA 3, 2–9).
template <class T>
std::pair<std::vector<T>, std::vector<T>> uniform_crossover(const std::vector<T>& a, const std::vector<T>& b, calx::random::Stream& s, double rate = 0.5) {
  detail::same_length(a, b, "uniform_crossover");
  std::pair<std::vector<T>, std::vector<T>> c{a, b};
  for (std::size_t i = 0; i < a.size(); i++)
    if (s.uniform() < rate) std::swap(c.first[i], c.second[i]);
  return c;
}

// k cut points drawn without replacement from the n − 1 places between positions; the parents' segments
// alternate between the children (De Jong 1975, An Analysis of the Behavior of a Class of Genetic Adaptive
// Systems, PhD thesis, University of Michigan). k = 1 is one-point crossover (Holland 1975).
template <class T>
std::pair<std::vector<T>, std::vector<T>> k_point_crossover(const std::vector<T>& a, const std::vector<T>& b, std::size_t k, calx::random::Stream& s) {
  detail::same_length(a, b, "k_point_crossover");
  const std::size_t n = a.size();
  std::pair<std::vector<T>, std::vector<T>> c{a, b};
  if (n < 2 || k == 0) return c;
  k = std::min(k, n - 1);
  // Floyd's sampling of k distinct places in {1, …, n − 1} (Bentley and Floyd 1987, "A sample of
  // brilliance", CACM 30, 754–757).
  std::vector<std::size_t> cuts;
  for (std::size_t j = n - 1 - k; j < n - 1; j++) {
    const std::size_t t = 1 + s.below(j + 1);
    if (std::find(cuts.begin(), cuts.end(), t) == cuts.end()) cuts.push_back(t);
    else cuts.push_back(j + 1);
  }
  std::sort(cuts.begin(), cuts.end());
  bool swap = false;
  std::size_t next = 0;
  for (std::size_t i = 0; i < n; i++) {
    while (next < cuts.size() && cuts[next] == i) { swap = !swap; next++; }
    if (swap) std::swap(c.first[i], c.second[i]);
  }
  return c;
}

template <class T>
std::pair<std::vector<T>, std::vector<T>> one_point_crossover(const std::vector<T>& a, const std::vector<T>& b, calx::random::Stream& s) { return k_point_crossover(a, b, 1, s); }

template <class T>
std::pair<std::vector<T>, std::vector<T>> two_point_crossover(const std::vector<T>& a, const std::vector<T>& b, calx::random::Stream& s) { return k_point_crossover(a, b, 2, s); }

}  // namespace heuresis::variation
