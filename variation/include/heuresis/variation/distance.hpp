// Distances on the spaces, for diversity, niching and the analysis of landscapes.
//
// - Hamming distance on sequences: the graph distance of the hypercube.
// - Euclidean distance on real vectors.
// - Kendall's distance on permutations: the number of discordant pairs, the graph distance of Sₙ's Cayley graph
//   under adjacent transpositions (the inversions of π₂ ∘ π₁⁻¹), counted in O(n log n) by a Fenwick tree.
// - Cayley's distance: the least number of transpositions taking one permutation to the other, n minus the
//   number of cycles of π₂ ∘ π₁⁻¹ (Diaconis 1988, Group Representations in Probability and Statistics, ch. 6).
// Both distances on permutations are invariant under the group acting on either side, so they measure
// differences of order, not of labels.

#pragma once

#include <heuresis/variation/permutation.hpp>

#include <cmath>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace heuresis::variation {

template <class T>
std::size_t hamming(const std::vector<T>& a, const std::vector<T>& b) {
  if (a.size() != b.size()) throw std::invalid_argument("heuresis::variation::hamming: sequences of different lengths");
  std::size_t d = 0;
  for (std::size_t i = 0; i < a.size(); i++) d += a[i] != b[i];
  return d;
}

inline double euclidean(std::span<const double> a, std::span<const double> b) {
  if (a.size() != b.size()) throw std::invalid_argument("heuresis::variation::euclidean: vectors of different lengths");
  double s = 0;
  for (std::size_t i = 0; i < a.size(); i++) s += (a[i] - b[i]) * (a[i] - b[i]);
  return std::sqrt(s);
}

// The number of pairs i < j ordered differently by p and q.
inline std::uint64_t kendall_distance(const Permutation& p, const Permutation& q) {
  if (p.size() != q.size()) throw std::invalid_argument("heuresis::variation::kendall_distance: permutations of different orders");
  const std::size_t n = p.size();
  const Permutation pos_q = inverse(q);
  std::vector<std::uint32_t> tree(n + 1, 0);
  std::uint64_t inv = 0;
  // Read p's values in order; each is at position pos_q[v] in q; count earlier ones placed after it in q.
  for (std::size_t k = 0; k < n; k++) {
    const std::size_t r = pos_q[p[k]] + 1;
    std::uint64_t not_after = 0;
    for (std::size_t i = r; i > 0; i -= i & (~i + 1)) not_after += tree[i];
    inv += k - not_after;
    for (std::size_t i = r; i <= n; i += i & (~i + 1)) tree[i]++;
  }
  return inv;
}

inline std::size_t cayley_distance(const Permutation& p, const Permutation& q) {
  if (p.size() != q.size()) throw std::invalid_argument("heuresis::variation::cayley_distance: permutations of different orders");
  const std::size_t n = p.size();
  const Permutation pinv = inverse(p);
  std::vector<char> seen(n, 0);
  std::size_t cycles = 0;
  for (std::size_t i = 0; i < n; i++) {
    if (seen[i]) continue;
    cycles++;
    for (std::size_t j = i; !seen[j]; j = pinv[q[j]]) seen[j] = 1;
  }
  return n - cycles;
}

}  // namespace heuresis::variation
