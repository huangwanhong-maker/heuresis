// Permutations and their variation.
//
// The space is the symmetric group Sₙ; a permutation π is stored as the sequence (π(0), …, π(n − 1)).
// A mutation is a step of a random walk on a Cayley graph of Sₙ: a swap multiplies by a transposition, an
// inversion (2-opt) reverses a segment, an insertion moves one element, a scramble shuffles a segment. Which
// generators to use is which neighbourhood the problem's structure is smooth on: adjacency (tours) under
// inversion, absolute position (assignments) under swaps, relative order (schedules) under insertion.
//
// Crossovers, each giving two children that are permutations, keep a different property of the parents:
// - PMX keeps absolute positions in a segment (Goldberg and Lingle 1985, "Alleles, loci, and the travelling
//   salesman problem", ICGA 1, 154–159);
// - OX1 keeps a segment in place and the relative order of the rest (Davis 1985, "Applying adaptive algorithms
//   to epistatic domains", IJCAI 1, 162–164);
// - CX keeps every position: each position of a child holds the value of one parent at that position (Oliver,
//   Smith and Holland 1987, "A study of permutation crossover operators on the TSP", ICGA 2, 224–230).

#pragma once

#include <calx/random/stream.hpp>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace heuresis::variation {

using Permutation = std::vector<std::uint32_t>;

inline Permutation identity_permutation(std::size_t n) {
  Permutation p(n);
  std::iota(p.begin(), p.end(), 0u);
  return p;
}

inline bool is_permutation(const Permutation& p) {
  std::vector<char> seen(p.size(), 0);
  for (auto v : p) {
    if (v >= p.size() || seen[v]) return false;
    seen[v] = 1;
  }
  return true;
}

// Fisher and Yates's shuffle of positions [b, e) (Knuth, TAOCP 2, §3.4.2, Algorithm P).
inline void shuffle(Permutation& p, std::size_t b, std::size_t e, calx::random::Stream& s) {
  for (std::size_t i = e; i > b + 1; i--) std::swap(p[i - 1], p[b + s.below(i - b)]);
}

inline Permutation random_permutation(std::size_t n, calx::random::Stream& s) {
  Permutation p = identity_permutation(n);
  shuffle(p, 0, n, s);
  return p;
}

inline Permutation inverse(const Permutation& p) {
  Permutation q(p.size());
  for (std::size_t i = 0; i < p.size(); i++) q[p[i]] = static_cast<std::uint32_t>(i);
  return q;
}

namespace detail {
// Two distinct positions i < j.
inline std::pair<std::size_t, std::size_t> two_positions(std::size_t n, calx::random::Stream& s) {
  std::size_t i = s.below(n), j = s.below(n - 1);
  if (j >= i) j++;
  return {std::min(i, j), std::max(i, j)};
}
}  // namespace detail

inline Permutation swap_mutation(const Permutation& p, calx::random::Stream& s) {
  Permutation q = p;
  if (q.size() < 2) return q;
  auto [i, j] = detail::two_positions(q.size(), s);
  std::swap(q[i], q[j]);
  return q;
}

// Reverses the segment [i, j], i < j: the 2-opt move of a tour.
inline Permutation inversion_mutation(const Permutation& p, calx::random::Stream& s) {
  Permutation q = p;
  if (q.size() < 2) return q;
  auto [i, j] = detail::two_positions(q.size(), s);
  std::reverse(q.begin() + static_cast<std::ptrdiff_t>(i), q.begin() + static_cast<std::ptrdiff_t>(j) + 1);
  return q;
}

// Removes the element at one position and inserts it at another.
inline Permutation insertion_mutation(const Permutation& p, calx::random::Stream& s) {
  Permutation q = p;
  if (q.size() < 2) return q;
  const std::size_t from = s.below(q.size());
  std::size_t to = s.below(q.size() - 1);
  if (to >= from) to++;
  const auto v = q[from];
  q.erase(q.begin() + static_cast<std::ptrdiff_t>(from));
  q.insert(q.begin() + static_cast<std::ptrdiff_t>(to), v);
  return q;
}

inline Permutation scramble_mutation(const Permutation& p, calx::random::Stream& s) {
  Permutation q = p;
  if (q.size() < 2) return q;
  auto [i, j] = detail::two_positions(q.size(), s);
  shuffle(q, i, j + 1, s);
  return q;
}

namespace detail {
inline Permutation pmx_child(const Permutation& keep, const Permutation& other, std::size_t a, std::size_t b) {
  const std::size_t n = keep.size();
  Permutation c(n);
  std::vector<std::uint32_t> where(n);  // position of each value in `keep`
  std::vector<char> in_segment(n, 0);
  for (std::size_t i = 0; i < n; i++) where[keep[i]] = static_cast<std::uint32_t>(i);
  for (std::size_t i = a; i <= b; i++) { c[i] = keep[i]; in_segment[keep[i]] = 1; }
  for (std::size_t i = 0; i < n; i++) {
    if (i >= a && i <= b) continue;
    std::uint32_t v = other[i];
    while (in_segment[v]) v = other[where[v]];
    c[i] = v;
  }
  return c;
}
inline Permutation ox_child(const Permutation& keep, const Permutation& other, std::size_t a, std::size_t b) {
  const std::size_t n = keep.size();
  Permutation c(n);
  std::vector<char> used(n, 0);
  for (std::size_t i = a; i <= b; i++) { c[i] = keep[i]; used[keep[i]] = 1; }
  std::size_t pos = (b + 1) % n;
  for (std::size_t k = 0; k < n; k++) {
    const std::uint32_t v = other[(b + 1 + k) % n];
    if (used[v]) continue;
    c[pos] = v;
    used[v] = 1;
    pos = (pos + 1) % n;
  }
  return c;
}
inline void same(const Permutation& a, const Permutation& b, const char* who) {
  if (a.size() != b.size() || !is_permutation(a) || !is_permutation(b)) throw std::invalid_argument(std::string("heuresis::variation::") + who + ": parents must be permutations of one order");
}
}  // namespace detail

inline std::pair<Permutation, Permutation> partially_mapped_crossover(const Permutation& p1, const Permutation& p2, calx::random::Stream& s) {
  detail::same(p1, p2, "partially_mapped_crossover");
  if (p1.size() < 2) return {p1, p2};
  auto [a, b] = detail::two_positions(p1.size(), s);
  return {detail::pmx_child(p1, p2, a, b), detail::pmx_child(p2, p1, a, b)};
}

inline std::pair<Permutation, Permutation> order_crossover(const Permutation& p1, const Permutation& p2, calx::random::Stream& s) {
  detail::same(p1, p2, "order_crossover");
  if (p1.size() < 2) return {p1, p2};
  auto [a, b] = detail::two_positions(p1.size(), s);
  return {detail::ox_child(p1, p2, a, b), detail::ox_child(p2, p1, a, b)};
}

// Cycle crossover: the cycles of the pair alternate between the parents; the stream decides which parent the
// first cycle comes from.
inline std::pair<Permutation, Permutation> cycle_crossover(const Permutation& p1, const Permutation& p2, calx::random::Stream& s) {
  detail::same(p1, p2, "cycle_crossover");
  const std::size_t n = p1.size();
  const Permutation where1 = inverse(p1);
  Permutation c1(n), c2(n);
  std::vector<char> done(n, 0);
  bool from_first = s.uniform() < 0.5;
  for (std::size_t start = 0; start < n; start++) {
    if (done[start]) continue;
    for (std::size_t i = start; !done[i]; i = where1[p2[i]]) {
      done[i] = 1;
      c1[i] = from_first ? p1[i] : p2[i];
      c2[i] = from_first ? p2[i] : p1[i];
    }
    from_first = !from_first;
  }
  return {c1, c2};
}

}  // namespace heuresis::variation
