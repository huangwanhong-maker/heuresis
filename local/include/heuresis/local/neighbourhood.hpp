// Neighbourhoods: the moves a local search may make from a point.
//
// A neighbourhood N assigns to each point x a finite list of neighbours, the k-th written apply(x, k); with it
// the space is a graph, and a local optimum is a point no neighbour improves (Papadimitriou and Steiglitz 1982,
// Combinatorial Optimization: Algorithms and Complexity, ch. 19; Aarts and Lenstra 1997, Local Search in
// Combinatorial Optimization). Which graph is chosen decides which points are local optima. A move's
// attribute names what it changes (a bit's position, a pair of positions), for the tabu lists of tabu search.
//
// Provided: one bit flipped and two bits flipped on strings; on permutations, the reversal of a segment
// (2-opt, whose neighbours of a tour differ in two edges; Croes 1958, "A method for solving traveling-salesman
// problems", Oper. Res. 6, 791–812), the exchange of two positions, and the move of one element.

#pragma once

#include <heuresis/variation/binary.hpp>
#include <heuresis/variation/permutation.hpp>

#include <algorithm>
#include <functional>

namespace heuresis::local {

using heuresis::variation::Bits;
using heuresis::variation::Permutation;

template <class G>
struct Neighbourhood {
  std::function<std::size_t(const G&)> size;                         // the number of moves at x
  std::function<G(const G&, std::size_t)> apply;                     // the k-th neighbour of x
  std::function<std::size_t(const G&, std::size_t)> attribute = {};  // what the k-th move changes (tabu search)
};

inline Neighbourhood<Bits> one_flip() {
  return {[](const Bits& x) { return x.size(); },
          [](const Bits& x, std::size_t k) {
            Bits y = x;
            y[k] ^= 1u;
            return y;
          },
          [](const Bits&, std::size_t k) { return k; }};
}

namespace detail {
// The k-th pair i < j of {0, …, n − 1} in the order (0,1), (0,2), …, (1,2), …
inline std::pair<std::size_t, std::size_t> pair_of(std::size_t n, std::size_t k) {
  std::size_t i = 0;
  while (k >= n - 1 - i) {
    k -= n - 1 - i;
    i++;
  }
  return {i, i + 1 + k};
}
}  // namespace detail

inline Neighbourhood<Bits> two_flip() {
  return {[](const Bits& x) { return x.size() * (x.size() - 1) / 2; },
          [](const Bits& x, std::size_t k) {
            Bits y = x;
            auto [i, j] = detail::pair_of(x.size(), k);
            y[i] ^= 1u;
            y[j] ^= 1u;
            return y;
          },
          [](const Bits& x, std::size_t k) { return detail::pair_of(x.size(), k).first; }};
}

// Reversal of the segment [i, j], 0 ≤ i < j < n, except the whole tour (which is the same cycle).
inline Neighbourhood<Permutation> two_opt() {
  return {[](const Permutation& p) { return p.size() < 3 ? 0 : p.size() * (p.size() - 1) / 2 - 1; },
          [](const Permutation& p, std::size_t k) {
            auto [i, j] = detail::pair_of(p.size(), k + (k >= p.size() - 2 ? 1 : 0));  // skip (0, n − 1)
            Permutation q = p;
            std::reverse(q.begin() + static_cast<std::ptrdiff_t>(i), q.begin() + static_cast<std::ptrdiff_t>(j) + 1);
            return q;
          },
          [](const Permutation& p, std::size_t k) { return detail::pair_of(p.size(), k + (k >= p.size() - 2 ? 1 : 0)).first; }};
}

inline Neighbourhood<Permutation> exchanges() {
  return {[](const Permutation& p) { return p.size() * (p.size() - 1) / 2; },
          [](const Permutation& p, std::size_t k) {
            auto [i, j] = detail::pair_of(p.size(), k);
            Permutation q = p;
            std::swap(q[i], q[j]);
            return q;
          },
          [](const Permutation& p, std::size_t k) { return detail::pair_of(p.size(), k).first; }};
}

// Element at position i moved to position j ≠ i: n(n − 1) moves.
inline Neighbourhood<Permutation> insertions() {
  return {[](const Permutation& p) { return p.size() * (p.size() - 1); },
          [](const Permutation& p, std::size_t k) {
            const std::size_t n = p.size(), i = k / (n - 1);
            std::size_t j = k % (n - 1);
            if (j >= i) j++;
            Permutation q = p;
            const auto v = q[i];
            q.erase(q.begin() + static_cast<std::ptrdiff_t>(i));
            q.insert(q.begin() + static_cast<std::ptrdiff_t>(j), v);
            return q;
          },
          [](const Permutation& p, std::size_t k) { return static_cast<std::size_t>(p[k / (p.size() - 1)]); }};
}

}  // namespace heuresis::local
