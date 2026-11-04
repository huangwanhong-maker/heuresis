// 2-opt on tours, with the change of length computed from the four edges a move touches.
//
// A tour is a cyclic permutation; reversing its segment from position i + 1 to j replaces the edges (tᵢ, tᵢ₊₁)
// and (tⱼ, tⱼ₊₁) by (tᵢ, tⱼ) and (tᵢ₊₁, tⱼ₊₁), so the change of length is known from four distances in O(1)
// rather than from the whole tour in O(n) (Croes 1958; Lin 1965, "Computer solutions of the traveling salesman
// problem", Bell Syst. Tech. J. 44, 2245–2269). This is a white-box local search: it needs the distances, not
// a black-box objective, and is offered as a function, not as an ask–tell algorithm, for use inside an
// evaluation (a memetic algorithm) or as a kick's partner in iterated local search.
//
// In the Euclidean plane a 2-opt local optimum has no two edges that cross, since uncrossing two crossing edges
// shortens the tour by the triangle inequality (Flood 1956, "The traveling-salesman problem", Oper. Res. 4,
// 61–75); the checks hold the result to that.

#pragma once

#include <heuresis/variation/permutation.hpp>

#include <algorithm>
#include <cstdint>

namespace heuresis::local {

// Improving 2-opt moves, first improvement in passes over (i, j), until none improves by more than `epsilon`;
// returns the number of moves made.
template <class Distance>
std::size_t two_opt_descent(heuresis::variation::Permutation& tour, Distance&& d, double epsilon = 1e-12) {
  const std::size_t n = tour.size();
  if (n < 4) return 0;
  std::size_t moves = 0;
  for (bool improved = true; improved;) {
    improved = false;
    for (std::size_t i = 0; i + 2 < n; i++)
      for (std::size_t j = i + 2; j < n; j++) {
        if (i == 0 && j == n - 1) continue;  // the same two edges
        const auto a = tour[i], b = tour[i + 1], c = tour[j], e = tour[(j + 1) % n];
        const double delta = d(a, c) + d(b, e) - d(a, b) - d(c, e);
        if (delta < -epsilon) {
          std::reverse(tour.begin() + static_cast<std::ptrdiff_t>(i) + 1, tour.begin() + static_cast<std::ptrdiff_t>(j) + 1);
          moves++;
          improved = true;
        }
      }
  }
  return moves;
}

}  // namespace heuresis::local
