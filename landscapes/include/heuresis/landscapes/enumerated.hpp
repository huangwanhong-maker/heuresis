// Landscapes on small hypercubes, known exactly by enumeration.
//
// A function on {0, 1}ⁿ (n ≤ 26) is tabulated, the point x stored at the index Σ xᵢ 2ⁱ, and its structure under
// one-bit flips computed exactly:
// - the one-step correlation of the random walk, ρ(1) = 1 − E[(f(x) − f(y))²]/(2 Var f), x uniform and y a
//   uniform neighbour (the walk's stationary law is uniform on the regular graph);
// - Grover's wave equation: f is elementary when Δf = −λ (f − f̄), Δ the graph Laplacian, i.e. when the mean of
//   f over the n neighbours of every x is f(x) + (λ/n)(f̄ − f(x)) (Grover 1992, "Local search and the local
//   structure of NP-complete problems", Oper. Res. Lett. 12, 235–243; Stadler 1996). The fit gives λ and the
//   largest residual. On the hypercube the Walsh functions of order q are the eigenfunctions with λ = 2q, so a
//   function is elementary exactly when its Walsh expansion has one order besides the mean: OneMax (q = 1), an
//   Ising energy Σ J_ij sᵢ sⱼ without fields (q = 2);
// - local minima (no neighbour strictly better) and their basins under steepest descent, each point descending
//   to its best neighbour while one is strictly better (ties to the lower bit), the "basins of attraction" of
//   local optima networks (Ochoa, Tomassini, Vérel and Darabos 2008, "A study of NK landscapes' basins and local
//   optima networks", GECCO, 555–562);
// - neutrality: the fraction of a point's neighbours of equal value, and the neutral networks, the components
//   of the graph restricted to edges of equal value (Reidys and Stadler 2001, "Neutrality in fitness
//   landscapes", Appl. Math. Comput. 117, 321–350).

#pragma once

#include <heuresis/landscapes/walks.hpp>
#include <heuresis/variation/binary.hpp>
#include <heuresis/variation/distance.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace heuresis::landscapes {

using heuresis::variation::Bits;

class Hypercube {
 public:
  template <class F>
  static Hypercube tabulate(std::size_t n, F&& f) {
    if (n == 0 || n > 26) throw std::invalid_argument("heuresis::landscapes::Hypercube: 1 ≤ n ≤ 26");
    Hypercube h;
    h.n_ = n;
    h.values_.resize(std::size_t{1} << n);
    for (std::size_t m = 0; m < h.values_.size(); m++) h.values_[m] = detail::value_of(f(point(m, n)));
    return h;
  }
  static Bits point(std::size_t m, std::size_t n) {
    Bits x(n);
    for (std::size_t i = 0; i < n; i++) x[i] = (m >> i) & 1u;
    return x;
  }
  static std::size_t index(const Bits& x) {
    std::size_t m = 0;
    for (std::size_t i = 0; i < x.size(); i++) m |= static_cast<std::size_t>(x[i]) << i;
    return m;
  }
  std::size_t n() const { return n_; }
  std::size_t size() const { return values_.size(); }
  double operator[](std::size_t m) const { return values_[m]; }
  std::span<const double> values() const { return values_; }
  double mean() const { return tyche::descriptive::mean(values_); }
  double variance() const { return tyche::descriptive::variance(values_, 0); }

 private:
  std::size_t n_ = 0;
  std::vector<double> values_;
};

// ρ(1) of the random walk of one-bit flips.
inline double one_step_correlation(const Hypercube& h) {
  double sq = 0;
  for (std::size_t m = 0; m < h.size(); m++)
    for (std::size_t i = 0; i < h.n(); i++) {
      const double d = h[m] - h[m ^ (std::size_t{1} << i)];
      sq += d * d;
    }
  sq /= static_cast<double>(h.size() * h.n());
  return 1 - sq / (2 * h.variance());
}

struct Elementary {
  double lambda = 0;    // the fitted eigenvalue of the Laplacian
  double residual = 0;  // max over x of |mean over N(x) − f(x) − (λ/n)(f̄ − f(x))|
};

inline Elementary elementary(const Hypercube& h) {
  const double fbar = h.mean(), n = static_cast<double>(h.n());
  std::vector<double> lap(h.size());
  double num = 0, den = 0;
  for (std::size_t m = 0; m < h.size(); m++) {
    double s = 0;
    for (std::size_t i = 0; i < h.n(); i++) s += h[m ^ (std::size_t{1} << i)];
    lap[m] = s / n - h[m];
    num += lap[m] * (fbar - h[m]);
    den += (fbar - h[m]) * (fbar - h[m]);
  }
  Elementary e;
  e.lambda = den > 0 ? n * num / den : 0;
  for (std::size_t m = 0; m < h.size(); m++) e.residual = std::max(e.residual, std::fabs(lap[m] - e.lambda / n * (fbar - h[m])));
  return e;
}

inline std::vector<std::size_t> local_minima(const Hypercube& h) {
  std::vector<std::size_t> out;
  for (std::size_t m = 0; m < h.size(); m++) {
    bool local = true;
    for (std::size_t i = 0; i < h.n() && local; i++) local = h[m] <= h[m ^ (std::size_t{1} << i)];
    if (local) out.push_back(m);
  }
  return out;
}

struct Basins {
  std::vector<std::size_t> optima;     // the local minima reached, ascending by index
  std::vector<std::size_t> sizes;      // the size of each one's basin
  std::vector<std::size_t> attractor;  // for every point, the index into optima of where it descends
};

inline Basins basins(const Hypercube& h) {
  const std::size_t N = h.size();
  std::vector<std::size_t> next(N);
  for (std::size_t m = 0; m < N; m++) {
    std::size_t best = m;
    for (std::size_t i = 0; i < h.n(); i++) {
      const std::size_t y = m ^ (std::size_t{1} << i);
      if (h[y] < h[best]) best = y;
    }
    next[m] = best;
  }
  std::vector<std::size_t> end(N, N), path;  // descent strictly decreases f, so it never cycles
  for (std::size_t m = 0; m < N; m++) {
    path.clear();
    std::size_t y = m;
    while (end[y] == N && next[y] != y) {
      path.push_back(y);
      y = next[y];
    }
    const std::size_t e = end[y] != N ? end[y] : y;
    end[y] = e;
    for (auto z : path) end[z] = e;
  }
  Basins b;
  for (std::size_t m = 0; m < N; m++)
    if (next[m] == m) b.optima.push_back(m);
  b.sizes.assign(b.optima.size(), 0);
  b.attractor.resize(N);
  for (std::size_t m = 0; m < N; m++) {
    const std::size_t k = static_cast<std::size_t>(std::lower_bound(b.optima.begin(), b.optima.end(), end[m]) - b.optima.begin());
    b.attractor[m] = k;
    b.sizes[k]++;
  }
  return b;
}

inline double neutral_degree(const Hypercube& h, std::size_t m) {
  std::size_t equal = 0;
  for (std::size_t i = 0; i < h.n(); i++) equal += h[m ^ (std::size_t{1} << i)] == h[m];
  return static_cast<double>(equal) / static_cast<double>(h.n());
}

// The number of neutral networks, by union–find over the edges of equal value.
inline std::size_t neutral_networks(const Hypercube& h) {
  std::vector<std::size_t> parent(h.size());
  std::iota(parent.begin(), parent.end(), std::size_t{0});
  auto find = [&](std::size_t a) {
    while (parent[a] != a) a = parent[a] = parent[parent[a]];
    return a;
  };
  std::size_t components = h.size();
  for (std::size_t m = 0; m < h.size(); m++)
    for (std::size_t i = 0; i < h.n(); i++) {
      const std::size_t y = m ^ (std::size_t{1} << i);
      if (y < m || h[y] != h[m]) continue;
      const std::size_t a = find(m), b = find(y);
      if (a != b) {
        parent[a] = b;
        components--;
      }
    }
  return components;
}

}  // namespace heuresis::landscapes
