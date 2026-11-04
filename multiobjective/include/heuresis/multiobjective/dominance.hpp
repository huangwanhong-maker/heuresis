// The Pareto order, constrained dominance, and the decomposition of a population into fronts.
//
// On ℝᵐ, with every objective minimised, a weakly dominates b (a ≼ b) when aᵢ ≤ bᵢ for every i; ≼ is the product
// order, a partial order. a dominates b (a ≺ b) when a ≼ b and a ≠ b: the strict part of ≼. Two points neither of
// which dominates the other are incomparable; a set of pairwise incomparable points is an antichain. A finite
// population P with ≺ is a finite poset; its minimal elements are its non-dominated set, the first front.
//
// Constrained dominance (Deb, Pratap, Agarwal and Meyarivan 2002, "A fast and elitist multiobjective genetic
// algorithm: NSGA-II", IEEE Trans. Evol. Comput. 6, 182–197, §VI) extends Deb's feasibility rule to several
// objectives: a feasible point dominates every infeasible one; of two infeasible points, the one of smaller
// total violation dominates; two feasible points are compared by ≺. A failed evaluation is dominated by every
// one that did not fail. The relation is again a strict partial order (it is the lexicographic product of the
// order of violations with ≺, restricted so that only feasible points meet ≺), so everything below applies to it.
//
// Fronts. The height of x in a finite poset is the length of the longest chain of points dominating it: 0 for the
// minimal points, otherwise 1 + max{height(y) : y ≺ x}. The fronts F₀, F₁, … are the level sets of the height;
// each is an antichain, and F_k is the minimal set of P ∖ (F₀ ∪ … ∪ F_{k−1}). Deb et al.'s fast non-dominated
// sort computes them in O(m N²) by counting, for each point, how many dominate it, and peeling the points whose
// count falls to zero (their §III.A). Ties are kept in index order within each front, so the decomposition is a
// function of the population's order, which the algorithms fix by ticket.

#pragma once

#include <heuresis/search/evaluation.hpp>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <span>
#include <stdexcept>
#include <vector>

namespace heuresis::multiobjective {

using heuresis::search::Evaluation;
using Point = std::vector<double>;

// a ≼ b: aᵢ ≤ bᵢ for every i.
inline bool weakly_dominates(std::span<const double> a, std::span<const double> b) {
  if (a.size() != b.size()) throw std::invalid_argument("heuresis::multiobjective::weakly_dominates: points of different dimensions");
  for (std::size_t i = 0; i < a.size(); i++)
    if (a[i] > b[i]) return false;
  return true;
}

// a ≺ b: a ≼ b and a ≠ b.
inline bool dominates(std::span<const double> a, std::span<const double> b) {
  if (a.size() != b.size()) throw std::invalid_argument("heuresis::multiobjective::dominates: points of different dimensions");
  bool strict = false;
  for (std::size_t i = 0; i < a.size(); i++) {
    if (a[i] > b[i]) return false;
    strict = strict || a[i] < b[i];
  }
  return strict;
}

// Deb's constrained dominance on evaluations.
inline bool constrained_dominates(const Evaluation& a, const Evaluation& b) {
  const bool fa = a.failed(), fb = b.failed();
  if (fa || fb) return !fa && fb;
  const double va = a.violation(), vb = b.violation();
  if (va > 0 || vb > 0) return va < vb;
  return dominates(a.objectives, b.objectives);
}

// The fronts of a population of n under a strict partial order `prec(i, j)` (i dominates j), first front first,
// each in index order; and each point's front.
struct Fronts {
  std::vector<std::vector<std::size_t>> fronts;
  std::vector<std::size_t> rank;
};

inline Fronts non_dominated_sort(std::size_t n, const std::function<bool(std::size_t, std::size_t)>& prec) {
  Fronts out;
  out.rank.assign(n, 0);
  std::vector<std::vector<std::size_t>> dominated(n);  // S_p: the points p dominates
  std::vector<std::size_t> count(n, 0);                // n_p: how many dominate p
  for (std::size_t p = 0; p < n; p++)
    for (std::size_t q = p + 1; q < n; q++) {
      if (prec(p, q)) { dominated[p].push_back(q); count[q]++; }
      else if (prec(q, p)) { dominated[q].push_back(p); count[p]++; }
    }
  std::vector<std::size_t> current;
  for (std::size_t p = 0; p < n; p++)
    if (count[p] == 0) current.push_back(p);
  for (std::size_t k = 0; !current.empty(); k++) {
    std::vector<std::size_t> next;
    for (std::size_t p : current) {
      out.rank[p] = k;
      for (std::size_t q : dominated[p])
        if (--count[q] == 0) next.push_back(q);
    }
    std::sort(next.begin(), next.end());
    out.fronts.push_back(std::move(current));
    current = std::move(next);
  }
  return out;
}

// The fronts of objective vectors under ≺.
inline Fronts non_dominated_sort(std::span<const Point> points) {
  return non_dominated_sort(points.size(), [&](std::size_t i, std::size_t j) { return dominates(points[i], points[j]); });
}

// The fronts of evaluations under constrained dominance.
inline Fronts non_dominated_sort(std::span<const Evaluation> e) {
  return non_dominated_sort(e.size(), [&](std::size_t i, std::size_t j) { return constrained_dominates(e[i], e[j]); });
}

// The indices of the non-dominated points (the first front), in index order.
inline std::vector<std::size_t> non_dominated(std::span<const Point> points) {
  std::vector<std::size_t> out;
  for (std::size_t i = 0; i < points.size(); i++) {
    bool dominated = false;
    for (std::size_t j = 0; j < points.size() && !dominated; j++) dominated = j != i && dominates(points[j], points[i]);
    if (!dominated) out.push_back(i);
  }
  return out;
}

}  // namespace heuresis::multiobjective
