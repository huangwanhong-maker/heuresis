// Combination without one score (decision 0010; the survey, Part V, design §7). Candidates are ordered by tiers before any
// Pareto comparison, so that a weak kind of evidence cannot outrank a strong one:
//   1. failures, refusals and abstentions last;
//   2. constraints of sufficient standing bind, by Deb's feasibility rule (Deb 2000, Comput. Methods Appl. Mech. Engrg. 186,
//      311–338): the feasible before the infeasible, the infeasible by their total violation;
//   3. Pareto dominance over objectives of sufficient standing;
//   4. the weaker objectives (estimated, proposed) only as tie-breakers, in their order, and as descriptors.
// The standing that suffices is declared (by default computed: what a declared model gives exactly binds and selects; estimates
// and proposals do not). An assessment maps to heuresis's Evaluation the same way, so that the evolutionary loop selects only on
// what may select; the rest travels as descriptors, for diversity and for reading.

#pragma once

#include <heuresis/assessment/record.hpp>

#include <heuresis/multiobjective/dominance.hpp>
#include <heuresis/search/evaluation.hpp>

#include <algorithm>
#include <numeric>
#include <vector>

namespace heuresis::assessment {

struct Tiering {
  Standing objectives_up_to = Standing::computed;   // objectives at least this strong select
  Standing constraints_up_to = Standing::computed;  // constraints at least this strong bind
};

namespace detail {
inline bool strong_enough(Standing s, Standing floor) { return static_cast<int>(s) <= static_cast<int>(floor); }
}  // namespace detail

inline search::Evaluation to_evaluation(const Assessment& a, const Tiering& t = {}) {
  search::Evaluation e;
  e.cost = a.cost;
  if (!a.ok()) {
    e.status = search::Status::failed;
    e.reason = std::string(name(a.status)) + (a.reason.empty() ? "" : ": " + a.reason);
    return e;
  }
  for (const auto& m : a.metrics) {
    if (m.role == Role::objective && detail::strong_enough(m.standing, t.objectives_up_to)) e.objectives.push_back(m.value);
    else if (m.role == Role::constraint && detail::strong_enough(m.standing, t.constraints_up_to)) e.constraints.push_back(m.value);
    else if (m.role != Role::report) e.descriptors.push_back(m.value);
  }
  if (e.objectives.empty()) {
    e.status = search::Status::failed;
    e.reason = "no objective of sufficient standing";
  }
  return e;
}

struct Tiered {
  bool failed = false;
  double violation = 0;               // of the binding constraints
  std::vector<double> select;         // objectives that select
  std::vector<double> tie;            // weaker objectives, in order
};

inline Tiered tiered(const Assessment& a, const Tiering& t = {}) {
  Tiered r;
  r.failed = !a.ok();
  for (const auto& m : a.metrics) {
    if (m.role == Role::objective) (detail::strong_enough(m.standing, t.objectives_up_to) ? r.select : r.tie).push_back(m.value);
    else if (m.role == Role::constraint && detail::strong_enough(m.standing, t.constraints_up_to)) r.violation += std::max(0.0, m.value);
  }
  if (r.select.empty()) r.failed = true;
  return r;
}

// The order in which candidates are reported, best first: failures last, then the binding constraints' violation, then the
// Pareto front over selecting objectives, then the weaker objectives lexicographically; ties by index.
inline std::vector<std::size_t> tiered_ranking(const std::vector<Assessment>& as, const Tiering& t = {}) {
  const std::size_t n = as.size();
  std::vector<Tiered> v;
  for (const auto& a : as) v.push_back(tiered(a, t));
  for (const auto& x : v)
    if (!x.failed && x.select.size() != v[0].select.size()) throw Refusal("candidates assessed on different numbers of selecting objectives");
  // Pareto ranks among the feasible, non-failed candidates.
  std::vector<std::size_t> feasible;
  for (std::size_t i = 0; i < n; i++)
    if (!v[i].failed && v[i].violation == 0) feasible.push_back(i);
  std::vector<std::size_t> rank(n, 0);
  if (!feasible.empty()) {
    const auto fr = multiobjective::non_dominated_sort(feasible.size(), [&](std::size_t i, std::size_t j) {
      return multiobjective::dominates(v[feasible[i]].select, v[feasible[j]].select);
    });
    for (std::size_t k = 0; k < feasible.size(); k++) rank[feasible[k]] = fr.rank[k];
  }
  std::vector<std::size_t> order(n);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](std::size_t i, std::size_t j) {
    const Tiered &a = v[i], &b = v[j];
    if (a.failed != b.failed) return !a.failed;
    if (a.failed) return false;
    if ((a.violation == 0) != (b.violation == 0)) return a.violation == 0;
    if (a.violation != b.violation) return a.violation < b.violation;
    if (rank[i] != rank[j]) return rank[i] < rank[j];
    return std::lexicographical_compare(a.tie.begin(), a.tie.end(), b.tie.begin(), b.tie.end());
  });
  return order;
}

}  // namespace heuresis::assessment
