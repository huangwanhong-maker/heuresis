// Operators on orders by the feature they keep (decision 0009; the survey, Part II, §4). An order of items has three kinds of
// feature, and an operator that keeps the one the objective depends on works best (Starkweather, McDaniel, Mathias, Whitley and
// Whitley 1991, "A comparison of genetic sequencing operators", ICGA, 69–76; Whitley 2019 for a review):
// - precedence, which item comes before which (distance: Kendall's, the pairs ordered differently);
// - adjacency, which items stand next to each other (distance: n − 1 less the edges both orders share, read undirected);
// - position, which item stands where (distance: Hamming's).
//
// The operators here, with what each keeps exactly:
// - precedence-preserving crossover, PPX (Bierwirth, Mattfeld and Kopfer 1996, PPSN IV, 310–318): the child repeatedly takes the
//   first item not yet taken from a parent chosen at random. Every precedence both parents share is kept: when an item is taken from
//   a parent, everything before it there is taken already.
// - order-based crossover, OX2 (Syswerda 1991, in Davis, Handbook of Genetic Algorithms, 332–349): items at random positions of the
//   second parent take, in its order, the places they hold in the first; the other items keep their places in the first.
// - position-based crossover, POS (Syswerda 1991): random positions keep the first parent's items; the other items fill the rest in
//   the second parent's order.
// - non-wrapping order crossover, NWOX (Cicirello 2006, GECCO, 1125–1132): the second parent's segment at its own positions; the
//   other items in the first parent's order, around it.
// - edge recombination with shared edges kept (after Whitley, Starkweather and Fuquay 1989, ICGA, 133–140, and the enhanced form of
//   Starkweather et al. 1991, which favours shared edges): the paths of edges both parents share are contracted into blocks, which
//   are joined by the parents' other edges, choosing at each step the neighbour with fewest unvisited neighbours. Every shared
//   edge is kept, by construction.
// heuresis's `variation` holds the others: cycle crossover and PMX (which keep every shared position), order crossover, and the
// swap, insertion, inversion and scramble mutations.
//
// A feature probe correlates the difference of an objective between two orders with each feature's distance between them, on pairs
// a few moves apart: the feature with the strongest correlation is the one the objective's structure lies in.

#pragma once

#include <heuresis/sequential/grants.hpp>

#include <heuresis/evolution/evolution.hpp>
#include <heuresis/spaces/space.hpp>
#include <heuresis/variation/distance.hpp>
#include <heuresis/variation/permutation.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <set>
#include <string>
#include <vector>

namespace heuresis::sequential {

using variation::Permutation;

namespace detail {
inline void same_items(const Permutation& a, const Permutation& b, const char* who) {
  if (a.size() != b.size() || !variation::is_permutation(a) || !variation::is_permutation(b))
    throw Refusal(std::string(who) + ": two permutations of the same items");
}
inline std::vector<std::size_t> positions(const Permutation& p) {
  std::vector<std::size_t> pos(p.size());
  for (std::size_t i = 0; i < p.size(); i++) pos[p[i]] = i;
  return pos;
}
inline std::vector<std::array<std::int64_t, 2>> neighbours(const Permutation& p) {
  std::vector<std::array<std::int64_t, 2>> nb(p.size(), {-1, -1});
  for (std::size_t i = 0; i + 1 < p.size(); i++) {
    nb[p[i]][1] = p[i + 1];
    nb[p[i + 1]][0] = p[i];
  }
  return nb;
}
inline bool adjacent(const std::vector<std::array<std::int64_t, 2>>& nb, std::size_t x, std::size_t y) {
  return nb[x][0] == static_cast<std::int64_t>(y) || nb[x][1] == static_cast<std::int64_t>(y);
}
}  // namespace detail

// ---- feature distances ----

inline std::size_t precedence_distance(const Permutation& a, const Permutation& b) { return static_cast<std::size_t>(variation::kendall_distance(a, b)); }
inline std::size_t position_distance(const Permutation& a, const Permutation& b) { return variation::hamming(a, b); }
inline std::size_t adjacency_distance(const Permutation& a, const Permutation& b) {
  detail::same_items(a, b, "adjacency_distance");
  if (a.size() < 2) return 0;
  const auto nb = detail::neighbours(b);
  std::size_t shared = 0;
  for (std::size_t i = 0; i + 1 < a.size(); i++) shared += detail::adjacent(nb, a[i], a[i + 1]);
  return a.size() - 1 - shared;
}

// ---- crossovers ----

inline Permutation precedence_preserving(const Permutation& a, const Permutation& b, calx::random::Stream& s) {
  detail::same_items(a, b, "precedence_preserving");
  const std::size_t n = a.size();
  std::vector<char> used(n, 0);
  std::size_t ia = 0, ib = 0;
  Permutation c;
  c.reserve(n);
  for (std::size_t k = 0; k < n; k++) {
    const bool from_a = s.uniform() < 0.5;
    const Permutation& p = from_a ? a : b;
    std::size_t& i = from_a ? ia : ib;
    while (used[p[i]]) i++;
    used[p[i]] = 1;
    c.push_back(p[i]);
  }
  return c;
}

inline Permutation order_based(const Permutation& a, const Permutation& b, calx::random::Stream& s) {
  detail::same_items(a, b, "order_based");
  const std::size_t n = a.size();
  std::vector<char> chosen(n, 0);  // by item
  Permutation picked;              // in b's order
  for (std::size_t i = 0; i < n; i++)
    if (s.uniform() < 0.5) chosen[b[i]] = 1, picked.push_back(b[i]);
  Permutation c = a;
  std::size_t k = 0;
  for (std::size_t i = 0; i < n; i++)
    if (chosen[a[i]]) c[i] = picked[k++];
  return c;
}

inline Permutation position_based(const Permutation& a, const Permutation& b, calx::random::Stream& s) {
  detail::same_items(a, b, "position_based");
  const std::size_t n = a.size();
  std::vector<char> keep(n, 0), used(n, 0);
  for (std::size_t i = 0; i < n; i++)
    if (s.uniform() < 0.5) keep[i] = 1, used[a[i]] = 1;
  Permutation c(n);
  std::size_t j = 0;
  for (std::size_t i = 0; i < n; i++) {
    if (keep[i]) {
      c[i] = a[i];
      continue;
    }
    while (used[b[j]]) j++;
    c[i] = b[j];
    used[b[j]] = 1;
  }
  return c;
}

inline Permutation non_wrapping_order(const Permutation& a, const Permutation& b, calx::random::Stream& s) {
  detail::same_items(a, b, "non_wrapping_order");
  const std::size_t n = a.size();
  if (n == 0) return a;
  std::size_t l = static_cast<std::size_t>(s.below(n)), r = static_cast<std::size_t>(s.below(n));
  if (l > r) std::swap(l, r);
  std::vector<char> in_segment(n, 0);
  for (std::size_t i = l; i <= r; i++) in_segment[b[i]] = 1;
  Permutation rest;
  for (auto x : a)
    if (!in_segment[x]) rest.push_back(x);
  Permutation c(n);
  std::size_t k = 0;
  for (std::size_t i = 0; i < l; i++) c[i] = rest[k++];
  for (std::size_t i = l; i <= r; i++) c[i] = b[i];
  for (std::size_t i = r + 1; i < n; i++) c[i] = rest[k++];
  return c;
}

inline Permutation edge_recombination(const Permutation& a, const Permutation& b, calx::random::Stream& s) {
  detail::same_items(a, b, "edge_recombination");
  const std::size_t n = a.size();
  if (n <= 2) return s.uniform() < 0.5 ? a : b;
  const auto na = detail::neighbours(a), nbb = detail::neighbours(b);
  // Shared edges, at most two at each item; their components are paths: the blocks.
  std::vector<std::vector<std::size_t>> shared(n);
  for (std::size_t i = 0; i + 1 < n; i++)
    if (detail::adjacent(nbb, a[i], a[i + 1])) shared[a[i]].push_back(a[i + 1]), shared[a[i + 1]].push_back(a[i]);
  std::vector<std::int64_t> block_of(n, -1);
  std::vector<std::vector<std::size_t>> blocks;
  for (std::size_t x = 0; x < n; x++) {
    if (block_of[x] >= 0 || shared[x].size() == 2) continue;  // start each path at an end
    std::vector<std::size_t> path{x};
    block_of[x] = static_cast<std::int64_t>(blocks.size());
    std::size_t prev = n, cur = x;
    for (;;) {
      std::size_t next = n;
      for (auto y : shared[cur])
        if (y != prev) next = y;
      if (next == n || block_of[next] >= 0) break;
      block_of[next] = static_cast<std::int64_t>(blocks.size());
      path.push_back(next);
      prev = cur, cur = next;
    }
    blocks.push_back(path);
  }
  const std::size_t m = blocks.size();
  std::vector<char> visited(m, 0);
  // The parents' edges from an item to items of other blocks (only ends of blocks have them).
  auto links = [&](std::size_t x) {
    std::vector<std::size_t> out;
    for (const auto* nb : {&na, &nbb})
      for (auto y : (*nb)[x])
        if (y >= 0 && block_of[static_cast<std::size_t>(y)] != block_of[x] && !visited[static_cast<std::size_t>(block_of[static_cast<std::size_t>(y)])])
          out.push_back(static_cast<std::size_t>(y));
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
  };
  auto far_end = [&](std::size_t x) {
    const auto& p = blocks[static_cast<std::size_t>(block_of[x])];
    return p.front() == x ? p.back() : p.front();
  };
  Permutation c;
  c.reserve(n);
  std::size_t enter = s.uniform() < 0.5 ? a[0] : b[0];
  for (std::size_t placed = 0; placed < m; placed++) {
    const auto bi = static_cast<std::size_t>(block_of[enter]);
    visited[bi] = 1;
    const auto& p = blocks[bi];
    if (p.front() == enter) c.insert(c.end(), p.begin(), p.end());
    else c.insert(c.end(), p.rbegin(), p.rend());
    if (placed + 1 == m) break;
    const std::size_t exit = far_end(enter);
    const auto cand = links(exit);
    if (!cand.empty()) {
      std::size_t best_count = n + 1;
      std::vector<std::size_t> best;
      for (auto y : cand) {
        visited[static_cast<std::size_t>(block_of[y])] = 1;  // as if entered, to count what would remain beyond it
        const std::size_t count = links(far_end(y)).size();
        visited[static_cast<std::size_t>(block_of[y])] = 0;
        if (count < best_count) best_count = count, best.clear();
        if (count == best_count) best.push_back(y);
      }
      enter = best[s.below(best.size())];
    } else {
      std::vector<std::size_t> open;
      for (std::size_t j = 0; j < m; j++)
        if (!visited[j]) open.push_back(j);
      const auto& q = blocks[open[s.below(open.size())]];
      enter = s.uniform() < 0.5 ? q.front() : q.back();
    }
  }
  return c;
}

// ---- what each keeps, checked ----

inline bool keeps_shared_precedences(const Permutation& a, const Permutation& b, const Permutation& c) {
  const auto pa = detail::positions(a), pb = detail::positions(b), pc = detail::positions(c);
  for (std::size_t x = 0; x < a.size(); x++)
    for (std::size_t y = 0; y < a.size(); y++)
      if (pa[x] < pa[y] && pb[x] < pb[y] && !(pc[x] < pc[y])) return false;
  return true;
}
inline bool keeps_shared_edges(const Permutation& a, const Permutation& b, const Permutation& c) {
  const auto nb = detail::neighbours(b), nc = detail::neighbours(c);
  for (std::size_t i = 0; i + 1 < a.size(); i++)
    if (detail::adjacent(nb, a[i], a[i + 1]) && !detail::adjacent(nc, a[i], a[i + 1])) return false;
  return true;
}
inline bool keeps_shared_positions(const Permutation& a, const Permutation& b, const Permutation& c) {
  for (std::size_t i = 0; i < a.size(); i++)
    if (a[i] == b[i] && c[i] != a[i]) return false;
  return true;
}

// ---- operators for the evolutionary loop, by feature ----

enum class Feature { precedence, adjacency, position };

inline const char* name(Feature f) {
  switch (f) {
    case Feature::precedence: return "precedence";
    case Feature::adjacency: return "adjacency";
    case Feature::position: return "position";
  }
  return "";
}

namespace detail {
inline Permutation from_value(const spaces::Value& v) {
  Permutation p(v.ints.size());
  for (std::size_t i = 0; i < p.size(); i++) p[i] = static_cast<std::uint32_t>(v.ints[i]);
  return p;
}
inline spaces::Value to_value(const Permutation& p) { return spaces::permutation_value(std::vector<std::int64_t>(p.begin(), p.end())); }
}  // namespace detail

// Operators on permutations of n items that keep a feature: precedence (PPX; insertion), adjacency (edge recombination with
// shared edges kept; inversion), position (cycle crossover; swap). With every feature, the loop's schedule credits whichever pays.
inline evolution::Operators<spaces::Value> order_operators(std::size_t n, std::vector<Feature> features = {Feature::precedence, Feature::adjacency, Feature::position}) {
  if (n == 0) throw Refusal("orders of no items");
  using Ctx = evolution::Context<spaces::Value>;
  using M = evolution::Member<spaces::Value>;
  using V = spaces::Value;
  evolution::Operators<V> ops;
  ops.makers.push_back({"random order", [n](std::size_t k, const Ctx&, calx::random::Stream& s) {
                          std::vector<V> out;
                          for (std::size_t i = 0; i < k; i++) out.push_back(detail::to_value(variation::random_permutation(n, s)));
                          return out;
                        }, 1.0});
  for (Feature f : features) {
    switch (f) {
      case Feature::precedence:
        ops.mutations.push_back({"insertion", [](const M& p, const Ctx&, calx::random::Stream& s) -> std::optional<V> { return detail::to_value(variation::insertion_mutation(detail::from_value(p.genotype), s)); }, 1.0});
        ops.crossovers.push_back({"PPX", [](const M& a, const M& b, const Ctx&, calx::random::Stream& s) -> std::optional<V> { return detail::to_value(precedence_preserving(detail::from_value(a.genotype), detail::from_value(b.genotype), s)); }, 1.0});
        break;
      case Feature::adjacency:
        ops.mutations.push_back({"inversion", [](const M& p, const Ctx&, calx::random::Stream& s) -> std::optional<V> { return detail::to_value(variation::inversion_mutation(detail::from_value(p.genotype), s)); }, 1.0});
        ops.crossovers.push_back({"edge recombination", [](const M& a, const M& b, const Ctx&, calx::random::Stream& s) -> std::optional<V> { return detail::to_value(edge_recombination(detail::from_value(a.genotype), detail::from_value(b.genotype), s)); }, 1.0});
        break;
      case Feature::position:
        ops.mutations.push_back({"swap", [](const M& p, const Ctx&, calx::random::Stream& s) -> std::optional<V> { return detail::to_value(variation::swap_mutation(detail::from_value(p.genotype), s)); }, 1.0});
        ops.crossovers.push_back({"cycle crossover", [](const M& a, const M& b, const Ctx&, calx::random::Stream& s) -> std::optional<V> { return detail::to_value(variation::cycle_crossover(detail::from_value(a.genotype), detail::from_value(b.genotype), s).first); }, 1.0});
        break;
    }
  }
  ops.distance = [](const V& a, const V& b) { return static_cast<double>(precedence_distance(detail::from_value(a), detail::from_value(b))); };
  ops.key = [](const V& v) { return spaces::key(v); };
  return ops;
}

// ---- the feature probe ----

struct FeatureProbe {
  std::size_t pairs = 0;
  double precedence = 0, adjacency = 0, position = 0;  // Pearson's correlation of |Δf| with each distance
  Feature strongest() const {
    if (precedence >= adjacency && precedence >= position) return Feature::precedence;
    return adjacency >= position ? Feature::adjacency : Feature::position;
  }
};

// Pairs of orders of n items, the second a few random moves (swaps, insertions, inversions) from the first.
inline FeatureProbe feature_probe(std::size_t n, const std::function<double(const Permutation&)>& f, calx::random::Stream& s, std::size_t pairs = 400) {
  std::vector<double> df, dp, da, dq;
  for (std::size_t i = 0; i < pairs; i++) {
    const Permutation x = variation::random_permutation(n, s);
    Permutation y = x;
    const std::size_t moves = 1 + static_cast<std::size_t>(s.below(3));
    for (std::size_t k = 0; k < moves; k++) {
      switch (s.below(3)) {
        case 0: y = variation::swap_mutation(y, s); break;
        case 1: y = variation::insertion_mutation(y, s); break;
        default: y = variation::inversion_mutation(y, s); break;
      }
    }
    df.push_back(std::abs(f(x) - f(y)));
    dp.push_back(static_cast<double>(precedence_distance(x, y)));
    da.push_back(static_cast<double>(adjacency_distance(x, y)));
    dq.push_back(static_cast<double>(position_distance(x, y)));
  }
  auto corr = [](const std::vector<double>& u, const std::vector<double>& v) {
    const double N = static_cast<double>(u.size());
    double mu = 0, mv = 0;
    for (std::size_t i = 0; i < u.size(); i++) mu += u[i], mv += v[i];
    mu /= N, mv /= N;
    double suv = 0, suu = 0, svv = 0;
    for (std::size_t i = 0; i < u.size(); i++) suv += (u[i] - mu) * (v[i] - mv), suu += (u[i] - mu) * (u[i] - mu), svv += (v[i] - mv) * (v[i] - mv);
    return suu > 0 && svv > 0 ? suv / std::sqrt(suu * svv) : 0.0;
  };
  return {pairs, corr(df, dp), corr(df, da), corr(df, dq)};
}

}  // namespace heuresis::sequential
