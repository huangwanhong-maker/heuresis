// Variation of terms that keeps them well-sorted, and the control of their growth.
//
// Each operator replaces a subterm by another of the same sort, so the term's sort, and the well-sortedness of
// every node, is kept (Montana 1995, §2.3):
// - subtree crossover exchanges a subterm of one parent with a subterm of the same sort of the other; crossover
//   points are internal nodes with probability 0.9 where the term has any (Koza 1992, §6.4, the 90/10 rule);
// - subtree mutation replaces a subterm by a grown term of its sort (Koza 1992, §6.5.1);
// - point mutation replaces each node, with probability `rate`, by a symbol of the same arity giving the same sort
//   from the same argument sorts, or a terminal by another terminal of its sort (an ephemeral constant redraws
//   its value) (Poli and Langdon 1998, "Schema theory for genetic programming with one-point crossover and point
//   mutation", Evol. Comput. 6, 231–252);
// - hoist mutation replaces the whole term by one of its subterms of the root's sort (Kinnear 1994, "Fitness
//   landscapes and difficulty in genetic programming", IEEE WCCI, 142–147);
// - shrink mutation replaces a subterm by one of its own arguments of the same sort (as DEAP's mutShrink does;
//   Fortin, De Rainville, Gardner, Parizeau and Gagné 2012, "DEAP: evolutionary algorithms made easy", JMLR 13,
//   2171–2175).
// A child beyond the depth or size limit is replaced by its parent (Koza 1992, §6.4: depth 17).
//
// Lexicographic parsimony pressure compares by fitness first and, among equals, prefers the smaller term (Luke
// and Panait 2002, "Lexicographic parsimony pressure", GECCO, 829–836), which slows the growth of terms without a
// penalty coefficient.

#pragma once

#include <heuresis/programs/generation.hpp>
#include <heuresis/search/evaluation.hpp>
#include <heuresis/selection/parents.hpp>

#include <calx/random/stream.hpp>

#include <utility>

namespace heuresis::programs {

struct Limits {
  std::size_t depth = 17;
  std::size_t size = 512;
  std::size_t mutation_depth = 4;  // the depth of the subterms that subtree mutation grows
};

inline bool within(const Signature& sig, const Tree& t, const Limits& l) { return t.size() <= l.size && depth(sig, t) <= l.depth; }

namespace detail {
// A node, internal with probability 0.9 when there are internal nodes, among those of the given sort (or any).
inline std::optional<std::size_t> pick(const Signature& sig, const Tree& t, calx::random::Stream& s, const Sort* sort = nullptr) {
  std::vector<std::size_t> internal, leaves;
  for (std::size_t i = 0; i < t.size(); i++) {
    if (sort && !(t[i].sort == *sort)) continue;
    (sig.arity(t[i].symbol) ? internal : leaves).push_back(i);
  }
  const bool inner = !internal.empty() && (leaves.empty() || s.uniform() < 0.9);
  const auto& from = inner ? internal : leaves;
  if (from.empty()) return std::nullopt;
  return from[s.below(from.size())];
}

inline Tree splice(const Tree& host, std::size_t b, std::size_t e, const Tree& donor, std::size_t db, std::size_t de) {
  Tree out(host.begin(), host.begin() + static_cast<std::ptrdiff_t>(b));
  out.insert(out.end(), donor.begin() + static_cast<std::ptrdiff_t>(db), donor.begin() + static_cast<std::ptrdiff_t>(de));
  out.insert(out.end(), host.begin() + static_cast<std::ptrdiff_t>(e), host.end());
  return out;
}
}  // namespace detail

inline std::pair<Tree, Tree> subtree_crossover(const Signature& sig, const Tree& a, const Tree& b, const Limits& limits, calx::random::Stream& s) {
  const auto i = detail::pick(sig, a, s);
  const Sort sort = a[*i].sort;
  const auto j = detail::pick(sig, b, s, &sort);
  if (!j) return {a, b};
  const std::size_t ie = subtree_end(sig, a, *i), je = subtree_end(sig, b, *j);
  Tree c1 = detail::splice(a, *i, ie, b, *j, je), c2 = detail::splice(b, *j, je, a, *i, ie);
  if (!within(sig, c1, limits)) c1 = a;
  if (!within(sig, c2, limits)) c2 = b;
  return {std::move(c1), std::move(c2)};
}

inline Tree subtree_mutation(const Signature& sig, const Tree& t, const Limits& limits, calx::random::Stream& s) {
  const std::size_t i = s.below(t.size());
  const auto d = node_depths(sig, t)[i];
  const std::size_t room = limits.depth > d ? limits.depth - d : 0;
  std::size_t budget = std::min(limits.mutation_depth, room);
  while (!sig.generable(t[i].sort, budget) && budget < room) budget++;
  Tree fresh = grow(sig, t[i].sort, budget, s);
  Tree c = detail::splice(t, i, subtree_end(sig, t, i), fresh, 0, fresh.size());
  return within(sig, c, limits) ? c : t;
}

inline Tree point_mutation(const Signature& sig, const Tree& t, double rate, calx::random::Stream& s) {
  Tree c = t;
  // The sorts of each node's arguments, read from the term.
  for (std::size_t i = 0; i < c.size(); i++) {
    if (!(s.uniform() < rate)) continue;
    const auto sym = c[i].symbol;
    const std::size_t k = sig.arity(sym);
    std::vector<std::size_t> options;
    if (k == 0) {
      for (auto u : sig.terminals(c[i].sort))
        if (u != sym || (sig.is_constant(u) && sig.constant(u).ephemeral)) options.push_back(u);
    } else {
      std::vector<Sort> args;
      for (std::size_t j = i + 1; args.size() < k; j = subtree_end(sig, c, j)) args.push_back(c[j].sort);
      for (std::size_t f = 0; f < sig.primitive_count(); f++) {
        if (f == sym || sig.arity(f) != k) continue;
        const auto r = sig.synthesise(f, args);
        if (r && *r == c[i].sort) options.push_back(f);
      }
    }
    if (options.empty()) continue;
    const std::size_t u = options[s.below(options.size())];
    c[i].symbol = static_cast<std::uint32_t>(u);
    if (k == 0 && sig.is_constant(u)) {
      const Constant& con = sig.constant(u);
      c[i].value = con.ephemeral ? s.uniform(con.lower, con.upper) : con.value;
    }
  }
  return c;
}

inline Tree hoist_mutation(const Signature& sig, const Tree& t, calx::random::Stream& s) {
  std::vector<std::size_t> options;
  for (std::size_t i = 1; i < t.size(); i++)
    if (t[i].sort == t[0].sort) options.push_back(i);
  if (options.empty()) return t;
  const std::size_t i = options[s.below(options.size())];
  return Tree(t.begin() + static_cast<std::ptrdiff_t>(i), t.begin() + static_cast<std::ptrdiff_t>(subtree_end(sig, t, i)));
}

inline Tree shrink_mutation(const Signature& sig, const Tree& t, calx::random::Stream& s) {
  std::vector<std::pair<std::size_t, std::size_t>> options;  // (node, argument of the same sort)
  for (std::size_t i = 0; i < t.size(); i++) {
    const std::size_t k = sig.arity(t[i].symbol);
    for (std::size_t j = i + 1, a = 0; a < k; a++, j = subtree_end(sig, t, j))
      if (t[j].sort == t[i].sort) options.push_back({i, j});
  }
  if (options.empty()) return t;
  const auto [i, j] = options[s.below(options.size())];
  return detail::splice(t, i, subtree_end(sig, t, i), t, j, subtree_end(sig, t, j));
}

// Deb's order on the evaluations and, among equals, the smaller term.
inline selection::Less lexicographic_parsimony(std::span<const search::Evaluation> e, std::span<const std::size_t> sizes) {
  return [e, sizes](std::size_t i, std::size_t j) {
    if (search::better(e[i], e[j])) return true;
    if (search::better(e[j], e[i])) return false;
    return sizes[i] < sizes[j];
  };
}

}  // namespace heuresis::programs
