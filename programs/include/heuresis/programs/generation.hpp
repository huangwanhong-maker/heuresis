// Random well-sorted terms: Koza's grow and full methods, typed, and ramped half-and-half.
//
// To make a term of sort s and depth ≤ d, choose a symbol that can head one: a terminal of sort s, or a function
// symbol f with argument sorts (s₁, …, s_k) yielding s, each sᵢ generable at depth ≤ d − 1 (signature.hpp, G_{d−1});
// then make each argument the same way with depth ≤ d − 1. Because the choice is restricted to sorts that are
// generable, generation never fails when s ∈ G_d, and every term it makes is well-sorted (Montana 1995, §2.2,
// with the table of generable types computed here as a fixpoint).
//
// - grow: the head is chosen uniformly among every symbol that can head a term of sort s within depth d;
// - full: among the function symbols while d > 0, a terminal at depth 0; when no function symbol can head a term of
//   sort s, a terminal is used, so a typed full term may be shallower than d;
// - ramped half-and-half: depths spread evenly over [d_min, d_max], half grown and half full at each depth (Koza
//   1992, §6.2), the population's k-th term at depth d_min + (k / 2) mod (d_max − d_min + 1), full when k is odd.
// Argument sorts are chosen uniformly among the options; an ephemeral constant draws its value where it is made.

#pragma once

#include <heuresis/programs/trees.hpp>

#include <calx/random/stream.hpp>

namespace heuresis::programs {

namespace detail {
inline void make(const Signature& sig, const Sort& target, std::size_t d, bool full, calx::random::Stream& s, Tree& out) {
  std::vector<std::size_t> terminals = sig.terminals(target);
  std::vector<std::pair<std::size_t, std::vector<std::vector<Sort>>>> functions;
  if (d > 0)
    for (std::size_t f = 0; f < sig.primitive_count(); f++) {
      auto options = sig.argument_options(f, target, d - 1);
      if (!options.empty()) functions.push_back({f, std::move(options)});
    }
  const bool use_function = !functions.empty() && (terminals.empty() || full || s.below(terminals.size() + functions.size()) >= terminals.size());
  if (!use_function) {
    if (terminals.empty()) throw std::invalid_argument("heuresis::programs::generate: no term of this sort within this depth");
    const std::size_t t = terminals[s.below(terminals.size())];
    Node n{static_cast<std::uint32_t>(t), 0, target};
    if (sig.is_constant(t)) {
      const Constant& c = sig.constant(t);
      n.value = c.ephemeral ? s.uniform(c.lower, c.upper) : c.value;
    }
    out.push_back(n);
    return;
  }
  const auto& [f, options] = functions[s.below(functions.size())];
  const auto& args = options[s.below(options.size())];
  out.push_back(Node{static_cast<std::uint32_t>(f), 0, target});
  for (const auto& a : args) make(sig, a, d - 1, full, s, out);
}
}  // namespace detail

// A term of sort `target` and depth ≤ d, grown or full.
inline Tree generate(const Signature& sig, const Sort& target, std::size_t d, bool full, calx::random::Stream& s) {
  if (!sig.generable(target, d)) throw std::invalid_argument("heuresis::programs::generate: no term of this sort within this depth");
  Tree t;
  detail::make(sig, target, d, full, s, t);
  return t;
}

inline Tree grow(const Signature& sig, const Sort& target, std::size_t d, calx::random::Stream& s) { return generate(sig, target, d, false, s); }
inline Tree full(const Signature& sig, const Sort& target, std::size_t d, calx::random::Stream& s) { return generate(sig, target, d, true, s); }

// The depth and method of the k-th term of a ramped half-and-half population: (depth, full).
inline std::pair<std::size_t, bool> ramp(std::size_t k, std::size_t min_depth, std::size_t max_depth) {
  const std::size_t span = max_depth - min_depth + 1;
  return {min_depth + (k / 2) % span, (k % 2) == 1};
}

// The k-th term of a ramped half-and-half population. A depth at which the sort is not generable is raised to the
// least at which it is.
inline Tree ramped(const Signature& sig, const Sort& target, std::size_t k, std::size_t min_depth, std::size_t max_depth, calx::random::Stream& s) {
  auto [d, is_full] = ramp(k, min_depth, max_depth);
  while (!sig.generable(target, d) && d < max_depth) d++;
  return generate(sig, target, d, is_full, s);
}

}  // namespace heuresis::programs
