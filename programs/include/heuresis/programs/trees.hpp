// Terms stored as trees in prefix order, their measures, their sorts, and their interpretation.
//
// A term is the sequence of its nodes in prefix (Polish) order, each node a symbol of the signature with, for an
// ephemeral constant, its value; the arities determine the tree, so the sequence is the term (Łukasiewicz's
// notation). Each node also records its sort, which variation reads to keep terms well-sorted; `well_sorted`
// recomputes every sort from the leaves by the signature's homomorphism and compares.
//
// The depth of a term is the number of edges on its longest path from the root (a terminal has depth 0), and its
// size the number of nodes (Koza 1992, §6.4).
//
// Interpretation is the evaluation homomorphism from the term algebra to the algebra of real numbers given by the
// primitives' implementations and an assignment of the variables, computed with a stack read from the right.

#pragma once

#include <heuresis/programs/signature.hpp>

#include <cmath>
#include <span>
#include <string>
#include <vector>

namespace heuresis::programs {

struct Node {
  std::uint32_t symbol = 0;
  double value = 0;  // a constant's value
  Sort sort;
  bool operator==(const Node&) const = default;
};

using Tree = std::vector<Node>;

// One past the last node of the subterm rooted at i.
inline std::size_t subtree_end(const Signature& sig, const Tree& t, std::size_t i) {
  std::size_t need = 1;
  while (need > 0) {
    if (i >= t.size()) throw std::invalid_argument("heuresis::programs::subtree_end: a malformed term");
    need += sig.arity(t[i].symbol);
    need -= 1;
    i++;
  }
  return i;
}

// The depth of every node from the root.
inline std::vector<std::size_t> node_depths(const Signature& sig, const Tree& t) {
  std::vector<std::size_t> depth(t.size(), 0);
  std::vector<std::pair<std::size_t, std::size_t>> open;  // (depth of children, children remaining)
  for (std::size_t i = 0; i < t.size(); i++) {
    if (!open.empty()) {
      depth[i] = open.back().first;
      if (--open.back().second == 0) open.pop_back();
    }
    const std::size_t k = sig.arity(t[i].symbol);
    if (k) open.push_back({depth[i] + 1, k});
  }
  return depth;
}

// The depth of the subterm rooted at i.
inline std::size_t depth(const Signature& sig, const Tree& t, std::size_t i = 0) {
  const std::size_t e = subtree_end(sig, t, i);
  const auto d = node_depths(sig, Tree(t.begin() + static_cast<std::ptrdiff_t>(i), t.begin() + static_cast<std::ptrdiff_t>(e)));
  return d.empty() ? 0 : *std::max_element(d.begin(), d.end());
}

// Whether the sequence is a term whose recorded sorts are those the signature computes.
inline bool well_sorted(const Signature& sig, const Tree& t) {
  std::vector<Sort> stack;
  for (std::size_t i = t.size(); i-- > 0;) {
    const auto s = t[i].symbol;
    if (s >= sig.symbols()) return false;
    const std::size_t k = sig.arity(s);
    if (stack.size() < k) return false;
    if (k == 0) {
      if (!(sig.terminal_sort(s) == t[i].sort)) return false;
      stack.push_back(t[i].sort);
      continue;
    }
    std::vector<Sort> args(k);
    for (std::size_t a = 0; a < k; a++) { args[a] = stack.back(); stack.pop_back(); }
    const auto r = sig.synthesise(s, args);
    if (!r || !(*r == t[i].sort)) return false;
    stack.push_back(*r);
  }
  return stack.size() == 1;
}

// The value of the term under an assignment of the variables (inputs[k] is the k-th variable).
inline double interpret(const Signature& sig, const Tree& t, std::span<const double> inputs) {
  std::vector<double> stack;
  stack.reserve(t.size());
  double args[8];
  for (std::size_t i = t.size(); i-- > 0;) {
    const auto s = t[i].symbol;
    if (sig.is_variable(s)) stack.push_back(inputs[sig.variable_index(s)]);
    else if (sig.is_constant(s)) stack.push_back(t[i].value);
    else {
      const std::size_t k = sig.arity(s);
      std::vector<double> big;
      double* a = args;
      if (k > 8) { big.resize(k); a = big.data(); }
      for (std::size_t j = 0; j < k; j++) { a[j] = stack.back(); stack.pop_back(); }
      stack.push_back(sig.primitive(s).apply(std::span<const double>(a, k)));
    }
  }
  return stack.back();
}

// The term written with its function symbols applied to their arguments: *(a, *(t, t)).
inline std::string to_string(const Signature& sig, const Tree& t) {
  std::string out;
  std::vector<std::size_t> remaining;
  for (std::size_t i = 0; i < t.size(); i++) {
    const auto s = t[i].symbol;
    if (sig.is_constant(s) && sig.constant(s).ephemeral) out += std::to_string(t[i].value);
    else out += sig.name(s);
    if (const std::size_t k = sig.arity(s)) {
      out += "(";
      remaining.push_back(k);
      continue;
    }
    while (!remaining.empty() && --remaining.back() == 0) { remaining.pop_back(); out += ")"; }
    if (!remaining.empty()) out += ", ";
  }
  return out;
}

}  // namespace heuresis::programs
