// Programs as derivations of a context-free grammar: grammar-guided genetic programming and grammatical
// evolution.
//
// Given a context-free grammar Γ = (V, Σ, R, S) (morphe::formal::Grammar), the programs are the words of L(Γ), and
// a genotype is either
//
// - a derivation tree, searched directly: random derivations of bounded depth, crossover exchanging subtrees rooted
//   at the same nonterminal, mutation regrowing a subtree from its nonterminal (Whigham 1995, "Grammatically-based
//   genetic programming", Workshop on Genetic Programming, ML'95, 33–41). Every yield is a word of L(Γ), since a
//   subtree rooted at A may be replaced by any derivation from A;
// - a sequence of codons, read through the leftmost derivation from S: at each step the leftmost nonterminal A is
//   expanded by its (c mod k)-th rule, c the next codon and k the number of A's rules, a codon being read only where
//   A has a choice; when the codons run out the reading wraps to the first, at most `max_wraps` times, after which
//   the genotype maps to no word (Ryan, Collins and O'Neill 1998, "Grammatical evolution: evolving programs for an
//   arbitrary language", EuroGP, LNCS 1391, 83–96; O'Neill and Ryan 2001, "Grammatical evolution", IEEE Trans. Evol.
//   Comput. 5, 349–358). The genotype–phenotype map is many-to-one (codons beyond the derivation, and residues of
//   the same class modulo k, are neutral), so the codon space has large neutral networks; the codons are integer
//   vectors, varied by variation/integer.hpp and the crossovers of variation/generic.hpp.
//
// A derivation tree is stored in prefix order: a nonterminal node records the rule applied, and its children are
// the symbols of that rule's right side in order; a terminal is a leaf. The prefix order is the order of a leftmost
// derivation, so a grammatical-evolution mapping writes its tree directly.
//
// The least depth of a derivation from each nonterminal, m(A) = min over rules A → α of 1 + max over the
// nonterminals B of α of m(B) (1 when α has none), is the least fixpoint of those equations; a rule is usable with a
// depth budget d when 1 + max m(B) ≤ d. A nonterminal with no finite m derives no word.

#pragma once

#include <morphe/formal/grammar.hpp>

#include <calx/random/stream.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace heuresis::programs {

using morphe::formal::Grammar;
using morphe::formal::Symbol;

struct DerivationNode {
  Symbol symbol = 0;
  std::int32_t rule = -1;  // the rule applied; −1 for a terminal leaf
};

using Derivation = std::vector<DerivationNode>;

class GrammarSpace {
 public:
  explicit GrammarSpace(const Grammar& g) : g_(&g), rules_(g.nonterminals.size()) {
    g.validate();
    if (!g.context_free()) throw std::invalid_argument("heuresis::programs::GrammarSpace: the grammar is not context-free");
    for (std::size_t r = 0; r < g.rules.size(); r++) rules_[g.index(g.rules[r].lhs[0])].push_back(r);
    constexpr std::size_t inf = std::numeric_limits<std::size_t>::max();
    least_.assign(g.nonterminals.size(), inf);
    rule_least_.assign(g.rules.size(), inf);
    for (bool changed = true; changed;) {
      changed = false;
      for (std::size_t r = 0; r < g.rules.size(); r++) {
        std::size_t m = 1;
        for (Symbol s : g.rules[r].rhs) {
          if (g.terminal(s)) continue;
          m = least_[g.index(s)] == inf ? inf : std::max(m, 1 + least_[g.index(s)]);
          if (m == inf) break;
        }
        rule_least_[r] = m;
        std::size_t& a = least_[g.index(g.rules[r].lhs[0])];
        if (m < a) { a = m; changed = true; }
      }
    }
  }

  const Grammar& grammar() const { return *g_; }
  // The least depth of a derivation from the nonterminal A (a terminal has depth 0).
  std::size_t least_depth(Symbol a) const { return g_->terminal(a) ? 0 : least_[g_->index(a)]; }
  const std::vector<std::size_t>& rules(Symbol a) const { return rules_[g_->index(a)]; }

  // A random derivation from A of depth ≤ d, each rule chosen uniformly among those usable within the budget.
  Derivation grow(Symbol a, std::size_t d, calx::random::Stream& s) const {
    if (least_depth(a) > d) throw std::invalid_argument("heuresis::programs::GrammarSpace::grow: no derivation within this depth");
    Derivation out;
    make(a, d, s, out);
    return out;
  }

  std::size_t subtree_end(const Derivation& t, std::size_t i) const {
    std::size_t need = 1;
    while (need > 0) {
      need += children(t[i]);
      need -= 1;
      i++;
    }
    return i;
  }

  // The level of every node: the number of rule applications above it.
  std::vector<std::size_t> levels(const Derivation& t) const {
    std::vector<std::size_t> lv(t.size(), 0);
    std::vector<std::pair<std::size_t, std::size_t>> open;
    for (std::size_t j = 0; j < t.size(); j++) {
      if (!open.empty()) {
        lv[j] = open.back().first;
        if (--open.back().second == 0) open.pop_back();
      }
      if (const std::size_t k = children(t[j])) open.push_back({lv[j] + 1, k});
    }
    return lv;
  }

  // The depth of the subtree at i: the most rule applications on a path down from it.
  std::size_t depth(const Derivation& t, std::size_t i = 0) const {
    const std::size_t e = subtree_end(t, i);
    const auto lv = levels(Derivation(t.begin() + static_cast<std::ptrdiff_t>(i), t.begin() + static_cast<std::ptrdiff_t>(e)));
    std::size_t best = 0;
    for (std::size_t j = 0; j < lv.size(); j++) best = std::max(best, t[i + j].rule < 0 ? lv[j] : lv[j] + 1);
    return best;
  }

  // The word the derivation yields.
  std::vector<Symbol> yield(const Derivation& t) const {
    std::vector<Symbol> w;
    for (const auto& n : t)
      if (n.rule < 0) w.push_back(n.symbol);
    return w;
  }

  // Exchanges subtrees rooted at one nonterminal; a child deeper than max_depth is replaced by its parent.
  std::pair<Derivation, Derivation> crossover(const Derivation& a, const Derivation& b, std::size_t max_depth, calx::random::Stream& s) const {
    std::vector<std::size_t> inner;
    for (std::size_t i = 0; i < a.size(); i++)
      if (a[i].rule >= 0) inner.push_back(i);
    if (inner.empty()) return {a, b};
    const std::size_t i = inner[s.below(inner.size())];
    std::vector<std::size_t> match;
    for (std::size_t j = 0; j < b.size(); j++)
      if (b[j].rule >= 0 && b[j].symbol == a[i].symbol) match.push_back(j);
    if (match.empty()) return {a, b};
    const std::size_t j = match[s.below(match.size())];
    const std::size_t ie = subtree_end(a, i), je = subtree_end(b, j);
    Derivation c1(a.begin(), a.begin() + static_cast<std::ptrdiff_t>(i)), c2(b.begin(), b.begin() + static_cast<std::ptrdiff_t>(j));
    c1.insert(c1.end(), b.begin() + static_cast<std::ptrdiff_t>(j), b.begin() + static_cast<std::ptrdiff_t>(je));
    c1.insert(c1.end(), a.begin() + static_cast<std::ptrdiff_t>(ie), a.end());
    c2.insert(c2.end(), a.begin() + static_cast<std::ptrdiff_t>(i), a.begin() + static_cast<std::ptrdiff_t>(ie));
    c2.insert(c2.end(), b.begin() + static_cast<std::ptrdiff_t>(je), b.end());
    if (depth(c1) > max_depth) c1 = a;
    if (depth(c2) > max_depth) c2 = b;
    return {std::move(c1), std::move(c2)};
  }

  // Regrows the subtree of a random nonterminal node within the depth that remains.
  Derivation mutate(const Derivation& t, std::size_t max_depth, calx::random::Stream& s) const {
    std::vector<std::size_t> inner;
    for (std::size_t i = 0; i < t.size(); i++)
      if (t[i].rule >= 0) inner.push_back(i);
    if (inner.empty()) return t;
    const std::size_t i = inner[s.below(inner.size())];
    const std::size_t level = levels(t)[i];
    const std::size_t d = max_depth > level ? max_depth - level : 0;
    if (least_depth(t[i].symbol) > d) return t;
    Derivation fresh = grow(t[i].symbol, d, s);
    Derivation c(t.begin(), t.begin() + static_cast<std::ptrdiff_t>(i));
    c.insert(c.end(), fresh.begin(), fresh.end());
    c.insert(c.end(), t.begin() + static_cast<std::ptrdiff_t>(subtree_end(t, i)), t.end());
    return depth(c) <= max_depth ? c : t;
  }

 private:
  std::size_t children(const DerivationNode& n) const { return n.rule < 0 ? 0 : g_->rules[static_cast<std::size_t>(n.rule)].rhs.size(); }

  void make(Symbol a, std::size_t d, calx::random::Stream& s, Derivation& out) const {
    if (g_->terminal(a)) {
      out.push_back({a, -1});
      return;
    }
    std::vector<std::size_t> usable;
    for (auto r : rules_[g_->index(a)])
      if (rule_least_[r] <= d) usable.push_back(r);
    const std::size_t r = usable[s.below(usable.size())];
    out.push_back({a, static_cast<std::int32_t>(r)});
    for (Symbol b : g_->rules[r].rhs) make(b, d - 1, s, out);
  }

  const Grammar* g_;
  std::vector<std::vector<std::size_t>> rules_;
  std::vector<std::size_t> least_, rule_least_;
};

// The result of grammatical evolution's mapping.
struct Mapping {
  bool valid = false;
  std::vector<Symbol> word;
  Derivation derivation;
  std::size_t codons_read = 0, wraps = 0;
};

// Maps codons through the leftmost derivation from the start symbol; at most max_nodes nodes are expanded.
inline Mapping map_codons(const Grammar& g, std::span<const std::int64_t> codons, std::size_t max_wraps = 2, std::size_t max_nodes = 100000) {
  if (!g.context_free()) throw std::invalid_argument("heuresis::programs::map_codons: the grammar is not context-free");
  std::vector<std::vector<std::size_t>> rules(g.nonterminals.size());
  for (std::size_t r = 0; r < g.rules.size(); r++) rules[g.index(g.rules[r].lhs[0])].push_back(r);
  Mapping m;
  std::vector<Symbol> stack{g.start()};
  std::size_t pos = 0;
  while (!stack.empty()) {
    if (m.derivation.size() >= max_nodes) return m;
    const Symbol a = stack.back();
    stack.pop_back();
    if (g.terminal(a)) {
      m.word.push_back(a);
      m.derivation.push_back({a, -1});
      continue;
    }
    const auto& choices = rules[g.index(a)];
    if (choices.empty()) return m;
    std::size_t r = choices[0];
    if (choices.size() > 1) {
      if (codons.empty()) return m;
      if (pos == codons.size()) {
        if (m.wraps == max_wraps) return m;
        m.wraps++;
        pos = 0;
      }
      const std::int64_t c = codons[pos++];
      m.codons_read++;
      r = choices[static_cast<std::size_t>(((c % static_cast<std::int64_t>(choices.size())) + static_cast<std::int64_t>(choices.size())) % static_cast<std::int64_t>(choices.size()))];
    }
    m.derivation.push_back({a, static_cast<std::int32_t>(r)});
    const auto& rhs = g.rules[r].rhs;
    for (auto it = rhs.rbegin(); it != rhs.rend(); ++it) stack.push_back(*it);
  }
  m.valid = true;
  return m;
}

}  // namespace heuresis::programs
