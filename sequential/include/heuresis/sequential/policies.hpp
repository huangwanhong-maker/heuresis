// Policies against sequences: a rule that decides each grant from the state, learned once for a family of instances, set beside
// the word searched for each instance (decision 0011; the survey of evolutionary allocation, Part I and Part II, gaps: no study
// compared evolved sequences against evolved policies on one allocation problem).
//
// A SEQUENCE is a word: what one instance should grant, in order, found by searching that instance. A POLICY is a score over the
// features of a grant at a state; followed, it grants at each step the allowed grant of greatest score (each at most once, ties
// to the least index), until none is allowed or the greatest score falls below a declared threshold, and so decodes any instance
// of a family into a word, with no search of its own. The two are dispatching rules against schedules, in the language of
// scheduling (Branke, Nguyen, Pickardt and Zhang 2016, "Automated design of production scheduling heuristics: a review", IEEE
// Trans. Evol. Comput. 20, 110–124). A policy may be linear in the features, its weights found by CMA-ES over the family's
// training instances, or a program over them, found by genetic programming (Koza 1992; dispatching rules evolved as programs
// since Miyashita 2000).
//
// What each can claim. A sequence searched for an instance is the best found for it; on another instance it means nothing. A
// policy is worse than the best word of each instance it is set on (followed, it decodes into a word), and its worth on instances
// it was not trained on is its generalisation, measured on held-out instances of the family, never on its training ones. The
// comparison reports both against each instance's best word, found by enumeration where the instance is small.
#pragma once

#include <heuresis/programs/genetic_programming.hpp>
#include <heuresis/programs/trees.hpp>
#include <heuresis/search/run.hpp>
#include <heuresis/sequential/grants.hpp>
#include <heuresis/strategies/cma_es.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <vector>

namespace heuresis::sequential {

template <class S>
using Features = std::function<std::vector<double>(std::size_t, const S&)>;
using Score = std::function<double(const std::vector<double>&)>;

// An instance of a family: its grant system, the features of a grant at a state, and the worth of a final state (maximised).
template <class S>
struct Instance {
  GrantSystem<S> system;
  Features<S> features;
  std::function<double(const S&)> worth;
};

// Follows a policy through an instance's grant system.
template <class S>
Decoded<S> follow(const GrantSystem<S>& sys, const Features<S>& features, const Score& score, double threshold = -std::numeric_limits<double>::infinity()) {
  sys.check();
  if (!features || !score) throw Refusal("a policy needs features and a score");
  Decoded<S> d{sys.initial, {}, {}};
  std::vector<char> done(sys.grants, 0);
  for (;;) {
    std::size_t pick = sys.grants;
    double best = -std::numeric_limits<double>::infinity();
    std::optional<S> next;
    for (std::size_t g = 0; g < sys.grants; g++) {
      if (done[g]) continue;
      auto y = sys.apply(g, d.state);
      if (!y) continue;
      const double v = score(features(g, d.state));
      if (std::isfinite(v) && (pick == sys.grants || v > best)) pick = g, best = v, next = std::move(y);
    }
    if (pick == sys.grants || best < threshold) break;
    d.state = std::move(*next);
    d.granted.push_back(pick);
    done[pick] = 1;
  }
  for (std::size_t g = 0; g < sys.grants; g++)
    if (!done[g]) d.unmet.push_back(g);
  return d;
}

inline Score linear(std::vector<double> w) {
  return [w](const std::vector<double>& f) {
    if (f.size() != w.size()) throw Refusal("a linear policy has one weight per feature");
    double s = 0;
    for (std::size_t k = 0; k < w.size(); k++) s += w[k] * f[k];
    return s;
  };
}

inline Score program(const programs::Signature& sig, programs::Tree t) {
  return [&sig, t](const std::vector<double>& f) { return programs::interpret(sig, t, f); };
}

template <class S>
double worth(const Instance<S>& in, const Score& score, double threshold = -std::numeric_limits<double>::infinity()) {
  return in.worth(follow(in.system, in.features, score, threshold).state);
}

template <class S>
double mean_worth(const std::vector<Instance<S>>& family, const Score& score) {
  if (family.empty()) throw Refusal("a family needs at least one instance");
  double s = 0;
  for (const auto& in : family) s += worth(in, score);
  return s / static_cast<double>(family.size());
}

// ---- learning a policy over a family ------------------------------------------------------------------------------------

struct LinearTrained {
  std::vector<double> weights;
  double train = 0;           // mean worth on the training instances
  double evaluations = 0;     // policies tried
};

template <class S>
LinearTrained train_linear(const std::vector<Instance<S>>& train, std::size_t features, double evaluations, std::uint64_t seed) {
  if (features == 0) throw Refusal("a linear policy needs at least one feature");
  strategies::Cma cma(std::vector<double>(features, 0.0), 1.0, {}, search::Draws(seed));
  auto out = search::run(cma, [&](const std::vector<double>& w) { return -mean_worth(train, linear(w)); }, search::Budget{evaluations});
  LinearTrained T;
  if (out.best) T.weights = out.best->genotype, T.train = -out.best->evaluation.objectives[0];
  T.evaluations = out.evaluations;
  return T;
}

struct ProgramTrained {
  programs::Tree tree;
  double train = 0, evaluations = 0;
};

template <class S>
ProgramTrained train_program(const std::vector<Instance<S>>& train, const programs::Signature& sig, double evaluations, std::uint64_t seed, programs::GpOptions o = {}) {
  programs::GeneticProgramming gp(sig, o, search::Draws(seed));
  auto out = search::run(gp, [&](const programs::Tree& t) { return -mean_worth(train, program(sig, t)); }, search::Budget{evaluations});
  ProgramTrained T;
  if (out.best) T.tree = out.best->genotype, T.train = -out.best->evaluation.objectives[0];
  T.evaluations = out.evaluations;
  return T;
}

// ---- searching each instance's sequence -------------------------------------------------------------------------------------

// Random keys searched by CMA-ES, read as an order by sorting and decoded by the deferring rule (whose outputs over all orders
// are exactly the maximal feasible words); the best worth found within the evaluations.
template <class S>
double search_sequence(const Instance<S>& in, double evaluations, std::uint64_t seed, Word* best = nullptr) {
  const std::size_t n = in.system.grants;
  auto order = [n](const std::vector<double>& keys) {
    Word w(n);
    std::iota(w.begin(), w.end(), std::size_t{0});
    std::stable_sort(w.begin(), w.end(), [&](std::size_t a, std::size_t b) { return keys[a] < keys[b]; });
    return w;
  };
  strategies::Cma cma(std::vector<double>(n, 0.0), 1.0, {}, search::Draws(seed));
  auto out = search::run(cma, [&](const std::vector<double>& k) { return -in.worth(serial(in.system, order(k), Unmet::defer).state); }, search::Budget{evaluations});
  if (!out.best) return -std::numeric_limits<double>::infinity();
  if (best) *best = serial(in.system, order(out.best->genotype), Unmet::defer).granted;
  return -out.best->evaluation.objectives[0];
}

// The best worth over every order of the grants (small instances only), by the deferring rule.
template <class S>
double best_order(const Instance<S>& in, Word* best = nullptr, std::size_t most = 9) {
  const std::size_t n = in.system.grants;
  if (n > most) throw Refusal("enumerating every order of " + std::to_string(n) + " grants is beyond the limit of " + std::to_string(most));
  Word w(n);
  std::iota(w.begin(), w.end(), std::size_t{0});
  double top = -std::numeric_limits<double>::infinity();
  do {
    const auto d = serial(in.system, w, Unmet::defer);
    const double v = in.worth(d.state);
    if (v > top) {
      top = v;
      if (best) *best = d.granted;
    }
  } while (std::next_permutation(w.begin(), w.end()));
  return top;
}

// ---- the comparison -----------------------------------------------------------------------------------------------------------

struct Comparison {
  double best = 0;                 // mean of each test instance's best word
  double sequence = 0;             // mean of the sequence searched for each test instance
  double linear_test = 0, linear_train = 0;
  double program_test = 0, program_train = 0;
  double sequence_evaluations = 0; // per test instance
  double policy_evaluations = 0;   // once, for the family (each a pass over the training instances)
};

}  // namespace heuresis::sequential
