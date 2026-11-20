// Grant systems: a state, an alphabet of grants acting on it partially, and the words that order them (the infrastructure's
// survey Evolutionary heterogeneous resource allocation, Part II; decision 0009). An allocation made one grant at a time is a word;
// what it reaches depends on the order wherever grants do not commute.
//
// INDEPENDENCE. Two grants g and h commute at a state x when, applied in either order and each skipped where it is not allowed,
// they reach the same state, and either both orders grant both or neither does. The first condition makes the skipping decoder's
// outcome indifferent to their order; the second is the diamond of an asynchronous transition system (Bednarczyk 1987; Diekert
// and Rozenberg 1995, ch. 1), under which feasible words that are one trace reach one state. Two grants competing for the last
// unit of a stock do not commute, though both orders are infeasible: one order gives it to the first, the other to the second.
// Independence is known two ways, kept apart:
// - declared: each grant states the parts of the state it reads and writes (morphe's footprints), and two grants are independent
//   when neither writes what the other reads or writes (Mazurkiewicz 1977);
// - tested: g and h are tried in both orders at states reached by random walks from the initial state (a commutator test). A
//   failure is a witness, the word that reached the state; success on every sampled state is evidence, scoped to the states
//   tried, not proof.
// A declared independence that a test refutes is a contradiction, reported with its witness. For a finite system the question is
// decided exactly by kategoria's `resources` (its decision 0004); here it is tested, for systems of any size.
//
// DECODERS read a word as an allocation, each with what it can reach (its reach), since a decoder can exclude every optimum:
// - serial, skipping: walk the word, grant each letter when the state allows it, else drop it. Every maximal feasible word w is
//   the decoding of w followed by the other grants; an output need not be maximal, since a dropped grant may become possible later.
// - serial, deferring: at each step grant the first letter of the word still waiting that the state allows, and begin again. An
//   output is feasible and maximal, and over all orders of the alphabet the outputs are exactly the maximal feasible words of
//   distinct grants (a greedy priority rule: Kolisch 1996, "Serial and parallel resource-constrained project scheduling methods
//   revisited", EJOR 90, 320–333, for the scheme in scheduling).
// - serial, stopping: walk the word until a letter is not allowed. The outputs are the feasible prefixes.
// - stepped: in each step of time, grant in order every waiting letter the state allows, then advance the state by the system's
//   step of time. A grant never waits while it is allowed, so a plan that holds an allowed grant back for a later step is not
//   reached; with a separator letter in the word, which ends the step at once, every timed plan is the decoding of itself.

#pragma once

#include <heuresis/spaces/space.hpp>

#include <morphe/formal/traces.hpp>

#include <calx/random/stream.hpp>

#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace heuresis::sequential {

using spaces::Refusal;
using Word = std::vector<std::size_t>;

template <class S>
struct GrantSystem {
  std::size_t grants = 0;
  std::vector<std::string> names;                                  // optional, one per grant
  S initial{};
  std::function<std::optional<S>(std::size_t, const S&)> apply;    // the partial action: nothing where the grant is not allowed
  std::function<bool(const S&, const S&)> same;                    // equality of states (operator== when not given)
  std::vector<morphe::formal::traces::Footprint> footprints;       // declared, one per grant, or none
  std::function<S(const S&)> advance;                              // a step of time, for the stepped decoder

  void check() const {
    if (grants == 0) throw Refusal("a grant system needs at least one grant");
    if (!apply) throw Refusal("a grant system needs its action");
    if (!names.empty() && names.size() != grants) throw Refusal("a grant system's names must be one per grant");
    if (!footprints.empty() && footprints.size() != grants) throw Refusal("a grant system's footprints must be one per grant");
  }
  bool equal(const S& a, const S& b) const {
    if (same) return same(a, b);
    if constexpr (requires { a == b; }) return a == b;
    else throw Refusal("a grant system whose states cannot be compared: give `same`");
  }
  std::string name(std::size_t g) const { return names.empty() ? "grant " + std::to_string(g) : names[g]; }
};

// ---- applying words ----

// The state a word reaches when every letter is allowed in turn, or nothing.
template <class S>
std::optional<S> run(const GrantSystem<S>& sys, const Word& w, std::optional<S> from = std::nullopt) {
  std::optional<S> x = from ? from : std::optional<S>(sys.initial);
  for (std::size_t g : w) {
    if (g >= sys.grants) throw Refusal("a word with a letter outside the system's grants");
    x = sys.apply(g, *x);
    if (!x) return std::nullopt;
  }
  return x;
}

template <class S>
bool feasible(const GrantSystem<S>& sys, const Word& w) {
  return run(sys, w).has_value();
}

// ---- independence ----

struct IndependenceReport {
  std::size_t states = 0;                                      // states the tests were made at
  morphe::formal::traces::Independence declared;               // from footprints (all dependent when none are declared)
  morphe::formal::traces::Independence tested;                 // commuting at every state tried
  std::vector<std::vector<std::size_t>> trials;                // trials[g][h]: states at which both orders were compared
  std::vector<std::string> witnesses;                          // a pair that failed to commute and the word reaching the state
  std::vector<std::string> contradictions;                     // declared independent, refuted by a test
  bool order_may_matter() const {
    for (std::size_t a = 0; a < tested.letters; a++)
      for (std::size_t b = a + 1; b < tested.letters; b++)
        if (!tested.commute(a, b)) return true;
    return false;
  }
};

// Whether g and h commute at x: either order, each skipped where not allowed, reaches the same state, and both orders grant both
// or neither does.
template <class S>
bool commute_at(const GrantSystem<S>& sys, std::size_t g, std::size_t h, const S& x) {
  auto both = [&](std::size_t first, std::size_t second, bool& all) {
    S y = x;
    all = true;
    if (auto z = sys.apply(first, y)) y = std::move(*z);
    else all = false;
    if (auto z = sys.apply(second, y)) y = std::move(*z);
    else all = false;
    return y;
  };
  bool all_gh = false, all_hg = false;
  const S gh = both(g, h, all_gh), hg = both(h, g, all_hg);
  return all_gh == all_hg && sys.equal(gh, hg);
}

// Commutator tests at the states reached by `walks` random walks from the initial state (each of a random length up to
// `depth`, through allowed grants), and the declared independence beside them.
template <class S>
IndependenceReport independence(const GrantSystem<S>& sys, calx::random::Stream& s, std::size_t walks = 100, std::size_t depth = 0) {
  sys.check();
  const std::size_t k = sys.grants;
  if (depth == 0) depth = k;
  IndependenceReport r;
  if (!sys.footprints.empty()) r.declared = morphe::formal::traces::independence(sys.footprints);
  else r.declared = morphe::formal::traces::independence(k, {});
  std::vector<std::vector<char>> fails(k, std::vector<char>(k, 0));
  r.trials.assign(k, std::vector<std::size_t>(k, 0));
  auto words_of = [&](const Word& w) {
    std::string t = "⟨";
    for (std::size_t i = 0; i < w.size(); i++) t += (i ? " " : "") + sys.name(w[i]);
    return t + "⟩";
  };
  for (std::size_t walk = 0; walk < walks; walk++) {
    S x = sys.initial;
    Word path;
    const std::size_t length = static_cast<std::size_t>(s.below(depth + 1));
    for (std::size_t step = 0;; step++) {
      r.states++;
      for (std::size_t g = 0; g < k; g++)
        for (std::size_t h = g + 1; h < k; h++) {
          if (!sys.apply(g, x) && !sys.apply(h, x)) continue;  // neither is allowed here: nothing to compare
          r.trials[g][h]++, r.trials[h][g]++;
          const bool commute = commute_at(sys, g, h, x);
          if (!commute && !fails[g][h]) {
            fails[g][h] = fails[h][g] = 1;
            const std::string w = sys.name(g) + " and " + sys.name(h) + " fail to commute after " + words_of(path);
            r.witnesses.push_back(w);
            if (r.declared.commute(g, h)) r.contradictions.push_back("declared independent, but " + w);
          }
        }
      if (step >= length) break;
      Word allowed;
      for (std::size_t g = 0; g < k; g++)
        if (sys.apply(g, x)) allowed.push_back(g);
      if (allowed.empty()) break;
      const std::size_t g = allowed[s.below(allowed.size())];
      x = *sys.apply(g, x);
      path.push_back(g);
    }
  }
  std::vector<std::pair<std::size_t, std::size_t>> commuting;
  for (std::size_t g = 0; g < k; g++)
    for (std::size_t h = g + 1; h < k; h++)
      if (!fails[g][h]) commuting.push_back({g, h});
  r.tested = morphe::formal::traces::independence(k, commuting);
  return r;
}

// ---- decoders ----

enum class Unmet { skip, defer, stop };

inline const char* name(Unmet u) {
  switch (u) {
    case Unmet::skip: return "skip";
    case Unmet::defer: return "defer";
    case Unmet::stop: return "stop";
  }
  return "";
}

struct Reach {
  std::string claim;
  bool maximal = false;          // every output is maximal: no grant left that the final state allows
  bool every_maximal = false;    // every maximal feasible word of distinct grants is an output of some order
};

inline Reach reach(Unmet u) {
  switch (u) {
    case Unmet::skip: return {"feasible words; every maximal feasible word is reached; an output need not be maximal", false, true};
    case Unmet::defer: return {"exactly the maximal feasible words of distinct grants", true, true};
    case Unmet::stop: return {"the feasible prefixes of the word", false, true};
  }
  return {};
}

template <class S>
struct Decoded {
  S state;
  Word granted;   // the letters granted, in the order granted
  Word unmet;     // the letters never granted, in the order of the word
};

template <class S>
Decoded<S> serial(const GrantSystem<S>& sys, const Word& w, Unmet rule = Unmet::skip) {
  sys.check();
  for (std::size_t g : w)
    if (g >= sys.grants) throw Refusal("a word with a letter outside the system's grants");
  Decoded<S> d{sys.initial, {}, {}};
  if (rule == Unmet::defer) {
    std::vector<std::size_t> waiting = w;
    for (;;) {
      bool granted = false;
      for (std::size_t i = 0; i < waiting.size(); i++)
        if (auto y = sys.apply(waiting[i], d.state)) {
          d.state = std::move(*y);
          d.granted.push_back(waiting[i]);
          waiting.erase(waiting.begin() + static_cast<std::ptrdiff_t>(i));
          granted = true;
          break;
        }
      if (!granted) break;
    }
    d.unmet = waiting;
    return d;
  }
  for (std::size_t i = 0; i < w.size(); i++) {
    if (auto y = sys.apply(w[i], d.state)) {
      d.state = std::move(*y);
      d.granted.push_back(w[i]);
    } else if (rule == Unmet::stop) {
      d.unmet.insert(d.unmet.end(), w.begin() + static_cast<std::ptrdiff_t>(i), w.end());
      return d;
    } else {
      d.unmet.push_back(w[i]);
    }
  }
  return d;
}

// The stepped decoder: `steps` steps of time; `separator`, if given, is a letter outside the grants that ends the step.
template <class S>
struct Timed {
  S state;
  std::vector<Word> granted;   // the letters granted in each step
  Word unmet;
};

template <class S>
Timed<S> stepped(const GrantSystem<S>& sys, const Word& w, std::size_t steps, std::optional<std::size_t> separator = std::nullopt) {
  sys.check();
  if (!sys.advance) throw Refusal("the stepped decoder needs the system's step of time");
  if (separator && *separator < sys.grants) throw Refusal("a separator must lie outside the system's grants");
  for (std::size_t g : w)
    if (g >= sys.grants && (!separator || g != *separator)) throw Refusal("a word with a letter outside the system's grants");
  Timed<S> t{sys.initial, {}, {}};
  std::vector<std::size_t> waiting = w;
  for (std::size_t step = 0; step < steps; step++) {
    Word now;
    std::vector<std::size_t> later;
    bool ended = false;
    for (std::size_t g : waiting) {
      if (ended) {
        later.push_back(g);
        continue;
      }
      if (separator && g == *separator) {
        ended = true;  // the separator is spent: the rest waits for the next step
        continue;
      }
      if (auto y = sys.apply(g, t.state)) t.state = std::move(*y), now.push_back(g);
      else later.push_back(g);
    }
    t.granted.push_back(now);
    waiting = later;
    t.state = sys.advance(t.state);
  }
  for (std::size_t g : waiting)
    if (!separator || g != *separator) t.unmet.push_back(g);
  return t;
}

}  // namespace heuresis::sequential
