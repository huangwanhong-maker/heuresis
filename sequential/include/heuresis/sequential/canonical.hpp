// Orders equal up to exchanges of commuting grants, evaluated once (decision 0009).
//
// Two words that differ only by exchanging adjacent independent grants reach the same state from any state where both are
// feasible (they are one trace; Diekert and Rozenberg 1995, ch. 1). A search over orders therefore keys what it evaluates by the
// trace of the word a genotype decodes to, its lexicographic normal form (morphe's `traces`, its decision 0021): the evolutionary
// loop evaluates a genotype once per key, so orders that differ only by such exchanges cost one evaluation. The key must be taken
// of the word actually granted, since a decoder that skips a grant can make two orders of independent grants grant different words.
//
// The redundancy of a representation of orders is the number of words per trace: k! permutations of k grants fall into as many
// traces as the causal orders admit, each with as many words as its linearisations.

#pragma once

#include <heuresis/sequential/grants.hpp>

#include <heuresis/evolution/evolution.hpp>

#include <map>
#include <set>
#include <string>
#include <vector>

namespace heuresis::sequential {

using morphe::formal::traces::Independence;

inline std::string trace_key(const Word& w, const Independence& I) {
  const Word f = morphe::formal::traces::lexicographic_normal_form(w, I);
  std::string k = "trace(";
  for (std::size_t g : f) k += std::to_string(g) + ",";
  return k + ")";
}

// Operators whose key is the trace of the word a genotype decodes to.
template <class G>
evolution::Operators<G> keyed_by_trace(evolution::Operators<G> ops, Independence I, std::function<Word(const G&)> word) {
  if (!word) throw Refusal("keying by trace needs the word a genotype decodes to");
  ops.key = [I = std::move(I), word = std::move(word)](const G& g) { return trace_key(word(g), I); };
  return ops;
}

// A cache of evaluations keyed by trace, counting what it saved.
template <class V>
class TraceCache {
 public:
  explicit TraceCache(Independence I) : I_(std::move(I)) {}
  template <class F>
  const V& evaluate(const Word& w, F&& f) {
    const std::string k = trace_key(w, I_);
    auto it = values_.find(k);
    if (it != values_.end()) {
      hits_++;
      return it->second;
    }
    misses_++;
    return values_.emplace(k, f(w)).first->second;
  }
  std::size_t hits() const { return hits_; }
  std::size_t misses() const { return misses_; }

 private:
  Independence I_;
  std::map<std::string, V> values_;
  std::size_t hits_ = 0, misses_ = 0;
};

struct Redundancy {
  std::size_t words = 0, traces = 0;
  double ratio() const { return traces ? static_cast<double>(words) / static_cast<double>(traces) : 0; }
};

inline Redundancy redundancy(const std::vector<Word>& words, const Independence& I) {
  std::set<std::string> keys;
  for (const auto& w : words) keys.insert(trace_key(w, I));
  return {words.size(), keys.size()};
}

// Every order of k distinct grants (k ≤ 10), with its redundancy.
inline Redundancy redundancy_of_orders(const Independence& I) {
  if (I.letters > 10) throw Refusal("the redundancy of all orders is enumerated for at most ten grants");
  Word w(I.letters);
  for (std::size_t i = 0; i < w.size(); i++) w[i] = i;
  std::set<std::string> keys;
  std::size_t words = 0;
  do keys.insert(trace_key(w, I)), words++;
  while (std::next_permutation(w.begin(), w.end()));
  return {words, keys.size()};
}

}  // namespace heuresis::sequential
