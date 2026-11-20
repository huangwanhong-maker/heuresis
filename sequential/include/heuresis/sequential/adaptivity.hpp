// A sequence, or a policy (decision 0009; the survey, Part II, §7). When the outcome of each grant is random and observed as it
// happens, a policy that chooses the next grant from what has happened can do better than any fixed sequence. The ratio of the best
// policy's value to the best sequence's is the adaptivity gap (Dean, Goemans and Vondrák 2008, "Approximating the stochastic
// knapsack problem: the benefit of adaptivity", Math. Oper. Res. 33(4), 945–964, who prove it at most 4 for the stochastic
// knapsack). Over sampled scenarios the gap is bracketed without solving for the best policy:
//   best common sequence  ≤  best policy  ≤  mean over scenarios of the best sequence for each (clairvoyance),
// the first since a sequence is a policy that ignores what it sees, the last since a policy cannot know a scenario before it
// unfolds. The bracket says whether a policy could be worth searching for: a ratio near one says a sequence suffices.
//
// Values here are maximised, as the gap is defined; the sequence is found by enumeration of every order of the items (at most
// nine), or among orders a caller proposes.

#pragma once

#include <heuresis/sequential/grants.hpp>

#include <algorithm>
#include <functional>
#include <limits>
#include <numeric>
#include <vector>

namespace heuresis::sequential {

struct AdaptivityBracket {
  double sequence = 0;      // the mean value of the best common sequence
  double clairvoyant = 0;   // the mean over scenarios of the best value for each
  Word best;                // the best common sequence
  std::size_t scenarios = 0, orders = 0;
  double ratio() const { return sequence > 0 ? clairvoyant / sequence : std::numeric_limits<double>::infinity(); }
};

template <class Scenario>
AdaptivityBracket adaptivity_bracket(const std::vector<Word>& orders, const std::vector<Scenario>& scenarios,
                                     const std::function<double(const Word&, const Scenario&)>& value) {
  if (orders.empty() || scenarios.empty()) throw Refusal("an adaptivity bracket needs orders and scenarios");
  AdaptivityBracket b;
  b.scenarios = scenarios.size(), b.orders = orders.size();
  std::vector<double> best_each(scenarios.size(), -std::numeric_limits<double>::infinity());
  b.sequence = -std::numeric_limits<double>::infinity();
  for (const auto& w : orders) {
    double mean = 0;
    for (std::size_t i = 0; i < scenarios.size(); i++) {
      const double v = value(w, scenarios[i]);
      mean += v;
      best_each[i] = std::max(best_each[i], v);
    }
    mean /= static_cast<double>(scenarios.size());
    if (mean > b.sequence) b.sequence = mean, b.best = w;
  }
  for (double v : best_each) b.clairvoyant += v;
  b.clairvoyant /= static_cast<double>(scenarios.size());
  return b;
}

// Every order of k items (k ≤ 9).
template <class Scenario>
AdaptivityBracket adaptivity_bracket(std::size_t items, const std::vector<Scenario>& scenarios, const std::function<double(const Word&, const Scenario&)>& value) {
  if (items == 0 || items > 9) throw Refusal("an adaptivity bracket enumerates the orders of one to nine items; give the orders otherwise");
  std::vector<Word> orders;
  Word w(items);
  std::iota(w.begin(), w.end(), 0);
  do orders.push_back(w);
  while (std::next_permutation(w.begin(), w.end()));
  return adaptivity_bracket(orders, scenarios, value);
}

}  // namespace heuresis::sequential
