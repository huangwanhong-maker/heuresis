// The strategies of an evolutionary loop, one choice at every stage (decision 0006). Each is a value with its parameters, read from
// its name by `*_of`, so that a caller (a page, a script, a request) states a whole algorithm as data.
//
// Stages, and what each choice means (sources in evolution.hpp):
//   scoring      how members are ordered when one order is needed: the first objective by Deb's rule; a weighted sum; the
//                objectives in order; Pareto fronts with crowding (NSGA-II)
//   parents      uniform; tournament of k; roulette on the score; linear ranking; Boltzmann; truncation; lexicase and ε-lexicase
//                over the objectives as cases; a tournament on novelty
//   schedule     how an operator (a generator, a mutation, a crossover) is chosen for each child: fixed weights; probability
//                matching; adaptive pursuit; UCB1; Thompson sampling
//   credit       what an operator is rewarded for: a child better than its best parent; a child that survives; the improvement
//   survival     generational with elites; (μ+λ); (μ,λ); steady state replacing the worst; aging (the oldest removed: regularized
//                evolution); deterministic crowding; restricted tournament; clearing; NSGA-II; ALPS age layers; MAP-Elites niches
//   novelty      none, or the novelty of a member (mean distance to its k nearest) as a further objective
//   archive      admission by a judge the caller supplies; retirement after long inefficacy, by age, or on recheck
//   termination  generations, evaluations, stagnation; a restart on stagnation
#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace heuresis::evolution {

enum class Scoring { first, weighted, lexicographic, pareto };
enum class Parents { uniform, tournament, roulette, ranking, boltzmann, truncation, lexicase, epsilon_lexicase, novelty };
enum class Schedule { fixed, probability_matching, adaptive_pursuit, ucb, thompson };
enum class Credit { better_than_parent, survival, improvement };
enum class Survival { generational, plus, comma, replace_worst, aging, deterministic_crowding, restricted_tournament, clearing, nsga2, alps, niches };

namespace detail {
template <class E>
struct Names;
template <> struct Names<Scoring> { static constexpr const char* all[] = {"first", "weighted", "lexicographic", "pareto"}; };
template <> struct Names<Parents> { static constexpr const char* all[] = {"uniform", "tournament", "roulette", "ranking", "boltzmann", "truncation", "lexicase", "epsilon-lexicase", "novelty"}; };
template <> struct Names<Schedule> { static constexpr const char* all[] = {"fixed", "probability-matching", "adaptive-pursuit", "ucb", "thompson"}; };
template <> struct Names<Credit> { static constexpr const char* all[] = {"better-than-parent", "survival", "improvement"}; };
template <> struct Names<Survival> { static constexpr const char* all[] = {"generational", "plus", "comma", "replace-worst", "aging", "deterministic-crowding", "restricted-tournament", "clearing", "nsga2", "alps", "niches"}; };
}  // namespace detail

template <class E>
inline const char* name(E e) { return detail::Names<E>::all[static_cast<std::size_t>(e)]; }

template <class E>
inline E of(const std::string& s) {
  const auto& all = detail::Names<E>::all;
  std::string known;
  for (std::size_t k = 0; k < std::size(all); k++) {
    if (s == all[k]) return static_cast<E>(k);
    known += (k ? ", " : "") + std::string(all[k]);
  }
  throw std::invalid_argument("there is no strategy '" + s + "' here (there are " + known + ")");
}

template <class E>
inline std::vector<std::string> names() {
  std::vector<std::string> out;
  for (const char* n : detail::Names<E>::all) out.push_back(n);
  return out;
}

struct Strategy {
  // Sizes.
  std::size_t mu = 40;       // the population kept
  std::size_t lambda = 40;   // children a generation (steady-state strategies take them one at a time)

  // Ordering.
  Scoring scoring = Scoring::first;
  std::vector<double> weights;          // for weighted: one per objective (1 each when empty)
  double lexicographic_tolerance = 0;   // for lexicographic: objectives closer than this tie

  // Parents.
  Parents parents = Parents::tournament;
  std::size_t tournament = 3;
  double ranking_pressure = 1.8;        // linear ranking's s in [1, 2]
  double temperature = 1;               // Boltzmann's T, in units of the score's spread
  double truncation = 0.5;              // the share of the best that parents are drawn from
  double from_archive = 0.1;            // the chance that a parent is drawn from the archive instead

  // Operators.
  Schedule schedule = Schedule::fixed;
  Credit credit = Credit::better_than_parent;
  double crossover_rate = 0.6;          // for fixed: the chance of a crossover, the rest split among mutations and generators
  double generator_rate = 0.1;          // for fixed: the share of children made new by generators
  double p_min = 0.05;                  // the least probability an adaptive schedule keeps for any operator
  double adaptation = 0.3;              // probability matching's and adaptive pursuit's rate
  double exploration = 1.4;             // UCB1's c
  std::size_t retries = 8;              // tries to make a child not already seen

  // Survival.
  Survival survival = Survival::plus;
  std::size_t elites = 2;               // for generational
  std::size_t window = 8;               // for restricted tournament
  double radius = 0.2;                  // for clearing: the niche's radius in the representation's distance
  std::size_t capacity = 1;             // for clearing: winners a niche keeps
  std::size_t layers = 4;               // for ALPS
  std::size_t age_gap = 4;              // for ALPS: the bottom layer's age limit and how often it is reseeded
  std::vector<std::size_t> bins;        // for niches: cells along each descriptor
  std::vector<double> lower, upper;     // for niches: the descriptors' ranges

  // Novelty.
  bool novelty = false;                 // add the novelty as a further objective (minimised as its negative)
  std::size_t novelty_k = 5;

  // Archive.
  std::size_t archive_capacity = 200;
  std::size_t retire_after = 10;        // generations an archived member may go without producing a child that survives
  std::size_t retire_age = 0;           // generations an archived member may stay at all (0: no limit)
  bool recheck = true;                  // ask the caller's recheck, when given, every generation

  // Termination.
  std::size_t generations = 50;
  std::size_t evaluations = 0;          // 0: no limit
  std::size_t stagnation = 0;           // generations without improving the best before stopping (0: never)
  bool restart_on_stagnation = false;   // instead of stopping, reseed all but the archive and the elites

  void validate() const {
    if (mu < 2) throw std::invalid_argument("a population needs at least two members");
    if (lambda < 1) throw std::invalid_argument("a generation needs at least one child");
    if (survival == Survival::comma && lambda < mu) throw std::invalid_argument("(μ,λ) needs λ ≥ μ");
    if (parents == Parents::tournament && tournament < 1) throw std::invalid_argument("a tournament needs at least one contestant");
    if (survival == Survival::alps && (layers < 1 || age_gap < 1)) throw std::invalid_argument("ALPS needs at least one layer and an age gap");
    if (survival == Survival::niches && (bins.empty() || bins.size() != lower.size() || bins.size() != upper.size()))
      throw std::invalid_argument("niches need bins, lower and upper for each descriptor");
    if (!(p_min >= 0) || p_min * 1 > 1) throw std::invalid_argument("p_min lies in [0, 1/operators]");
    if (!(crossover_rate >= 0 && crossover_rate <= 1) || !(generator_rate >= 0 && generator_rate <= 1))
      throw std::invalid_argument("the crossover and generator rates lie in [0, 1]");
  }
};

}  // namespace heuresis::evolution
