// Choosing survivors: which of parents and offspring form the next population.
//
// - Comma selection (μ, λ): the best μ of the λ offspring; parents never survive, so the search can leave a
//   local optimum and a step size can adapt (Schwefel 1981, Numerical Optimization of Computer Models).
// - Plus selection (μ + λ): the best μ of parents and offspring together; elitist, the best never lost.
// - Generational replacement with elitism: the offspring replace the parents except the e best parents, who
//   take the places of the e worst offspring (De Jong 1975).
// - Steady-state replacement of the worst (Whitley 1989, "The GENITOR algorithm and selection pressure", ICGA 3,
//   116–121) or of a tournament's loser.
// Each returns the indices of the survivors in the joined list (parents first, then offspring), best first;
// ties fall to the earlier index, so offspring that tie their parents do not displace them under plus selection
// unless the caller orders offspring first (which lets a plateau be crossed; Doerr and Neumann 2020, §1.4).

#pragma once

#include <heuresis/selection/parents.hpp>

namespace heuresis::selection {

inline std::vector<std::size_t> comma(std::span<const Evaluation> offspring, std::size_t mu) {
  if (mu > offspring.size()) throw std::invalid_argument("heuresis::selection::comma: μ exceeds λ");
  return truncation(offspring, mu);
}

// Indices into the joined list parents ++ offspring.
inline std::vector<std::size_t> plus(std::span<const Evaluation> parents, std::span<const Evaluation> offspring, std::size_t mu) {
  std::vector<Evaluation> all(parents.begin(), parents.end());
  all.insert(all.end(), offspring.begin(), offspring.end());
  return truncation(all, mu);
}

// The next generation as indices into parents ++ offspring: every offspring but the e worst, and the e best parents.
inline std::vector<std::size_t> generational(std::span<const Evaluation> parents, std::span<const Evaluation> offspring, std::size_t elites) {
  elites = std::min({elites, parents.size(), offspring.size()});
  std::vector<std::size_t> keep;
  const auto po = truncation(parents, elites);
  for (auto i : po) keep.push_back(i);
  const auto oo = order(offspring.size(), by_evaluation(offspring));
  for (std::size_t k = 0; k + elites < oo.size(); k++) keep.push_back(parents.size() + oo[k]);
  std::vector<Evaluation> all(parents.begin(), parents.end());
  all.insert(all.end(), offspring.begin(), offspring.end());
  std::stable_sort(keep.begin(), keep.end(), [&](std::size_t a, std::size_t b) { return heuresis::search::better(all[a], all[b]); });
  return keep;
}

// The index of the worst individual (ties to the later index), the one a steady-state algorithm replaces.
inline std::size_t worst(std::span<const Evaluation> e) {
  if (e.empty()) throw std::invalid_argument("heuresis::selection::worst: an empty population");
  std::size_t w = 0;
  for (std::size_t i = 1; i < e.size(); i++)
    if (!heuresis::search::better(e[i], e[w])) w = i;
  return w;
}

// The loser of a tournament of k drawn with replacement (ties to the later index).
inline std::size_t reverse_tournament(std::span<const Evaluation> e, std::size_t k, calx::random::Stream& s) {
  std::size_t loser = s.below(e.size());
  for (std::size_t t = 1; t < k; t++) {
    const std::size_t c = s.below(e.size());
    if (heuresis::search::better(e[loser], e[c]) || (!heuresis::search::better(e[c], e[loser]) && c > loser)) loser = c;
  }
  return loser;
}

}  // namespace heuresis::selection
