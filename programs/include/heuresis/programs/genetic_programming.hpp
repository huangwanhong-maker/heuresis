// Genetic programming: a generational search of the terms of one sort, on the ask–tell protocol.
//
// Generation 0 is a ramped half-and-half population of terms of the target sort (generation.hpp). Each later
// generation keeps the e best terms (elitism, never re-evaluated) and breeds the rest: each slot draws, from its own
// stream, one operation (subtree crossover, subtree, point, hoist or shrink mutation, or reproduction) with the
// given probabilities, and its parents by tournament (under lexicographic parsimony pressure if asked) or by
// ε-lexicase on the evaluations' descriptors (the errors case by case, regression.hpp) (Koza 1992, ch. 6–7; Luke and
// Panait 2002; La Cava, Spector and Danai 2016). Every term it offers is well-sorted, of the target sort, and within
// the limits.
//
// Being generational, its states do not depend on the order in which evaluations return (decision 0003 §2.2). It
// is migratory: emigrants are its best terms, and injected terms replace its worst.

#pragma once

#include <heuresis/programs/operators.hpp>
#include <heuresis/search/draws.hpp>
#include <heuresis/search/protocol.hpp>
#include <heuresis/selection/lexicase.hpp>
#include <heuresis/selection/parents.hpp>

namespace heuresis::programs {

struct GpOptions {
  Sort target = dimensionless();
  std::size_t population = 200;
  std::size_t min_depth = 2, max_depth = 6;  // of the initial ramp
  Limits limits;
  double crossover = 0.8, subtree = 0.1, point = 0.05, hoist = 0, shrink = 0;  // the rest is reproduction
  double point_rate = 0.1;
  std::size_t tournament = 7, elites = 1;
  bool parsimony = true;
  bool lexicase = false;
};

class GeneticProgramming {
 public:
  using Genotype = Tree;
  using Individual = search::Individual<Tree>;

  GeneticProgramming(const Signature& sig, GpOptions options, search::Draws draws = search::Draws{}) : sig_(&sig), o_(std::move(options)), draws_(draws) {
    if (o_.population == 0 || o_.elites >= o_.population) throw std::invalid_argument("heuresis::programs::GeneticProgramming: 0 ≤ elites < population");
    if (!sig.generable(o_.target, o_.max_depth)) throw std::invalid_argument("heuresis::programs::GeneticProgramming: no term of the target sort within the initial depth");
  }

  std::vector<search::Candidate<Tree>> ask(std::size_t limit) {
    if (!batch_.active()) breed();
    return batch_.take(limit);
  }

  void tell(std::span<const search::Told> told) {
    for (const auto& t : told)
      if (batch_.accept(t)) {
        std::vector<Individual> next = elite();
        for (auto& x : batch_.individuals()) next.push_back(std::move(x));
        population_ = std::move(next);
        batch_.close();
        generation_++;
      }
  }

  bool stopped() const { return false; }

  void inject(std::span<const Individual> in) {
    for (const auto& x : in) {
      if (population_.size() < o_.population) { population_.push_back(x); continue; }
      const auto order = ranked();
      population_[order.back()] = x;
    }
  }

  std::vector<Individual> emigrants(std::size_t k) const {
    std::vector<Individual> out;
    for (auto i : ranked()) {
      if (out.size() == k) break;
      out.push_back(population_[i]);
    }
    return out;
  }

  const std::vector<Individual>& population() const { return population_; }
  std::uint64_t generation() const { return generation_; }

  static constexpr std::uint32_t initialise = search::tag("programs.gp.initialise"), crossover = search::tag("programs.gp.crossover"),
                                 subtree = search::tag("programs.gp.subtree"), point = search::tag("programs.gp.point"), hoist = search::tag("programs.gp.hoist"),
                                 shrink = search::tag("programs.gp.shrink"), reproduce = search::tag("programs.gp.reproduce"), breeding = search::tag("programs.gp.breed");

 private:
  // The population's indices, best first.
  std::vector<std::size_t> ranked() const {
    std::vector<search::Evaluation> e;
    std::vector<std::size_t> sizes;
    for (const auto& x : population_) { e.push_back(x.evaluation); sizes.push_back(x.genotype.size()); }
    const selection::Less less = o_.parsimony ? lexicographic_parsimony(e, sizes) : selection::by_evaluation(e);
    return selection::order(population_.size(), less);
  }

  std::vector<Individual> elite() const {
    std::vector<Individual> out;
    if (population_.empty()) return out;
    for (auto i : ranked()) {
      if (out.size() == o_.elites) break;
      out.push_back(population_[i]);
    }
    return out;
  }

  void breed() {
    std::vector<search::Candidate<Tree>> c;
    if (population_.empty()) {
      for (std::size_t k = 0; k < o_.population; k++) {
        calx::random::Stream s = draws_(initialise, 0, k);
        c.push_back({search::make_ticket(draws_.island(), serial_++), ramped(*sig_, o_.target, k, o_.min_depth, o_.max_depth, s), search::Origin{initialise, {}}});
      }
      batch_.open(std::move(c));
      return;
    }
    std::vector<search::Evaluation> e;
    std::vector<std::size_t> sizes;
    std::vector<std::vector<double>> errors;
    for (const auto& x : population_) {
      e.push_back(x.evaluation);
      sizes.push_back(x.genotype.size());
      if (o_.lexicase) {
        auto d = x.evaluation.descriptors;
        if (x.evaluation.failed()) d.assign(population_.front().evaluation.descriptors.size(), INFINITY);
        errors.push_back(std::move(d));
      }
    }
    const selection::Less less = o_.parsimony ? lexicographic_parsimony(e, sizes) : selection::by_evaluation(e);
    std::vector<double> eps;
    if (o_.lexicase) {
      std::size_t width = 0;
      for (const auto& r : errors) width = std::max(width, r.size());
      for (auto& r : errors) r.resize(width, INFINITY);
      eps = selection::case_epsilons(errors);
    }
    auto parent = [&](calx::random::Stream& s) {
      return o_.lexicase ? selection::lexicase(errors, s, eps) : selection::tournament(population_.size(), o_.tournament, less, s);
    };
    const std::size_t lambda = o_.population - std::min(o_.elites, population_.size());
    for (std::size_t k = 0; k < lambda; k++) {
      calx::random::Stream s = draws_(breeding, generation_, k);
      const double u = s.uniform();
      const std::size_t a = parent(s);
      const Tree& pa = population_[a].genotype;
      std::uint32_t op;
      std::vector<search::Ticket> parents{population_[a].ticket};
      Tree child;
      if (u < o_.crossover) {
        const std::size_t b = parent(s);
        child = subtree_crossover(*sig_, pa, population_[b].genotype, o_.limits, s).first;
        parents.push_back(population_[b].ticket);
        op = crossover;
      } else if (u < o_.crossover + o_.subtree) {
        child = subtree_mutation(*sig_, pa, o_.limits, s);
        op = subtree;
      } else if (u < o_.crossover + o_.subtree + o_.point) {
        child = point_mutation(*sig_, pa, o_.point_rate, s);
        op = point;
      } else if (u < o_.crossover + o_.subtree + o_.point + o_.hoist) {
        child = hoist_mutation(*sig_, pa, s);
        op = hoist;
      } else if (u < o_.crossover + o_.subtree + o_.point + o_.hoist + o_.shrink) {
        child = shrink_mutation(*sig_, pa, s);
        op = shrink;
      } else {
        child = pa;
        op = reproduce;
      }
      c.push_back({search::make_ticket(draws_.island(), serial_++), std::move(child), search::Origin{op, std::move(parents)}});
    }
    batch_.open(std::move(c));
  }

  const Signature* sig_;
  GpOptions o_;
  search::Draws draws_;
  search::Batch<Tree> batch_;
  std::vector<Individual> population_;
  std::uint64_t generation_ = 0, serial_ = 0;
};

}  // namespace heuresis::programs
