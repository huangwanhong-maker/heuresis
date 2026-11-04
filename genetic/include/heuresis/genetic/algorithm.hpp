// The genetic algorithm of composable kernels, generational and steady-state.
//
// A population of μ individuals breeds λ offspring a generation: two parents by tournaments, crossed with
// probability p_c (otherwise copied), each child mutated (Holland 1975, Adaptation in Natural and Artificial
// Systems; Goldberg 1989, Genetic Algorithms in Search, Optimization, and Machine Learning; De Jong 2006,
// Evolutionary Computation: A Unified Approach, ch. 4). The survivors are chosen by
//   generational replacement with e elites (the default, e = 1),
//   plus selection, the best μ of parents and offspring together, or
//   comma selection, the best μ of the offspring (λ ≥ μ).
// The algorithm is generational in the protocol's sense: it breeds a generation from streams keyed by
// (generation, pair), and updates once all of it is told, in ticket order.
//
// The steady-state form (Whitley 1989, "The GENITOR algorithm and selection pressure", ICGA 3, 116–121; Syswerda
// 1991, "A study of reproduction in generational and steady-state genetic algorithms", FOGA 1, 94–101) breeds
// offspring whenever asked, at most `breadth` (default 1) from one state of the population, and each offspring
// told replaces the worst when it is not worse. It is asynchronous: it keeps every worker busy, and its run depends on the order of arrival
// (search/workers.hpp). Its first μ candidates are the initial sample.
//
// Both are Migratory: emigrants are the best k, immigrants replace the worst when not worse.

#pragma once

#include <heuresis/genetic/kernels.hpp>
#include <heuresis/search/search.hpp>
#include <heuresis/selection/selection.hpp>

#include <span>
#include <vector>

namespace heuresis::genetic {

template <class G>
struct Operators {
  Sampler<G> sample;
  Mutation<G> mutate;      // empty: no mutation
  Crossover<G> crossover;  // empty: no crossover
};

enum class Survival : std::uint8_t { generational, plus, comma };

struct GaOptions {
  std::size_t mu = 50;
  std::size_t lambda = 0;  // 0: λ = μ
  double crossover_rate = 0.9;
  std::size_t tournament = 2;
  Survival survival = Survival::generational;
  std::size_t elites = 1;  // for generational survival
  std::size_t breadth = 1;  // steady state: the most offspring bred from one state of the population per ask
};

namespace detail {
template <class G>
std::vector<heuresis::search::Evaluation> evaluations_of(const std::vector<heuresis::search::Individual<G>>& xs) {
  std::vector<heuresis::search::Evaluation> e;
  e.reserve(xs.size());
  for (const auto& x : xs) e.push_back(x.evaluation);
  return e;
}

// Two children of the population by tournaments, crossover and mutation, from one stream.
template <class G>
std::pair<heuresis::search::Candidate<G>, heuresis::search::Candidate<G>> breed(const std::vector<heuresis::search::Individual<G>>& pop, const std::vector<heuresis::search::Evaluation>& e, const Operators<G>& op, const GaOptions& o,
                                                                                 std::uint32_t operation, calx::random::Stream& s) {
  const auto& a = pop[heuresis::selection::tournament(e, o.tournament, s)];
  const auto& b = pop[heuresis::selection::tournament(e, o.tournament, s)];
  std::pair<G, G> c{a.genotype, b.genotype};
  if (op.crossover && s.uniform() < o.crossover_rate) c = op.crossover(a.genotype, b.genotype, s);
  if (op.mutate) {
    c.first = op.mutate(c.first, s);
    c.second = op.mutate(c.second, s);
  }
  heuresis::search::Origin origin{operation, {a.ticket, b.ticket}};
  return {{0, std::move(c.first), origin}, {0, std::move(c.second), origin}};
}

template <class G>
void replace_worst(std::vector<heuresis::search::Individual<G>>& pop, const heuresis::search::Individual<G>& x) {
  const std::size_t w = heuresis::selection::worst(evaluations_of(pop));
  if (!heuresis::search::better(pop[w].evaluation, x.evaluation)) pop[w] = x;
}

template <class G>
std::vector<heuresis::search::Individual<G>> best_of(const std::vector<heuresis::search::Individual<G>>& pop, std::size_t k) {
  std::vector<heuresis::search::Individual<G>> out;
  for (auto i : heuresis::selection::truncation(evaluations_of(pop), k)) out.push_back(pop[i]);
  return out;
}
}  // namespace detail

template <class G>
class GeneticAlgorithm {
 public:
  using Genotype = G;
  static constexpr std::uint32_t sample_tag = heuresis::search::tag("genetic.ga.sample");
  static constexpr std::uint32_t breed_tag = heuresis::search::tag("genetic.ga.breed");

  GeneticAlgorithm(Operators<G> operators, GaOptions options = GaOptions{}, heuresis::search::Draws draws = heuresis::search::Draws{}) : op_(std::move(operators)), o_(options), draws_(draws) {
    if (o_.lambda == 0) o_.lambda = o_.mu;
    if (o_.mu == 0 || o_.tournament == 0) throw std::invalid_argument("heuresis::genetic::GeneticAlgorithm: μ and the tournament must be at least 1");
    if (o_.survival == Survival::comma && o_.lambda < o_.mu) throw std::invalid_argument("heuresis::genetic::GeneticAlgorithm: comma selection needs λ ≥ μ");
    if (!op_.sample) throw std::invalid_argument("heuresis::genetic::GeneticAlgorithm: no sampler");
  }

  std::vector<heuresis::search::Candidate<G>> ask(std::size_t limit) {
    using heuresis::search::make_ticket;
    if (!batch_.active()) {
      std::vector<heuresis::search::Candidate<G>> g;
      if (population_.empty()) {
        for (std::size_t i = 0; i < o_.mu; i++) {
          calx::random::Stream s = draws_(sample_tag, 0, i);
          g.push_back({make_ticket(draws_.island(), serial_++), op_.sample(s), {sample_tag, {}}});
        }
      } else {
        const auto e = detail::evaluations_of(population_);
        for (std::size_t pair = 0; g.size() < o_.lambda; pair++) {
          calx::random::Stream s = draws_(breed_tag, generation_, pair);
          auto [c1, c2] = detail::breed(population_, e, op_, o_, breed_tag, s);
          c1.ticket = make_ticket(draws_.island(), serial_++);
          g.push_back(std::move(c1));
          if (g.size() < o_.lambda) {
            c2.ticket = make_ticket(draws_.island(), serial_++);
            g.push_back(std::move(c2));
          }
        }
      }
      batch_.open(std::move(g));
    }
    return batch_.take(limit);
  }

  void tell(std::span<const heuresis::search::Told> told) {
    for (const auto& t : told) {
      if (!batch_.accept(t)) continue;
      auto offspring = batch_.individuals();
      batch_.close();
      if (population_.empty()) population_ = std::move(offspring);
      else survive(std::move(offspring));
      generation_++;
    }
  }

  bool stopped() const { return false; }
  const std::vector<heuresis::search::Individual<G>>& population() const { return population_; }
  std::uint64_t generation() const { return generation_; }

  std::vector<heuresis::search::Individual<G>> emigrants(std::size_t k) const { return detail::best_of(population_, k); }
  void inject(std::span<const heuresis::search::Individual<G>> in) {
    for (const auto& x : in) {
      if (population_.size() < o_.mu) population_.push_back(x);
      else detail::replace_worst(population_, x);
    }
  }

 private:
  void survive(std::vector<heuresis::search::Individual<G>> offspring) {
    const auto pe = detail::evaluations_of(population_), oe = detail::evaluations_of(offspring);
    std::vector<std::size_t> keep;
    switch (o_.survival) {
      case Survival::generational: keep = heuresis::selection::generational(pe, oe, o_.elites); break;
      case Survival::plus: keep = heuresis::selection::plus(pe, oe, o_.mu); break;
      case Survival::comma:
        keep = heuresis::selection::comma(oe, o_.mu);
        for (auto& k : keep) k += population_.size();
        break;
    }
    std::vector<heuresis::search::Individual<G>> next;
    next.reserve(keep.size());
    for (auto k : keep) next.push_back(k < population_.size() ? population_[k] : offspring[k - population_.size()]);
    if (next.size() > o_.mu) next.resize(o_.mu);
    population_ = std::move(next);
  }

  Operators<G> op_;
  GaOptions o_;
  heuresis::search::Draws draws_;
  heuresis::search::Batch<G> batch_;
  std::vector<heuresis::search::Individual<G>> population_;
  std::uint64_t generation_ = 0, serial_ = 0;
};

template <class G>
class SteadyStateGA {
 public:
  using Genotype = G;
  static constexpr std::uint32_t sample_tag = heuresis::search::tag("genetic.steady.sample");
  static constexpr std::uint32_t breed_tag = heuresis::search::tag("genetic.steady.breed");

  SteadyStateGA(Operators<G> operators, GaOptions options = GaOptions{}, heuresis::search::Draws draws = heuresis::search::Draws{}) : op_(std::move(operators)), o_(options), draws_(draws) {
    if (o_.mu == 0 || o_.tournament == 0 || o_.breadth == 0) throw std::invalid_argument("heuresis::genetic::SteadyStateGA: μ, the tournament and the breadth must be at least 1");
    if (!op_.sample) throw std::invalid_argument("heuresis::genetic::SteadyStateGA: no sampler");
  }

  // Offers the initial samples together until μ have been asked; then breeds at most `breadth` offspring from the
  // population as it stands (protocol.hpp).
  std::vector<heuresis::search::Candidate<G>> ask(std::size_t limit) {
    using heuresis::search::make_ticket;
    std::vector<heuresis::search::Candidate<G>> out;
    const bool sampling = sampled_ < o_.mu;
    if (!sampling) limit = std::min(limit, o_.breadth);
    for (std::size_t i = 0; i < limit; i++) {
      heuresis::search::Candidate<G> c;
      if (sampling) {
        if (sampled_ == o_.mu) break;
        calx::random::Stream s = draws_(sample_tag, 0, serial_);
        c = {0, op_.sample(s), {sample_tag, {}}};
        sampled_++;
      } else {
        if (population_.empty()) break;  // the initial sample is still being evaluated
        calx::random::Stream s = draws_(breed_tag, 0, serial_);
        c = detail::breed(population_, detail::evaluations_of(population_), op_, o_, breed_tag, s).first;
      }
      c.ticket = make_ticket(draws_.island(), serial_++);
      pending_.add(c);
      out.push_back(std::move(c));
    }
    return out;
  }

  void tell(std::span<const heuresis::search::Told> told) {
    for (const auto& t : told) {
      auto [c, note] = pending_.take(t.ticket);
      heuresis::search::Individual<G> x{c.ticket, std::move(c.genotype), t.evaluation, std::move(c.origin)};
      if (population_.size() < o_.mu) population_.push_back(std::move(x));
      else detail::replace_worst(population_, x);
    }
  }

  bool stopped() const { return false; }
  const std::vector<heuresis::search::Individual<G>>& population() const { return population_; }

  std::vector<heuresis::search::Individual<G>> emigrants(std::size_t k) const { return detail::best_of(population_, k); }
  void inject(std::span<const heuresis::search::Individual<G>> in) {
    for (const auto& x : in) {
      if (population_.size() < o_.mu) population_.push_back(x);
      else detail::replace_worst(population_, x);
    }
  }

 private:
  Operators<G> op_;
  GaOptions o_;
  heuresis::search::Draws draws_;
  heuresis::search::Pending<G> pending_;
  std::vector<heuresis::search::Individual<G>> population_;
  std::uint64_t serial_ = 0;
  std::size_t sampled_ = 0;
};

}  // namespace heuresis::genetic
