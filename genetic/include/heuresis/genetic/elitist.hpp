// Elitist evolutionary algorithms of the theory: the (1+λ) EA, the (1+1) EA and randomised local search
// among them, and the (μ+1) EA.
//
// The (1+λ) EA keeps one parent x; each generation it makes λ offspring by a mutation and replaces x by the best
// of them when that one is not worse (minimisation; ties among offspring to the earliest ticket). Accepting
// offspring of equal value lets it drift across plateaus. With λ = 1 and standard bit mutation at rate 1/n it is
// the (1+1) EA, whose expected optimisation time on OneMax is e n ln n − O(n) and on LeadingOnes
// ((1 − p)^{1−n} − (1 − p))/(2p²) at rate p, ≈ (e − 1)n²/2 at p = 1/n (Droste, Jansen and Wegener 2002,
// "On the analysis of the (1+1) evolutionary algorithm", Theor. Comput. Sci. 276, 51–81; Böttcher, Doerr and
// Neumann 2010, "Optimal fixed and adaptive mutation rates for the LeadingOnes problem", PPSN XI, 1–10; Hwang,
// Panholzer, Rolin, Tsai and Chen 2018, "Probabilistic analysis of the (1+1)-evolutionary algorithm", Evol.
// Comput. 26, 299–345, for the OneMax expansion).
// With one uniformly chosen bit flipped it is randomised local search.
//
// The (μ+1) EA keeps μ individuals; each generation it mutates one parent drawn uniformly and the offspring
// replaces the worst individual when it is not worse than it (Witt 2006, "Runtime analysis of the (μ+1) EA on
// simple pseudo-Boolean functions", Evol. Comput. 14, 65–86). An optional crossover of two uniform parents
// before the mutation, with probability p_c, makes it the (μ+1) GA (Dang, Friedrich, Kötzing, Krejca, Lehre,
// Oliveto, Sudholt and Sutton 2018, "Escaping local optima using crossover with emergent diversity", IEEE
// Trans. Evol. Comput. 22, 484–497).
//
// Both are generational (protocol.hpp): generation 0 is the initial sample, and each update reads the
// generation's evaluations in ticket order. Both are Migratory: they send their best and receive individuals
// that replace their worst when not worse.

#pragma once

#include <heuresis/genetic/kernels.hpp>
#include <heuresis/search/search.hpp>
#include <heuresis/selection/selection.hpp>

#include <optional>
#include <span>
#include <vector>

namespace heuresis::genetic {

using heuresis::search::Batch;
using heuresis::search::Candidate;
using heuresis::search::Draws;
using heuresis::search::Individual;
using heuresis::search::Origin;
using heuresis::search::Told;
using heuresis::search::better;
using heuresis::search::make_ticket;
using heuresis::search::tag;

template <class G>
class OnePlusLambda {
 public:
  using Genotype = G;
  static constexpr std::uint32_t sample_tag = tag("genetic.one_plus_lambda.sample");
  static constexpr std::uint32_t mutate_tag = tag("genetic.one_plus_lambda.mutate");

  OnePlusLambda(Sampler<G> sample, Mutation<G> mutate, std::size_t lambda = 1, Draws draws = Draws{}) : sample_(std::move(sample)), mutate_(std::move(mutate)), lambda_(lambda), draws_(draws) {
    if (lambda_ == 0) throw std::invalid_argument("heuresis::genetic::OnePlusLambda: λ must be at least 1");
  }

  std::vector<Candidate<G>> ask(std::size_t limit) {
    if (!batch_.active()) {
      std::vector<Candidate<G>> g;
      if (!parent_) {
        calx::random::Stream s = draws_(sample_tag, 0, 0);
        g.push_back({make_ticket(draws_.island(), serial_++), sample_(s), Origin{sample_tag, {}}});
      } else {
        for (std::size_t i = 0; i < lambda_; i++) {
          calx::random::Stream s = draws_(mutate_tag, generation_, i);
          g.push_back({make_ticket(draws_.island(), serial_++), mutate_(parent_->genotype, s), Origin{mutate_tag, {parent_->ticket}}});
        }
      }
      batch_.open(std::move(g));
    }
    return batch_.take(limit);
  }

  void tell(std::span<const Told> told) {
    for (const auto& t : told) {
      if (!batch_.accept(t)) continue;
      const auto order = heuresis::search::ranking(batch_.evaluations());
      auto best = batch_.individuals()[order[0]];
      if (!parent_ || !better(parent_->evaluation, best.evaluation)) parent_ = std::move(best);
      batch_.close();
      generation_++;
    }
  }

  bool stopped() const { return false; }
  const std::optional<Individual<G>>& parent() const { return parent_; }
  std::uint64_t generation() const { return generation_; }

  std::vector<Individual<G>> emigrants(std::size_t k) const {
    if (!parent_ || k == 0) return {};
    return {*parent_};
  }
  void inject(std::span<const Individual<G>> in) {
    for (const auto& x : in)
      if (!parent_ || !better(parent_->evaluation, x.evaluation)) parent_ = x;
  }

 private:
  Sampler<G> sample_;
  Mutation<G> mutate_;
  std::size_t lambda_;
  Draws draws_;
  Batch<G> batch_;
  std::optional<Individual<G>> parent_;
  std::uint64_t generation_ = 0, serial_ = 0;
};

// The (1+1) EA with standard bit mutation at rate p (1/n when p < 0) on strings of length n.
inline OnePlusLambda<Bits> one_plus_one_ea(std::size_t n, Draws draws = Draws{}, double p = -1) { return OnePlusLambda<Bits>(random_bits(n), bit_flip(p), 1, draws); }

// Randomised local search: one uniformly chosen bit flipped, accepted when not worse.
inline OnePlusLambda<Bits> randomised_local_search(std::size_t n, Draws draws = Draws{}) { return OnePlusLambda<Bits>(random_bits(n), flip_one(), 1, draws); }

template <class G>
class MuPlusOne {
 public:
  using Genotype = G;
  static constexpr std::uint32_t sample_tag = tag("genetic.mu_plus_one.sample");
  static constexpr std::uint32_t breed_tag = tag("genetic.mu_plus_one.breed");

  MuPlusOne(std::size_t mu, Sampler<G> sample, Mutation<G> mutate, Draws draws = Draws{}, Crossover<G> crossover = {}, double crossover_rate = 0)
      : mu_(mu), sample_(std::move(sample)), mutate_(std::move(mutate)), crossover_(std::move(crossover)), crossover_rate_(crossover_rate), draws_(draws) {
    if (mu_ == 0) throw std::invalid_argument("heuresis::genetic::MuPlusOne: μ must be at least 1");
  }

  std::vector<Candidate<G>> ask(std::size_t limit) {
    if (!batch_.active()) {
      std::vector<Candidate<G>> g;
      if (population_.empty()) {
        for (std::size_t i = 0; i < mu_; i++) {
          calx::random::Stream s = draws_(sample_tag, 0, i);
          g.push_back({make_ticket(draws_.island(), serial_++), sample_(s), Origin{sample_tag, {}}});
        }
      } else {
        calx::random::Stream s = draws_(breed_tag, generation_, 0);
        const auto& a = population_[s.below(population_.size())];
        G child = a.genotype;
        Origin origin{breed_tag, {a.ticket}};
        if (crossover_ && s.uniform() < crossover_rate_) {
          const auto& b = population_[s.below(population_.size())];
          child = crossover_(a.genotype, b.genotype, s).first;
          origin.parents.push_back(b.ticket);
        }
        g.push_back({make_ticket(draws_.island(), serial_++), mutate_ ? mutate_(child, s) : child, std::move(origin)});
      }
      batch_.open(std::move(g));
    }
    return batch_.take(limit);
  }

  void tell(std::span<const Told> told) {
    for (const auto& t : told) {
      if (!batch_.accept(t)) continue;
      auto xs = batch_.individuals();
      if (population_.empty()) population_ = std::move(xs);
      else replace_worst(std::move(xs[0]));
      batch_.close();
      generation_++;
    }
  }

  bool stopped() const { return false; }
  const std::vector<Individual<G>>& population() const { return population_; }

  std::vector<Individual<G>> emigrants(std::size_t k) const {
    std::vector<Individual<G>> out;
    for (auto i : heuresis::selection::truncation(evaluations(), k)) out.push_back(population_[i]);
    return out;
  }
  void inject(std::span<const Individual<G>> in) {
    for (const auto& x : in) {
      if (population_.size() < mu_) population_.push_back(x);
      else replace_worst(x);
    }
  }

 private:
  std::vector<heuresis::search::Evaluation> evaluations() const {
    std::vector<heuresis::search::Evaluation> e;
    for (const auto& x : population_) e.push_back(x.evaluation);
    return e;
  }
  void replace_worst(Individual<G> x) {
    const std::size_t w = heuresis::selection::worst(evaluations());
    if (!better(population_[w].evaluation, x.evaluation)) population_[w] = std::move(x);
  }

  std::size_t mu_;
  Sampler<G> sample_;
  Mutation<G> mutate_;
  Crossover<G> crossover_;
  double crossover_rate_;
  Draws draws_;
  Batch<G> batch_;
  std::vector<Individual<G>> population_;
  std::uint64_t generation_ = 0, serial_ = 0;
};

}  // namespace heuresis::genetic
