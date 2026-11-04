// The (1+(λ,λ)) genetic algorithm: crossover used as a repair of large mutations.
//
// Each iteration has two phases (Doerr, Doerr and Ebel 2015, "From black-box complexity to designing new
// genetic algorithms", Theor. Comput. Sci. 567, 87–104):
//   mutation:  ℓ ~ Bin(n, λ/n), conditioned on ℓ ≥ 1; λ mutants of x, each flipping exactly ℓ bits; x′ the best;
//   crossover: λ biased crossovers of x and x′, each taking each of the ℓ differing bits from x′ with
//              probability c = 1/λ; y the best of them and x′;
//   selection: x ← y when y is not worse than x.
// The mutants explore far (ℓ ≈ λ bits) and the crossover keeps from them what helped, so on OneMax the expected
// time with a fitness-dependent λ, or with λ adjusted by the one-fifth rule below, is linear in n, against the
// Θ(n log n) of every mutation-only elitist algorithm (Doerr and Doerr 2018, "Optimal static and
// self-adjusting parameter tuning for the (1+(λ,λ)) genetic algorithm", Algorithmica 80, 1658–1709).
//
// The self-adjusting λ: after an iteration that improved x, λ ← max(λ/F, 1); otherwise λ ← min(λ F^{1/4}, n),
// F = 3/2. Each phase offers round(λ) candidates.
//
// "Practice-aware" evaluation (Carvalho Pinto and Doerr 2018, "Towards a more practice-aware runtime analysis of
// evolutionary algorithms", arXiv:1812.00493): ℓ ≥ 1 is enforced, and a crossover offspring equal to x or to x′
// is not offered for evaluation, since its value is known; an iteration whose offspring are all such copies
// spends nothing on its crossover phase.

#pragma once

#include <heuresis/genetic/kernels.hpp>
#include <heuresis/search/search.hpp>

#include <cmath>
#include <optional>
#include <span>
#include <vector>

namespace heuresis::genetic {

struct LambdaLambdaOptions {
  double lambda = 1;  // the initial (or fixed) λ
  bool self_adjusting = true;
  double F = 1.5;
};

class OnePlusLambdaLambda {
 public:
  using Genotype = Bits;
  static constexpr std::uint32_t sample_tag = heuresis::search::tag("genetic.lambda_lambda.sample");
  static constexpr std::uint32_t mutate_tag = heuresis::search::tag("genetic.lambda_lambda.mutate");
  static constexpr std::uint32_t cross_tag = heuresis::search::tag("genetic.lambda_lambda.cross");

  using Options = LambdaLambdaOptions;

  OnePlusLambdaLambda(std::size_t n, Options options = Options{}, heuresis::search::Draws draws = heuresis::search::Draws{}) : n_(n), options_(options), lambda_(options.lambda), draws_(draws) {
    if (n_ == 0) throw std::invalid_argument("heuresis::genetic::OnePlusLambdaLambda: an empty string");
  }

  std::vector<heuresis::search::Candidate<Bits>> ask(std::size_t limit) {
    if (!batch_.active()) open();
    return batch_.take(limit);
  }

  void tell(std::span<const heuresis::search::Told> told) {
    for (const auto& t : told) {
      if (!batch_.accept(t)) continue;
      auto xs = batch_.individuals();
      batch_.close();
      if (phase_ == Phase::initial) {
        parent_ = std::move(xs[0]);
        phase_ = Phase::mutation;
      } else if (phase_ == Phase::mutation) {
        const auto order = heuresis::search::ranking(batch_evaluations(xs));
        mutant_ = std::move(xs[order[0]]);
        phase_ = Phase::crossover;
      } else {
        finish(xs);
      }
    }
  }

  bool stopped() const { return false; }
  const std::optional<heuresis::search::Individual<Bits>>& parent() const { return parent_; }
  double lambda() const { return lambda_; }
  std::uint64_t iterations() const { return iteration_; }

  std::vector<heuresis::search::Individual<Bits>> emigrants(std::size_t k) const {
    if (!parent_ || k == 0) return {};
    return {*parent_};
  }
  void inject(std::span<const heuresis::search::Individual<Bits>> in) {
    for (const auto& x : in)
      if (!parent_ || !heuresis::search::better(parent_->evaluation, x.evaluation)) parent_ = x;
  }

 private:
  enum class Phase { initial, mutation, crossover };

  static std::vector<heuresis::search::Evaluation> batch_evaluations(const std::vector<heuresis::search::Individual<Bits>>& xs) {
    std::vector<heuresis::search::Evaluation> e;
    for (const auto& x : xs) e.push_back(x.evaluation);
    return e;
  }

  std::size_t count() const { return static_cast<std::size_t>(std::max(1.0, std::round(lambda_))); }

  // Opens the next phase's generation; a crossover phase of copies only is completed at once.
  void open() {
    using heuresis::search::make_ticket;
    std::vector<heuresis::search::Candidate<Bits>> g;
    if (phase_ == Phase::initial) {
      calx::random::Stream s = draws_(sample_tag, 0, 0);
      g.push_back({make_ticket(draws_.island(), serial_++), heuresis::variation::random_bits(n_, s), {sample_tag, {}}});
      batch_.open(std::move(g));
      return;
    }
    for (;;) {
      if (phase_ == Phase::mutation) {
        calx::random::Stream s = draws_(mutate_tag, iteration_, count());
        const double p = std::min(1.0, lambda_ / static_cast<double>(n_));
        std::int64_t ell = 0;
        while (ell == 0) ell = s.binomial(static_cast<std::int64_t>(n_), p);
        for (std::size_t i = 0; i < count(); i++) {
          calx::random::Stream r = draws_(mutate_tag, iteration_, i);
          Bits y = parent_->genotype;
          heuresis::variation::flip_exactly(y, static_cast<std::size_t>(ell), r);
          g.push_back({make_ticket(draws_.island(), serial_++), std::move(y), {mutate_tag, {parent_->ticket}}});
        }
        batch_.open(std::move(g));
        return;
      }
      // Crossover: bits where x′ differs from x are taken from x′ with probability 1/λ.
      std::vector<std::size_t> differ;
      for (std::size_t i = 0; i < n_; i++)
        if (mutant_->genotype[i] != parent_->genotype[i]) differ.push_back(i);
      const double c = 1 / lambda_;
      for (std::size_t i = 0; i < count(); i++) {
        calx::random::Stream r = draws_(cross_tag, iteration_, i);
        Bits y = parent_->genotype;
        std::size_t taken = 0;
        for (auto j : differ)
          if (r.uniform() < c) { y[j] = mutant_->genotype[j]; taken++; }
        if (taken == 0 || taken == differ.size()) continue;  // a copy of x or of x′: its value is known
        g.push_back({make_ticket(draws_.island(), serial_++), std::move(y), {cross_tag, {parent_->ticket, mutant_->ticket}}});
      }
      if (!g.empty()) {
        batch_.open(std::move(g));
        return;
      }
      finish({});
    }
  }

  // The end of an iteration: y the best of the crossover offspring and x′; x ← y when not worse; λ adjusted.
  void finish(const std::vector<heuresis::search::Individual<Bits>>& offspring) {
    heuresis::search::Individual<Bits> y = *mutant_;
    for (const auto& o : offspring)
      if (heuresis::search::better(o.evaluation, y.evaluation)) y = o;
    const bool improved = heuresis::search::better(y.evaluation, parent_->evaluation);
    if (!heuresis::search::better(parent_->evaluation, y.evaluation)) parent_ = std::move(y);
    if (options_.self_adjusting) lambda_ = improved ? std::max(lambda_ / options_.F, 1.0) : std::min(lambda_ * std::pow(options_.F, 0.25), static_cast<double>(n_));
    mutant_.reset();
    iteration_++;
    phase_ = Phase::mutation;
  }

  std::size_t n_;
  Options options_;
  double lambda_;
  heuresis::search::Draws draws_;
  heuresis::search::Batch<Bits> batch_;
  Phase phase_ = Phase::initial;
  std::optional<heuresis::search::Individual<Bits>> parent_, mutant_;
  std::uint64_t iteration_ = 0, serial_ = 0;
};

}  // namespace heuresis::genetic
