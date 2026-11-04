// The generational search by a distribution: sample a generation from P_θ, rank it, move θ by a rule.
//
// One algorithm serves every family, rule and sampler (decision 0003 §2.8): UMDA, PBIL and the compact GA on bits,
// UMDA on categorical vectors, the cross-entropy method and the pure rank-μ evolution strategy on ℝⁿ, Plackett–Luce
// on permutations, and any IGO algorithm. The k-th point of generation g is drawn from the stream keyed by (g, k), by
// the sampler given (the reference law by default); the rule reads the generation in ticket order, ranked best
// first by Deb's order with ties to the earlier ticket. A stopping predicate on θ may end the search (a Bernoulli law
// whose frequencies have all reached the borders, a Gaussian whose spread is below a tolerance).

#pragma once

#include <heuresis/distributions/family.hpp>
#include <heuresis/search/draws.hpp>
#include <heuresis/search/protocol.hpp>

#include <functional>
#include <utility>
#include <vector>

namespace heuresis::distributions {

template <Family F, class Rule, class S = Reference>
  requires Sampler<S, F>
class DistributionSearch {
 public:
  using Genotype = typename F::Point;
  using Parameters = typename F::Parameters;

  DistributionSearch(F family, Parameters theta, std::size_t lambda, Rule rule, heuresis::search::Draws draws = heuresis::search::Draws{}, S sampler = S{})
      : family_(std::move(family)), theta_(std::move(theta)), lambda_(lambda), rule_(std::move(rule)), draws_(draws), sampler_(std::move(sampler)) {
    if (lambda_ == 0) throw std::invalid_argument("heuresis::distributions::DistributionSearch: λ ≥ 1");
  }

  std::vector<heuresis::search::Candidate<Genotype>> ask(std::size_t limit) {
    if (!batch_.active()) {
      std::vector<heuresis::search::Candidate<Genotype>> g;
      g.reserve(lambda_);
      for (std::size_t k = 0; k < lambda_; k++) {
        calx::random::Stream s = draws_(sample_tag, generation_, k);
        g.push_back({heuresis::search::make_ticket(draws_.island(), serial_++), sampler_(family_, theta_, s), heuresis::search::Origin{sample_tag, {}}});
      }
      batch_.open(std::move(g));
    }
    return batch_.take(limit);
  }

  void tell(std::span<const heuresis::search::Told> told) {
    for (const auto& t : told) {
      if (!batch_.accept(t)) continue;
      const auto order = heuresis::search::ranking(batch_.evaluations());
      std::vector<Genotype> ranked;
      std::vector<Evaluation> evaluations;
      ranked.reserve(order.size());
      for (auto i : order) {
        ranked.push_back(batch_.candidate(i).genotype);
        evaluations.push_back(batch_.evaluation(i));
      }
      theta_ = rule_(family_, theta_, std::span<const Genotype>(ranked), std::span<const Evaluation>(evaluations));
      batch_.close();
      generation_++;
    }
  }

  bool stopped() const { return stop_ && !batch_.active() && stop_(theta_); }
  void stop_when(std::function<bool(const Parameters&)> stop) { stop_ = std::move(stop); }

  const Parameters& parameters() const { return theta_; }
  const F& family() const { return family_; }
  std::uint64_t generation() const { return generation_; }

  static constexpr std::uint32_t sample_tag = heuresis::search::tag("distributions.search.sample");

 private:
  F family_;
  Parameters theta_;
  std::size_t lambda_;
  Rule rule_;
  heuresis::search::Draws draws_;
  S sampler_;
  heuresis::search::Batch<Genotype> batch_;
  std::function<bool(const Parameters&)> stop_;
  std::uint64_t generation_ = 0, serial_ = 0;
};

// A rule from any IGO step: weights by rank (tied points sharing theirs), then the step.
template <class Step>
struct IgoRule {
  std::vector<double> weights;
  double rate;
  Step step;
  template <Family F>
  typename F::Parameters operator()(const F& f, const typename F::Parameters& theta, std::span<const typename F::Point> ranked, std::span<const Evaluation> evaluations) const {
    const std::vector<double> w = tied_weights(evaluations, weights);
    return step(f, theta, ranked, std::span<const double>(w), rate);
  }
};

// The IGO step in expectation parameters, and the generic step by the Fisher information, as rule steps.
struct ExpectationStep {
  template <ExponentialFamily F>
  typename F::Parameters operator()(const F& f, const typename F::Parameters& theta, std::span<const typename F::Point> points, std::span<const double> w, double dt) const { return expectation_step(f, theta, points, w, dt); }
};
struct FisherStep {
  template <Riemannian F>
  typename F::Parameters operator()(const F& f, const typename F::Parameters& theta, std::span<const typename F::Point> points, std::span<const double> w, double dt) const { return igo_step(f, theta, points, w, dt); }
};

}  // namespace heuresis::distributions
