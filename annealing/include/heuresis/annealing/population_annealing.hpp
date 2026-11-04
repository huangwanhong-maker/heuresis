// Population annealing, and the free energy it estimates.
//
// A population of R states is drawn from the Boltzmann law at β₀ (for β₀ = 0 on a finite space, the uniform law)
// and carried through inverse temperatures β₀ < β₁ < … < β_K. At each step from β_k to β_{k+1}:
//
//   reweight   each state by wᵢ = exp(−(β_{k+1} − β_k) E(xᵢ)), whose mean Q_k estimates Z(β_{k+1})/Z(β_k) without bias;
//   resample   R states with probabilities proportional to the weights (here by stochastic universal sampling,
//              selection/parents.hpp, which is unbiased);
//   equilibrate by `sweeps` Metropolis steps of every state at β_{k+1}, each step one generation of R candidates.
//
// So the population follows the Boltzmann laws as they cool, like a sequential Monte Carlo sampler, and
//
//   ln Z(β_K) − ln Z(β₀) ≈ Σ_k ln Q_k,
//
// from which the free energy F = −ln Z / β follows (Hukushima and Iba 2003, "Population annealing and its
// application to a spin glass", AIP Conf. Proc. 690, 200–206; Machta 2010, "Population annealing with weighted
// averages: a Monte Carlo method for rough free-energy landscapes", Phys. Rev. E 82, 026704, eqs. 2–5). The
// product of the Q_k estimates Z(β_K)/Z(β₀) without bias; its logarithm is biased low by about half its variance,
// which falls as 1/R. The weights are computed relative to the least energy, exp(−Δβ (Eᵢ − E_min)), so that they
// neither overflow nor underflow, and ln Q_k = −Δβ E_min + ln mean.
//
// Every step's R candidates are evaluated together; being generational, the run does not depend on the order in
// which they return. It stops when the last temperature has been equilibrated.

#pragma once

#include <heuresis/annealing/metropolis.hpp>
#include <heuresis/search/draws.hpp>
#include <heuresis/search/protocol.hpp>
#include <heuresis/selection/parents.hpp>

#include <algorithm>
#include <cmath>
#include <functional>

namespace heuresis::annealing {

template <class G>
class PopulationAnnealing {
 public:
  using Genotype = G;
  using Sampler = std::function<G(calx::random::Stream&)>;
  using Proposal = std::function<G(const G&, calx::random::Stream&)>;

  struct Options {
    std::size_t population = 1000;
    std::vector<double> betas;  // increasing inverse temperatures, β₀ first
    std::size_t sweeps = 10;    // Metropolis steps of every state at each new temperature
    double penalty = infinite_penalty;
  };

  PopulationAnnealing(Sampler sample, Proposal propose, Options options, search::Draws draws = search::Draws{})
      : sample_(std::move(sample)), propose_(std::move(propose)), o_(std::move(options)), draws_(draws) {
    if (o_.population == 0 || o_.betas.size() < 2 || o_.sweeps == 0) throw std::invalid_argument("heuresis::annealing::PopulationAnnealing: a population, two temperatures and a sweep");
    for (std::size_t k = 1; k < o_.betas.size(); k++)
      if (!(o_.betas[k] > o_.betas[k - 1]) || !(o_.betas[0] >= 0)) throw std::invalid_argument("heuresis::annealing::PopulationAnnealing: inverse temperatures nonnegative and increasing");
  }

  std::vector<search::Candidate<G>> ask(std::size_t limit) {
    if (stopped()) return {};
    if (!batch_.active()) {
      std::vector<search::Candidate<G>> c;
      for (std::size_t r = 0; r < o_.population; r++) {
        calx::random::Stream s = draws_(states_.empty() ? initialise : propose, generation_, r);
        if (states_.empty()) c.push_back({search::make_ticket(draws_.island(), serial_++), sample_(s), search::Origin{initialise, {}}});
        else c.push_back({search::make_ticket(draws_.island(), serial_++), propose_(states_[r].genotype, s), search::Origin{propose, {states_[r].ticket}}});
      }
      batch_.open(std::move(c));
    }
    return batch_.take(limit);
  }

  void tell(std::span<const search::Told> told) {
    for (const auto& t : told)
      if (batch_.accept(t)) update();
  }

  bool stopped() const { return done_; }

  // Σ ln Q_k so far: ln Z(β) − ln Z(β₀) at the current β.
  double log_partition_ratio() const { return log_ratio_; }
  // ln Z at the current β, given ln Z(β₀) (ln |X| when β₀ = 0 on a finite space X sampled uniformly).
  double log_partition(double log_z0) const { return log_z0 + log_ratio_; }
  double beta() const { return o_.betas[level_]; }
  std::size_t level() const { return level_; }
  const std::vector<search::Individual<G>>& states() const { return states_; }
  const std::vector<double>& energies() const { return energies_; }
  // The mean energy of the population: the Boltzmann mean at the current β, estimated.
  double mean_energy() const {
    double m = 0;
    for (double e : energies_) m += e;
    return m / static_cast<double>(energies_.size());
  }

  static constexpr std::uint32_t initialise = search::tag("annealing.pa.initialise"), propose = search::tag("annealing.pa.propose"),
                                 accept = search::tag("annealing.pa.accept"), resample = search::tag("annealing.pa.resample");

 private:
  // Reweights and resamples the population to the next temperature, or ends the run.
  void advance() {
    if (level_ + 1 == o_.betas.size()) { done_ = true; return; }
    const double db = o_.betas[level_ + 1] - o_.betas[level_];
    const double lo = *std::min_element(energies_.begin(), energies_.end());
    if (!std::isfinite(lo)) throw std::runtime_error("heuresis::annealing::PopulationAnnealing: every state has infinite energy");
    std::vector<double> w(energies_.size());
    double mean = 0;
    for (std::size_t i = 0; i < w.size(); i++) mean += (w[i] = std::isfinite(energies_[i]) ? std::exp(-db * (energies_[i] - lo)) : 0.0);
    mean /= static_cast<double>(w.size());
    log_ratio_ += -db * lo + std::log(mean);
    calx::random::Stream s = draws_(resample, level_, 0);
    const auto pick = selection::universal_sampling(w, o_.population, s);
    std::vector<search::Individual<G>> next;
    std::vector<double> e;
    for (auto i : pick) { next.push_back(states_[i]); e.push_back(energies_[i]); }
    states_ = std::move(next);
    energies_ = std::move(e);
    level_++;
    sweeps_done_ = 0;
  }

  void update() {
    auto offspring = batch_.individuals();
    batch_.close();
    if (states_.empty()) {
      for (auto& x : offspring) {
        energies_.push_back(energy(x.evaluation, o_.penalty));
        states_.push_back(std::move(x));
      }
      generation_++;
      advance();
      return;
    }
    for (std::size_t r = 0; r < states_.size(); r++) {
      const double e = energy(offspring[r].evaluation, o_.penalty);
      calx::random::Stream s = draws_(accept, generation_, r);
      if (metropolis(energies_[r], e, 1 / beta(), s.uniform())) { states_[r] = std::move(offspring[r]); energies_[r] = e; }
    }
    generation_++;
    if (++sweeps_done_ == o_.sweeps) advance();
  }

  Sampler sample_;
  Proposal propose_;
  Options o_;
  search::Draws draws_;
  search::Batch<G> batch_;
  std::vector<search::Individual<G>> states_;
  std::vector<double> energies_;
  double log_ratio_ = 0;
  std::size_t level_ = 0, sweeps_done_ = 0;
  std::uint64_t generation_ = 0, serial_ = 0;
  bool done_ = false;
};

}  // namespace heuresis::annealing
