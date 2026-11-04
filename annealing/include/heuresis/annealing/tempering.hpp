// Parallel tempering (replica exchange) on the ask–tell protocol.
//
// R Metropolis chains run at temperatures T₁ < … < T_R, one state at each. At each generation every chain proposes
// one move (R candidates, evaluated together, in parallel if the executor is) and accepts it by the Metropolis
// rule at its temperature; then neighbouring temperatures (i, i + 1), the pairs with i even at even generations and
// odd at odd ones, exchange their states with probability
//
//   min(1, exp((βᵢ − β_{i+1})(E(xᵢ) − E(x_{i+1})))),   β = 1/T,
//
// which keeps the product of the Boltzmann laws Π π_{Tᵢ} stationary: each exchange satisfies detailed balance for
// it (Swendsen and Wang 1986, "Replica Monte Carlo simulation of spin-glasses", Phys. Rev. Lett. 57, 2607–2609;
// Geyer 1991, "Markov chain Monte Carlo maximum likelihood", Computing Science and Statistics 23, 156–163;
// Hukushima and Nemoto 1996, "Exchange Monte Carlo method and application to spin glass simulations", J. Phys.
// Soc. Japan 65, 1604–1608). The hot chains cross barriers and hand their states down; the coldest samples the
// lowest energies. Being generational, its states do not depend on the order in which evaluations return.

#pragma once

#include <heuresis/annealing/metropolis.hpp>
#include <heuresis/search/draws.hpp>
#include <heuresis/search/protocol.hpp>

#include <functional>

namespace heuresis::annealing {

template <class G>
class ParallelTempering {
 public:
  using Genotype = G;
  using Sampler = std::function<G(calx::random::Stream&)>;
  using Proposal = std::function<G(const G&, calx::random::Stream&)>;

  ParallelTempering(Sampler sample, Proposal propose, std::vector<double> temperatures, search::Draws draws = search::Draws{}, double penalty = infinite_penalty)
      : sample_(std::move(sample)), propose_(std::move(propose)), t_(std::move(temperatures)), draws_(draws), penalty_(penalty),
        attempts_(t_.size() > 1 ? t_.size() - 1 : 0, 0), swaps_(attempts_.size(), 0) {
    if (t_.empty()) throw std::invalid_argument("heuresis::annealing::ParallelTempering: no temperatures");
    for (std::size_t i = 0; i < t_.size(); i++)
      if (!(t_[i] > 0) || (i > 0 && !(t_[i] > t_[i - 1]))) throw std::invalid_argument("heuresis::annealing::ParallelTempering: temperatures positive and increasing");
  }

  std::vector<search::Candidate<G>> ask(std::size_t limit) {
    if (!batch_.active()) {
      std::vector<search::Candidate<G>> c;
      for (std::size_t r = 0; r < t_.size(); r++) {
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

  bool stopped() const { return false; }

  // The state at each temperature, coldest first, and its energy.
  const std::vector<search::Individual<G>>& states() const { return states_; }
  const std::vector<double>& energies() const { return energies_; }
  const std::vector<double>& temperatures() const { return t_; }
  std::uint64_t generation() const { return generation_; }
  // The fraction of attempted exchanges accepted between temperatures i and i + 1.
  double exchange_rate(std::size_t i) const { return attempts_[i] ? static_cast<double>(swaps_[i]) / static_cast<double>(attempts_[i]) : 0; }

  void inject(std::span<const search::Individual<G>> in) {
    for (const auto& x : in) {
      const double e = energy(x.evaluation, penalty_);
      if (!states_.empty() && e < energies_[0]) { states_[0] = x; energies_[0] = e; }
    }
  }
  std::vector<search::Individual<G>> emigrants(std::size_t k) const {
    if (states_.empty() || k == 0) return {};
    return {states_[0]};
  }

  static constexpr std::uint32_t initialise = search::tag("annealing.pt.initialise"), propose = search::tag("annealing.pt.propose"),
                                 accept = search::tag("annealing.pt.accept"), exchange = search::tag("annealing.pt.exchange");

 private:
  void update() {
    auto offspring = batch_.individuals();
    batch_.close();
    if (states_.empty()) {
      for (auto& x : offspring) {
        energies_.push_back(energy(x.evaluation, penalty_));
        states_.push_back(std::move(x));
      }
    } else {
      for (std::size_t r = 0; r < t_.size(); r++) {
        const double e = energy(offspring[r].evaluation, penalty_);
        calx::random::Stream s = draws_(accept, generation_, r);
        if (metropolis(energies_[r], e, t_[r], s.uniform())) { states_[r] = std::move(offspring[r]); energies_[r] = e; }
      }
      for (std::size_t i = generation_ % 2; i + 1 < t_.size(); i += 2) {
        calx::random::Stream s = draws_(exchange, generation_, i);
        const double a = (1 / t_[i] - 1 / t_[i + 1]) * (energies_[i] - energies_[i + 1]);
        attempts_[i]++;
        if (a >= 0 || s.uniform() < std::exp(a)) {
          std::swap(states_[i], states_[i + 1]);
          std::swap(energies_[i], energies_[i + 1]);
          swaps_[i]++;
        }
      }
    }
    generation_++;
  }

  Sampler sample_;
  Proposal propose_;
  std::vector<double> t_;
  search::Draws draws_;
  double penalty_;
  search::Batch<G> batch_;
  std::vector<search::Individual<G>> states_;
  std::vector<double> energies_;
  std::vector<std::uint64_t> attempts_, swaps_;
  std::uint64_t generation_ = 0, serial_ = 0;
};

}  // namespace heuresis::annealing
