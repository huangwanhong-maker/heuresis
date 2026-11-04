// Simulated annealing on the ask–tell protocol.
//
// One chain: a state x, its energy, and a schedule. Step k proposes y from x by the kernel, drawn from the stream
// (propose, k, 0); when y's evaluation is told, the Metropolis rule at T_k accepts it or not with the uniform of the
// stream (accept, k, 0), and the schedule advances (Kirkpatrick, Gelatt and Vecchi 1983). The first candidate is the
// initial state, from the sampler. Since each step depends on the last, the chain offers one candidate and nothing
// more until it is told (breadth 1 in the protocol's sense, search/protocol.hpp), so a driver without a cap cannot
// breed ahead of it; spreading a search of this kind over workers is parallel tempering's or population annealing's.
//
// It is injectable: an injected individual of lower energy becomes the state (the best of a neighbour island, a
// restart); its emigrant is its state.

#pragma once

#include <heuresis/annealing/metropolis.hpp>
#include <heuresis/annealing/schedule.hpp>
#include <heuresis/search/draws.hpp>
#include <heuresis/search/protocol.hpp>

#include <functional>
#include <optional>

namespace heuresis::annealing {

template <class G>
class SimulatedAnnealing {
 public:
  using Genotype = G;
  using Sampler = std::function<G(calx::random::Stream&)>;
  using Proposal = std::function<G(const G&, calx::random::Stream&)>;

  struct Options {
    double penalty = infinite_penalty;
    double min_temperature = 0;  // stopped below it
  };

  SimulatedAnnealing(Sampler sample, Proposal propose, Schedule schedule, search::Draws draws = search::Draws{}, Options options = {})
      : sample_(std::move(sample)), propose_(std::move(propose)), schedule_(schedule), draws_(draws), o_(options) {}

  std::vector<search::Candidate<G>> ask(std::size_t limit) {
    if (limit == 0 || pending_) return {};
    calx::random::Stream s = draws_(state_ ? propose : initialise, schedule_.step(), 0);
    search::Candidate<G> c{search::make_ticket(draws_.island(), serial_++), state_ ? propose_(state_->genotype, s) : sample_(s),
                           search::Origin{state_ ? propose : initialise, state_ ? std::vector<search::Ticket>{state_->ticket} : std::vector<search::Ticket>{}}};
    pending_ = c;
    return {c};
  }

  void tell(std::span<const search::Told> told) {
    for (const auto& t : told) {
      if (!pending_ || t.ticket != pending_->ticket) throw std::invalid_argument("heuresis::annealing::SimulatedAnnealing: a ticket that is not outstanding");
      search::Individual<G> y{pending_->ticket, std::move(pending_->genotype), t.evaluation, std::move(pending_->origin)};
      pending_.reset();
      const double e = energy(y.evaluation, o_.penalty);
      if (!state_) {
        state_ = std::move(y);
        energy_ = e;
        continue;
      }
      calx::random::Stream s = draws_(accept, schedule_.step(), 0);
      const bool ok = metropolis(energy_, e, schedule_.temperature(), s.uniform());
      if (ok) {
        state_ = std::move(y);
        energy_ = e;
        accepted_++;
      }
      schedule_.advance(ok);
    }
  }

  bool stopped() const { return schedule_.temperature() < o_.min_temperature; }

  void inject(std::span<const search::Individual<G>> in) {
    for (const auto& x : in) {
      const double e = energy(x.evaluation, o_.penalty);
      if (!state_ || e < energy_) { state_ = x; energy_ = e; }
    }
  }
  std::vector<search::Individual<G>> emigrants(std::size_t k) const {
    if (!state_ || k == 0) return {};
    return {*state_};
  }

  const std::optional<search::Individual<G>>& state() const { return state_; }
  double current_energy() const { return energy_; }
  const Schedule& schedule() const { return schedule_; }
  std::uint64_t accepted() const { return accepted_; }

  static constexpr std::uint32_t initialise = search::tag("annealing.sa.initialise"), propose = search::tag("annealing.sa.propose"), accept = search::tag("annealing.sa.accept");

 private:
  Sampler sample_;
  Proposal propose_;
  Schedule schedule_;
  search::Draws draws_;
  Options o_;
  std::optional<search::Individual<G>> state_;
  std::optional<search::Candidate<G>> pending_;
  double energy_ = std::numeric_limits<double>::infinity();
  std::uint64_t serial_ = 0, accepted_ = 0;
};

}  // namespace heuresis::annealing
