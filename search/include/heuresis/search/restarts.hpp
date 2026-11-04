// Independent restarts of any search: whenever the current run has stopped, a new run is made and the search goes on.
//
// A search that settles (a (1+1)-ES whose step has collapsed, a tabu search at a deep optimum, a distribution whose
// frequencies have reached their borders) spends the rest of its budget where it is. Restarting it from scratch
// turns it into the strategy that repeats independent runs until one succeeds, whose expected cost is the expected
// running time of analysis/runtimes.hpp (Auger and Hansen 2005, "Performance evaluation of an advanced local search
// evolutionary algorithm", IEEE CEC, 1777–1784). The wrapper is that strategy, for any algorithm of the protocol:
//
// - a factory makes run r from its number and its draws, A(std::size_t restart, Draws); run r draws from streams keyed
//   by the wrapper's seed folded with r (as CmaRestarts keys its runs, strategies/restarts.hpp), so the sequence of
//   runs is reproducible and independent of where its evaluations are made;
// - a new run begins when the current one reports stopped() and nothing it offered is still outstanding, or when it
//   has offered the evaluations of a per-run budget, after which it is ended as if it had stopped; the wrapper stops
//   after its last run;
// - each run numbers its candidates from zero, so the wrapper issues tickets of its own, unique across runs, maps them
//   to the run's tickets and back, and rewrites the parents of each origin into its own tickets: the genealogy of the
//   whole sequence is one acyclic graph, whose sources are the runs' first samples.
//
// It is what it says, independent restarts: nothing one run learnt is handed to the next. A restart that grows the
// population (IPOP: Auger and Hansen 2005, "A restart CMA evolution strategy with increasing population size", IEEE
// CEC, 1769–1776), shrinks the step size or starts near the last run's best is the factory's business, which receives
// the restart's number (CMA-ES's own IPOP and BIPOP are strategies/restarts.hpp). A generational algorithm's runs stay
// generational and an asynchronous one's asynchronous. Migrants are not taken (no inject): a migrant belongs to no run.

#pragma once

#include <heuresis/search/draws.hpp>
#include <heuresis/search/protocol.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace heuresis::search {

struct RestartLimits {
  std::size_t runs = unlimited;                                  // the most runs, the first included
  double evaluations = std::numeric_limits<double>::infinity();  // offered by one run, after which it is ended
};

template <Optimiser A>
class Restarts {
 public:
  using Genotype = typename A::Genotype;
  using Factory = std::function<A(std::size_t restart, Draws draws)>;

  explicit Restarts(Factory factory, RestartLimits limits = {}, Draws draws = Draws{}) : factory_(std::move(factory)), limits_(limits), draws_(draws) {
    if (!factory_ || limits_.runs == 0 || !(limits_.evaluations >= 1)) throw std::invalid_argument("heuresis::search::Restarts: a factory, at least one run and a per-run budget of at least one evaluation");
    begin();
  }

  std::vector<Candidate<Genotype>> ask(std::size_t limit) {
    if (ended() && outstanding_.empty()) {
      if (runs_ >= limits_.runs) return {};
      begin();
    }
    if (ended()) return {};  // the run's last candidates are still outstanding, or a new run offers nothing
    if (std::isfinite(limits_.evaluations)) limit = std::min(limit, static_cast<std::size_t>(std::ceil(limits_.evaluations - offered_)));
    std::vector<Candidate<Genotype>> inner = current_->ask(limit);
    std::vector<Candidate<Genotype>> out;
    out.reserve(inner.size());
    for (auto& c : inner) {
      for (auto& p : c.origin.parents) {
        const auto it = issued_.find(p);
        if (it != issued_.end()) p = it->second;
      }
      const Ticket t = make_ticket(draws_.island(), serial_++);
      issued_.emplace(c.ticket, t);
      outstanding_.emplace(t, c.ticket);
      out.push_back({t, std::move(c.genotype), std::move(c.origin)});
    }
    offered_ += static_cast<double>(out.size());
    return out;
  }

  void tell(std::span<const Told> told) {
    std::vector<Told> inner;
    inner.reserve(told.size());
    for (const auto& t : told) {
      const auto it = outstanding_.find(t.ticket);
      if (it == outstanding_.end()) throw std::invalid_argument("heuresis::search::Restarts: a ticket that is not outstanding");
      inner.push_back({it->second, t.evaluation});
      outstanding_.erase(it);
    }
    current_->tell(std::span<const Told>(inner));
  }

  bool stopped() const { return runs_ >= limits_.runs && ended() && outstanding_.empty(); }

  // The number of runs begun, the current run, and what it has offered.
  std::size_t runs() const { return runs_; }
  const A& current() const { return *current_; }
  double offered() const { return offered_; }

 private:
  bool ended() const { return current_->stopped() || offered_ >= limits_.evaluations; }

  void begin() {
    const Draws run_draws(mix(draws_.seed() ^ mix(runs_ + 1)), draws_.island());
    current_.reset();
    current_.emplace(factory_(runs_, run_draws));
    issued_.clear();
    offered_ = 0;
    runs_++;
  }

  Factory factory_;
  RestartLimits limits_;
  Draws draws_;
  std::optional<A> current_;
  std::unordered_map<Ticket, Ticket> issued_;       // the current run's tickets → the wrapper's
  std::unordered_map<Ticket, Ticket> outstanding_;  // the wrapper's tickets not yet told → the run's
  double offered_ = 0;
  std::size_t runs_ = 0;
  std::uint64_t serial_ = 0;
};

}  // namespace heuresis::search
