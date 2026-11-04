// Cellular evolutionary algorithms: a population on a graph, each individual breeding with its neighbours.
//
// Each vertex of a topology holds one individual. Every generation each cell chooses its parents from its closed
// neighbourhood (itself and its out-neighbours), breeds one offspring, and keeps it when it is not worse than the
// cell's present occupant; all cells are replaced together (synchronous update). Selection is local, so good
// individuals spread as a wave and diversity survives longer than in one panmictic population (Manderick and
// Spiessens 1989, "Fine-grained parallel genetic algorithms", ICGA 3, 428–433; Alba and Dorronsoro 2008, Cellular
// Genetic Algorithms, Springer).
//
// The speed of that wave is measured by the takeover time: with selection alone (no variation) and a cell
// keeping the best of its closed neighbourhood, after g generations each cell holds the best of the ball of
// radius g about it, so a single best individual at s fills the graph in exactly ecc(s) generations, its
// eccentricity: ⌈(k − 1)/2⌉ on a ring of k, ⌊r/2⌋ + ⌊c/2⌋ on an r × c torus with von Neumann's neighbourhood
// (Sarma and De Jong 1996, "An analysis of the effects of neighborhood size and shape on local selection
// algorithms", PPSN IV, 236–244, for the takeover curves; Giacobini, Tomassini and Tettamanzi 2005, "Takeover
// time curves in random and small-world structured populations", GECCO, for their models).
//
// Parent selection: the best of the closed neighbourhood (ties to the lower index), or binary tournaments in
// it. It is generational in the protocol's sense.

#pragma once

#include <heuresis/populations/topology.hpp>
#include <heuresis/search/search.hpp>

#include <functional>
#include <optional>
#include <vector>

namespace heuresis::populations {

using heuresis::search::Candidate;
using heuresis::search::Individual;
using heuresis::search::Told;

enum class CellSelection : std::uint8_t { best, tournament };

template <class G>
class CellularEA {
 public:
  using Genotype = G;
  using Mutation = std::function<G(const G&, calx::random::Stream&)>;
  using Crossover = std::function<std::pair<G, G>(const G&, const G&, calx::random::Stream&)>;
  static constexpr std::uint32_t initial_tag = heuresis::search::tag("populations.cellular.initial");
  static constexpr std::uint32_t breed_tag = heuresis::search::tag("populations.cellular.breed");

  // One initial genotype per vertex.
  CellularEA(std::vector<G> initial, Topology neighbourhoods, CellSelection selection = CellSelection::tournament, Mutation mutate = {}, Crossover crossover = {}, double crossover_rate = 1,
             heuresis::search::Draws draws = heuresis::search::Draws{})
      : initial_(std::move(initial)), topology_(std::move(neighbourhoods)), selection_(selection), mutate_(std::move(mutate)), crossover_(std::move(crossover)), crossover_rate_(crossover_rate), draws_(draws) {
    if (initial_.size() != topology_.size()) throw std::invalid_argument("heuresis::populations::CellularEA: one initial individual per vertex");
  }

  std::vector<Candidate<G>> ask(std::size_t limit) {
    using heuresis::search::make_ticket;
    if (!batch_.active()) {
      std::vector<Candidate<G>> g;
      if (cells_.empty()) {
        for (auto& x : initial_) g.push_back({make_ticket(draws_.island(), serial_++), std::move(x), {initial_tag, {}}});
        initial_.clear();
      } else {
        for (std::size_t i = 0; i < cells_.size(); i++) {
          calx::random::Stream s = draws_(breed_tag, generation_, i);
          const std::size_t a = choose(i, s);
          G child = cells_[a].genotype;
          heuresis::search::Origin origin{breed_tag, {cells_[a].ticket}};
          if (crossover_ && s.uniform() < crossover_rate_) {
            const std::size_t b = choose(i, s);
            child = crossover_(cells_[a].genotype, cells_[b].genotype, s).first;
            origin.parents.push_back(cells_[b].ticket);
          }
          if (mutate_) child = mutate_(child, s);
          g.push_back({make_ticket(draws_.island(), serial_++), std::move(child), std::move(origin)});
        }
      }
      batch_.open(std::move(g));
    }
    return batch_.take(limit);
  }

  void tell(std::span<const Told> told) {
    for (const auto& t : told) {
      if (!batch_.accept(t)) continue;
      auto xs = batch_.individuals();
      batch_.close();
      if (cells_.empty()) cells_ = std::move(xs);
      else
        for (std::size_t i = 0; i < cells_.size(); i++)
          if (!heuresis::search::better(cells_[i].evaluation, xs[i].evaluation)) cells_[i] = std::move(xs[i]);
      generation_++;
    }
  }

  bool stopped() const { return false; }
  const std::vector<Individual<G>>& cells() const { return cells_; }
  std::uint64_t generation() const { return generation_; }

 private:
  // A parent from the closed neighbourhood of cell i.
  std::size_t choose(std::size_t i, calx::random::Stream& s) const {
    const auto& nb = topology_.out[i];
    auto at = [&](std::size_t r) { return r == 0 ? i : nb[r - 1]; };
    const std::size_t m = nb.size() + 1;
    auto beats = [&](std::size_t a, std::size_t b) { return heuresis::search::better(cells_[a].evaluation, cells_[b].evaluation) || (!heuresis::search::better(cells_[b].evaluation, cells_[a].evaluation) && a < b); };
    if (selection_ == CellSelection::best) {
      std::size_t best = i;
      for (std::size_t r = 1; r < m; r++)
        if (beats(at(r), best)) best = at(r);
      return best;
    }
    const std::size_t a = at(s.below(m)), b = at(s.below(m));
    return beats(a, b) ? a : b;
  }

  std::vector<G> initial_;
  Topology topology_;
  CellSelection selection_;
  Mutation mutate_;
  Crossover crossover_;
  double crossover_rate_;
  heuresis::search::Draws draws_;
  heuresis::search::Batch<G> batch_;
  std::vector<Individual<G>> cells_;
  std::uint64_t generation_ = 0, serial_ = 0;
};

}  // namespace heuresis::populations
