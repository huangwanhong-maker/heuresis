// NSGA-II: elitist selection by front and crowding (Deb, Pratap, Agarwal and Meyarivan 2002, "A fast and elitist
// multiobjective genetic algorithm: NSGA-II", IEEE Trans. Evol. Comput. 6, 182–197).
//
// A population of μ is kept. Each generation breeds λ = μ offspring: parents by binary tournament on the
// crowded-comparison order (lower front first; within a front, larger crowding distance first), children by the
// crossover and the mutation given. Parents and offspring are joined, sorted into fronts under constrained
// dominance (dominance.hpp), and the next population is filled front by front; the front that does not fit is cut
// by crowding distance, the least crowded kept. The survivors therefore never get worse in the Pareto sense: the
// non-dominated set of the joined population is kept whole whenever it fits.
//
// Generational under the ask–tell protocol: the first generation is μ samples; afterwards each generation is λ
// offspring, offered in one batch and read in ticket order when complete, so a run is the same whatever order and
// whatever executor its evaluations come back from. Failed evaluations are dominated by every other and lose
// every tournament. Migratory: inject() joins individuals to the population and reselects μ; emigrants(k) gives the
// first k in the crowded-comparison order.

#pragma once

#include <heuresis/multiobjective/dominance.hpp>
#include <heuresis/multiobjective/indicators.hpp>
#include <heuresis/multiobjective/kernels.hpp>
#include <heuresis/search/draws.hpp>
#include <heuresis/selection/parents.hpp>

#include <algorithm>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

namespace heuresis::multiobjective {

// The objective vectors of evaluations; a failed one is +∞ in every objective.
inline std::vector<Point> objective_points(std::span<const Evaluation> e) {
  std::size_t m = 0;
  for (const auto& x : e)
    if (!x.failed()) { m = x.objectives.size(); break; }
  std::vector<Point> out(e.size());
  for (std::size_t i = 0; i < e.size(); i++) out[i] = e[i].failed() ? Point(std::max<std::size_t>(m, 1), std::numeric_limits<double>::infinity()) : e[i].objectives;
  return out;
}

// NSGA-II's environmental selection: the indices of the μ survivors (fronts in order, the last cut by crowding),
// with each survivor's front and crowding distance (computed over its whole front).
struct RankAndCrowding {
  std::vector<std::size_t> survivors;
  std::vector<std::size_t> rank;
  std::vector<double> crowding;
};

inline RankAndCrowding rank_and_crowding(std::span<const Evaluation> e, std::size_t mu) {
  const auto fronts = non_dominated_sort(e);
  const auto points = objective_points(e);
  RankAndCrowding out;
  for (std::size_t k = 0; k < fronts.fronts.size() && out.survivors.size() < mu; k++) {
    const auto& f = fronts.fronts[k];
    const auto d = crowding_distance(points, f);
    std::vector<std::size_t> o(f.size());
    std::iota(o.begin(), o.end(), std::size_t{0});
    if (out.survivors.size() + f.size() > mu) std::stable_sort(o.begin(), o.end(), [&](std::size_t a, std::size_t b) { return d[a] > d[b]; });
    for (std::size_t j = 0; j < o.size() && out.survivors.size() < mu; j++) {
      out.survivors.push_back(f[o[j]]);
      out.rank.push_back(k);
      out.crowding.push_back(d[o[j]]);
    }
  }
  return out;
}

template <class G>
class Nsga2 {
 public:
  using Genotype = G;
  using Individual = heuresis::search::Individual<G>;
  using Candidate = heuresis::search::Candidate<G>;

  Nsga2(Kernels<G> kernels, std::size_t mu, heuresis::search::Draws draws = heuresis::search::Draws{}) : kernels_(std::move(kernels)), mu_(mu), draws_(draws) {
    if (mu < 2) throw std::invalid_argument("heuresis::multiobjective::Nsga2: a population of at least two");
  }

  std::vector<Candidate> ask(std::size_t limit) {
    if (!batch_.active()) open();
    return batch_.take(limit);
  }

  void tell(std::span<const heuresis::search::Told> told) {
    for (const auto& t : told)
      if (batch_.accept(t)) {
        auto pool = population_;
        for (auto& x : batch_.individuals()) pool.push_back(std::move(x));
        batch_.close();
        select(std::move(pool));
        generation_++;
      }
  }

  bool stopped() const { return false; }

  void inject(std::span<const Individual> in) {
    auto pool = population_;
    pool.insert(pool.end(), in.begin(), in.end());
    select(std::move(pool));
  }

  std::vector<Individual> emigrants(std::size_t k) const {
    std::vector<std::size_t> o(population_.size());
    std::iota(o.begin(), o.end(), std::size_t{0});
    std::stable_sort(o.begin(), o.end(), [&](std::size_t a, std::size_t b) { return crowded_less(a, b); });
    std::vector<Individual> out;
    for (std::size_t i = 0; i < std::min(k, o.size()); i++) out.push_back(population_[o[i]]);
    return out;
  }

  const std::vector<Individual>& population() const { return population_; }
  const std::vector<std::size_t>& ranks() const { return rank_; }
  std::uint64_t generation() const { return generation_; }

  // The objective vectors of the first front of the population.
  std::vector<Point> front() const {
    std::vector<Point> out;
    for (std::size_t i = 0; i < population_.size(); i++)
      if (rank_[i] == 0 && !population_[i].evaluation.failed()) out.push_back(population_[i].evaluation.objectives);
    return out;
  }

  static constexpr std::uint32_t sample_tag = heuresis::search::tag("multiobjective.nsga2.sample");
  static constexpr std::uint32_t breed_tag = heuresis::search::tag("multiobjective.nsga2.breed");

 private:
  bool crowded_less(std::size_t a, std::size_t b) const {
    if (rank_[a] != rank_[b]) return rank_[a] < rank_[b];
    return crowding_[a] > crowding_[b];
  }

  void open() {
    std::vector<Candidate> c;
    if (population_.empty()) {
      for (std::size_t i = 0; i < mu_; i++) {
        calx::random::Stream s = draws_(sample_tag, generation_, i);
        c.push_back({heuresis::search::make_ticket(draws_.island(), serial_++), kernels_.initialise(s), {sample_tag, {}}});
      }
    } else {
      const heuresis::selection::Less less = [this](std::size_t a, std::size_t b) { return crowded_less(a, b); };
      for (std::size_t pair = 0; c.size() < mu_; pair++) {
        calx::random::Stream s = draws_(breed_tag, generation_, pair);
        const std::size_t a = heuresis::selection::tournament(population_.size(), 2, less, s);
        const std::size_t b = heuresis::selection::tournament(population_.size(), 2, less, s);
        auto children = kernels_.crossover(population_[a].genotype, population_[b].genotype, s);
        const std::vector<heuresis::search::Ticket> parents{population_[a].ticket, population_[b].ticket};
        c.push_back({heuresis::search::make_ticket(draws_.island(), serial_++), kernels_.mutate(children.first, s), {breed_tag, parents}});
        if (c.size() < mu_) c.push_back({heuresis::search::make_ticket(draws_.island(), serial_++), kernels_.mutate(children.second, s), {breed_tag, parents}});
      }
    }
    batch_.open(std::move(c));
  }

  void select(std::vector<Individual> pool) {
    std::vector<Evaluation> e(pool.size());
    for (std::size_t i = 0; i < pool.size(); i++) e[i] = pool[i].evaluation;
    const auto rc = rank_and_crowding(e, mu_);
    std::vector<Individual> next;
    next.reserve(rc.survivors.size());
    for (auto i : rc.survivors) next.push_back(std::move(pool[i]));
    population_ = std::move(next);
    rank_ = rc.rank;
    crowding_ = rc.crowding;
  }

  Kernels<G> kernels_;
  std::size_t mu_;
  heuresis::search::Draws draws_;
  heuresis::search::Batch<G> batch_;
  std::vector<Individual> population_;
  std::vector<std::size_t> rank_;
  std::vector<double> crowding_;
  std::uint64_t generation_ = 0, serial_ = 0;
};

}  // namespace heuresis::multiobjective
