// Novelty search: selection for being different, not for being good (Lehman and Stanley 2011, "Abandoning
// objectives: evolution through the search for novelty alone", Evol. Comput. 19, 189–223).
//
// On a deceptive problem the objective's gradient leads away from the goal; a search that rewards behaviours
// unlike those seen before is not misled, and on such problems often reaches the goal sooner (their §5–6). The
// novelty of a behaviour b is its sparseness, the mean distance to its k nearest neighbours among the current
// population and an archive of past behaviours (their eq. 1):
//
//   ρ(b) = (1/k) Σ_{j=1..k} dist(b, μ_j).
//
// A behaviour enters the archive when its sparseness exceeds a threshold ρ_min, which rises by a fifth when more
// than `crowded` enter in one generation and falls by a twentieth after `quiet` generations in which none enter
// (the adaptive threshold of Lehman and Stanley's implementation, §3.2). The archive grows only, so the region of
// behaviour space it covers never shrinks.
//
// Generational under the ask–tell protocol: μ offspring are bred by tournaments on novelty (a mutation, or with
// probability p_v a two-parent variation first; either operator suffices alone, as in map_elites.hpp); when the
// generation is told, novelty is computed for parents and offspring together, the archive is extended in ticket
// order, and the μ most novel survive. Behaviours are the evaluations' descriptors; a failure or a missing descriptor
// has novelty 0 and never enters the archive. The objective is ignored by the selection; the driver's elite still
// records the best point seen.

#pragma once

#include <heuresis/diversity/map_elites.hpp>
#include <heuresis/search/draws.hpp>
#include <heuresis/selection/parents.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

namespace heuresis::diversity {

// The mean distance from b to its k nearest points among `others` (fewer if there are fewer).
inline double sparseness(std::span<const double> b, std::span<const std::vector<double>> others, std::size_t k) {
  std::vector<double> d;
  d.reserve(others.size());
  for (const auto& o : others) {
    double s = 0;
    for (std::size_t i = 0; i < b.size(); i++) s += (b[i] - o[i]) * (b[i] - o[i]);
    d.push_back(std::sqrt(s));
  }
  k = std::min(k, d.size());
  if (k == 0) return 0;
  std::partial_sort(d.begin(), d.begin() + static_cast<std::ptrdiff_t>(k), d.end());
  return std::accumulate(d.begin(), d.begin() + static_cast<std::ptrdiff_t>(k), 0.0) / static_cast<double>(k);
}

template <class G>
class NoveltySearch {
 public:
  using Genotype = G;
  using Individual = heuresis::search::Individual<G>;
  using Candidate = heuresis::search::Candidate<G>;

  struct Options {
    std::size_t population = 100;
    std::size_t neighbours = 15;       // k
    double threshold = 0.05;           // the initial ρ_min
    std::size_t crowded = 4;           // more additions than this in a generation raise ρ_min
    std::size_t quiet = 5;             // generations without additions before ρ_min falls
    std::size_t tournament = 2;
    double variation_probability = 0;  // p_v
  };

  NoveltySearch(Kernels<G> kernels, Options options, heuresis::search::Draws draws = heuresis::search::Draws{}) : kernels_(std::move(kernels)), options_(options), draws_(draws), threshold_(options.threshold) {
    if (options_.population < 2) throw std::invalid_argument("heuresis::diversity::NoveltySearch: a population of at least two");
    kernels_.validate("NoveltySearch");
  }

  std::vector<Candidate> ask(std::size_t limit) {
    if (!batch_.active()) open();
    return batch_.take(limit);
  }

  void tell(std::span<const heuresis::search::Told> told) {
    for (const auto& t : told)
      if (batch_.accept(t)) {
        auto offspring = batch_.individuals();
        batch_.close();
        advance(std::move(offspring));
        generation_++;
      }
  }

  bool stopped() const { return false; }

  const std::vector<Individual>& population() const { return population_; }
  const std::vector<double>& novelty() const { return novelty_; }
  const std::vector<std::vector<double>>& archive() const { return archive_; }
  double threshold() const { return threshold_; }
  std::uint64_t generation() const { return generation_; }

  static constexpr std::uint32_t sample_tag = heuresis::search::tag("diversity.novelty.sample");
  static constexpr std::uint32_t breed_tag = heuresis::search::tag("diversity.novelty.breed");

 private:
  bool has_behaviour(const Individual& x) const { return !x.evaluation.failed() && !x.evaluation.descriptors.empty(); }

  void open() {
    std::vector<Candidate> c;
    for (std::size_t i = 0; i < options_.population; i++) {
      if (population_.empty()) {
        calx::random::Stream s = draws_(sample_tag, generation_, i);
        c.push_back({heuresis::search::make_ticket(draws_.island(), serial_++), kernels_.initialise(s), {sample_tag, {}}});
        continue;
      }
      calx::random::Stream s = draws_(breed_tag, generation_, i);
      const heuresis::selection::Less more_novel = [this](std::size_t a, std::size_t b) { return novelty_[a] > novelty_[b]; };
      const auto& a = population_[heuresis::selection::tournament(population_.size(), options_.tournament, more_novel, s)];
      if (kernels_.varies(options_.variation_probability, s)) {
        const auto& b = population_[heuresis::selection::tournament(population_.size(), options_.tournament, more_novel, s)];
        c.push_back({heuresis::search::make_ticket(draws_.island(), serial_++), kernels_.child(a.genotype, b.genotype, s), {breed_tag, {a.ticket, b.ticket}}});
      } else {
        c.push_back({heuresis::search::make_ticket(draws_.island(), serial_++), kernels_.child(a.genotype, s), {breed_tag, {a.ticket}}});
      }
    }
    batch_.open(std::move(c));
  }

  void advance(std::vector<Individual> offspring) {
    const std::size_t parents = population_.size();
    std::vector<Individual> pool = std::move(population_);
    for (auto& x : offspring) pool.push_back(std::move(x));
    // Novelty against the pool and the archive, each point excluding itself.
    std::vector<std::vector<double>> behaviours;
    std::vector<std::size_t> where(pool.size(), static_cast<std::size_t>(-1));
    for (std::size_t i = 0; i < pool.size(); i++)
      if (has_behaviour(pool[i])) { where[i] = behaviours.size(); behaviours.push_back(pool[i].evaluation.descriptors); }
    std::vector<std::vector<double>> others;
    std::vector<double> score(pool.size(), 0.0);
    for (std::size_t i = 0; i < pool.size(); i++) {
      if (where[i] == static_cast<std::size_t>(-1)) continue;
      others.clear();
      for (std::size_t j = 0; j < behaviours.size(); j++)
        if (j != where[i]) others.push_back(behaviours[j]);
      others.insert(others.end(), archive_.begin(), archive_.end());
      score[i] = sparseness(pool[i].evaluation.descriptors, others, options_.neighbours);
    }
    std::size_t added = 0;
    for (std::size_t i = parents; i < pool.size(); i++)
      if (where[i] != static_cast<std::size_t>(-1) && score[i] > threshold_) {
        archive_.push_back(pool[i].evaluation.descriptors);
        added++;
      }
    if (added > options_.crowded) threshold_ *= 1.2;
    quiet_ = added ? 0 : quiet_ + 1;
    if (quiet_ >= options_.quiet) { threshold_ *= 0.95; quiet_ = 0; }
    std::vector<std::size_t> o(pool.size());
    std::iota(o.begin(), o.end(), std::size_t{0});
    std::stable_sort(o.begin(), o.end(), [&](std::size_t a, std::size_t b) { return score[a] > score[b]; });
    o.resize(std::min(options_.population, o.size()));
    population_.clear();
    novelty_.clear();
    for (auto i : o) {
      population_.push_back(std::move(pool[i]));
      novelty_.push_back(score[i]);
    }
  }

  Kernels<G> kernels_;
  Options options_;
  heuresis::search::Draws draws_;
  heuresis::search::Batch<G> batch_;
  std::vector<Individual> population_;
  std::vector<double> novelty_;
  std::vector<std::vector<double>> archive_;
  double threshold_;
  std::size_t quiet_ = 0;
  std::uint64_t generation_ = 0, serial_ = 0;
};

}  // namespace heuresis::diversity
