// SMS-EMOA: steady-state selection by the hypervolume (Beume, Naujoks and Emmerich 2007, "SMS-EMOA:
// multiobjective selection based on dominated hypervolume", Eur. J. Oper. Res. 181, 1653–1669; Emmerich, Beume and
// Naujoks 2005, EMO).
//
// One offspring at a time joins the population of μ; of the μ + 1, one is removed: from the last front under
// constrained dominance, the point whose exclusive hypervolume contribution is least (hypervolume.hpp). By strict
// Pareto compliance the population's hypervolume never decreases, and it converges to the μ-point configurations
// that maximise it (Beume et al. 2007, §3). When the last front is infeasible, the most violating point goes
// instead; when it holds failures, a failure goes.
//
// The reference point of the contributions is the worst value of each objective over the μ + 1 plus one, so the
// extremes of the front have finite, large contributions and are kept (the choice of Beume et al. 2007, §3.2 for
// two objectives).
//
// Asynchronous under the ask–tell protocol: it offers offspring whenever asked, bred from the population as it
// stands (two parents drawn uniformly), the k-th offspring from the stream keyed by k; each tell inserts one
// individual and removes one. It offers at most `breadth` offspring per ask (one by default, the steady state:
// a driver that asks for more gets them over later asks, each bred after the tells before it), so that a large
// request does not breed a whole generation from one state. Its run is reproducible from the order of arrival.
// The first μ candidates are samples, offered together; the population grows to μ before anything is removed,
// and offspring wait until two are told.

#pragma once

#include <heuresis/multiobjective/dominance.hpp>
#include <heuresis/multiobjective/hypervolume.hpp>
#include <heuresis/multiobjective/kernels.hpp>
#include <heuresis/multiobjective/nsga2.hpp>
#include <heuresis/search/draws.hpp>

#include <algorithm>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace heuresis::multiobjective {

template <class G>
class SmsEmoa {
 public:
  using Genotype = G;
  using Individual = heuresis::search::Individual<G>;
  using Candidate = heuresis::search::Candidate<G>;

  SmsEmoa(Kernels<G> kernels, std::size_t mu, heuresis::search::Draws draws = heuresis::search::Draws{}, std::size_t breadth = 1)
      : kernels_(std::move(kernels)), mu_(mu), breadth_(std::max<std::size_t>(1, breadth)), draws_(draws) {
    if (mu < 2) throw std::invalid_argument("heuresis::multiobjective::SmsEmoa: a population of at least two");
  }

  std::vector<Candidate> ask(std::size_t limit) {
    std::vector<Candidate> out;
    std::size_t offspring = 0;
    while (out.size() < limit) {
      if (serial_ < mu_) {
        calx::random::Stream s = draws_(sample_tag, 0, serial_);
        out.push_back({heuresis::search::make_ticket(draws_.island(), serial_), kernels_.initialise(s), {sample_tag, {}}});
      } else {
        if (population_.size() < 2 || offspring == breadth_) break;
        offspring++;
        calx::random::Stream s = draws_(breed_tag, 0, serial_);
        const std::size_t a = s.below(population_.size()), b = s.below(population_.size());
        auto children = kernels_.crossover(population_[a].genotype, population_[b].genotype, s);
        out.push_back({heuresis::search::make_ticket(draws_.island(), serial_), kernels_.mutate(children.first, s), {breed_tag, {population_[a].ticket, population_[b].ticket}}});
      }
      serial_++;
      pending_.add(out.back());
    }
    return out;
  }

  void tell(std::span<const heuresis::search::Told> told) {
    for (const auto& t : told) {
      auto [c, note] = pending_.take(t.ticket);
      (void)note;
      population_.push_back({c.ticket, std::move(c.genotype), t.evaluation, std::move(c.origin)});
      if (population_.size() > mu_) reduce();
    }
  }

  bool stopped() const { return false; }

  void inject(std::span<const Individual> in) {
    for (const auto& x : in) {
      population_.push_back(x);
      if (population_.size() > mu_) reduce();
    }
  }
  std::vector<Individual> emigrants(std::size_t k) const {
    std::vector<Evaluation> e;
    for (const auto& x : population_) e.push_back(x.evaluation);
    const auto rc = rank_and_crowding(e, std::min(k, e.size()));
    std::vector<Individual> out;
    for (auto i : rc.survivors) out.push_back(population_[i]);
    return out;
  }

  const std::vector<Individual>& population() const { return population_; }
  std::vector<Point> front() const {
    std::vector<Evaluation> e;
    for (const auto& x : population_) e.push_back(x.evaluation);
    const auto f = non_dominated_sort(e);
    std::vector<Point> out;
    if (!f.fronts.empty())
      for (auto i : f.fronts[0])
        if (!e[i].failed()) out.push_back(e[i].objectives);
    return out;
  }

  static constexpr std::uint32_t sample_tag = heuresis::search::tag("multiobjective.sms_emoa.sample");
  static constexpr std::uint32_t breed_tag = heuresis::search::tag("multiobjective.sms_emoa.breed");

 private:
  void reduce() {
    std::vector<Evaluation> e;
    for (const auto& x : population_) e.push_back(x.evaluation);
    const auto fr = non_dominated_sort(e);
    const auto& last = fr.fronts.back();
    std::size_t remove = last.back();
    bool failed = false, infeasible = false;
    for (auto i : last) { failed = failed || e[i].failed(); infeasible = infeasible || e[i].violation() > 0; }
    if (failed) {
      for (auto i : last)
        if (e[i].failed()) remove = i;
    } else if (infeasible) {
      double worst = -1;
      for (auto i : last)
        if (e[i].violation() >= worst) { worst = e[i].violation(); remove = i; }
    } else if (last.size() > 1) {
      std::vector<Point> pts;
      for (auto i : last) pts.push_back(e[i].objectives);
      const std::size_t m = pts[0].size();
      Point ref(m, -std::numeric_limits<double>::infinity());
      for (std::size_t i = 0; i < e.size(); i++)
        if (!e[i].failed())
          for (std::size_t k = 0; k < m; k++) ref[k] = std::max(ref[k], e[i].objectives[k]);
      for (auto& v : ref) v += 1;
      const auto c = contributions(pts, ref);
      std::size_t least = 0;
      for (std::size_t q = 1; q < c.size(); q++)
        if (c[q] < c[least]) least = q;
      remove = last[least];
    }
    population_.erase(population_.begin() + static_cast<std::ptrdiff_t>(remove));
  }

  Kernels<G> kernels_;
  std::size_t mu_, breadth_;
  heuresis::search::Draws draws_;
  heuresis::search::Pending<G> pending_;
  std::vector<Individual> population_;
  std::uint64_t serial_ = 0;
};

}  // namespace heuresis::multiobjective
