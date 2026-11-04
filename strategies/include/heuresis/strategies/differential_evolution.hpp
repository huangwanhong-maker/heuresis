// Differential evolution, and SHADE.
//
// Differential evolution moves a population of NP points by differences of its own members (Storn and Price
// 1997, "Differential evolution — a simple and efficient heuristic for global optimization over continuous
// spaces", J. Global Optim. 11, 341–359). For each target x_i a mutant is formed,
//
//   rand/1             v = x_r1 + F (x_r2 − x_r3)
//   best/1             v = x_best + F (x_r1 − x_r2)
//   current-to-best/1  v = x_i + F (x_best − x_i) + F (x_r1 − x_r2)
//   rand/2             v = x_r1 + F (x_r2 − x_r3) + F (x_r4 − x_r5)
//
// with r1, r2, … distinct and distinct from i; crossed with the target, binomially (each coordinate from the
// mutant with probability CR, one coordinate j_rand always) or exponentially (a cyclic run of coordinates from a
// random start, continued with probability CR); and the trial u replaces x_i unless x_i is better (Deb's rule).
// Because its steps are differences of the population, their scale and shape follow the population's spread:
// the search contracts as the population does. Coordinates that leave the box are set halfway between the bound
// and the target's coordinate (the rule of SHADE; Tanabe and Fukunaga 2013), or repaired by any of variation's
// repairs.
//
// SHADE adapts F and CR from a history of the values that succeeded (Tanabe and Fukunaga 2013, "Success-history
// based parameter adaptation for differential evolution", CEC, 71–78, Algorithm 1): each target draws a memory
// slot r, CR_i ~ N(M_CR[r], 0.1) clipped to [0, 1] and F_i ~ Cauchy(M_F[r], 0.1) redrawn until positive and cut
// at 1; the mutation is current-to-pbest/1, v = x_i + F_i (x_pbest − x_i) + F_i (x_r1 − x̃_r2), x_pbest uniform
// among the best ⌈p_i NP⌉ with p_i ~ U[2/NP, 0.2] and x̃_r2 drawn from the population and an archive of replaced
// parents (JADE's; Zhang and Sanderson 2009, IEEE Trans. Evol. Comput. 13, 945–958); a trial that is strictly
// better records (CR_i, F_i) with weight |Δf|, its parent goes to the archive (kept at NP by random deletion),
// and the slot k of the memory, k cycling over the H slots, takes the weighted mean of the successful CR and the
// weighted Lehmer mean Σ wF²/Σ wF of the successful F.
//
// Both are generational: the first generation is NP uniform points of the box; each later one, the NP trials.
// A generation's selection reads the trials in ticket order, so the run does not depend on the order in which
// evaluations return. Each trial's origin lists its target first, then the members its mutant read.

#pragma once

#include <heuresis/search/draws.hpp>
#include <heuresis/search/protocol.hpp>
#include <heuresis/variation/real.hpp>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <vector>

namespace heuresis::strategies {

enum class DeMutation : std::uint8_t { rand_1, best_1, current_to_best_1, rand_2 };
enum class DeCrossover : std::uint8_t { binomial, exponential };

struct DeOptions {
  std::size_t population = 0;  // 0: 10n
  double F = 0.5, CR = 0.9;
  DeMutation mutation = DeMutation::rand_1;
  DeCrossover crossover = DeCrossover::binomial;
  bool midpoint = true;  // false: repair by `repair`
  Repair repair = Repair::reflect;
};

namespace detail {

// r distinct indices in [0, n), none in `exclude`.
inline std::vector<std::size_t> distinct(std::size_t n, std::size_t r, std::vector<std::size_t> exclude, calx::random::Stream& s) {
  if (n < r + exclude.size()) throw std::invalid_argument("heuresis::strategies: the population is too small for the strategy");
  std::vector<std::size_t> out;
  while (out.size() < r) {
    const std::size_t c = s.below(n);
    if (std::find(exclude.begin(), exclude.end(), c) != exclude.end()) continue;
    out.push_back(c);
    exclude.push_back(c);
  }
  return out;
}

// Binomial or exponential crossover of a target and a mutant.
inline Real crossover(const Real& target, const Real& mutant, double cr, DeCrossover kind, calx::random::Stream& s) {
  const std::size_t n = target.size();
  Real u = target;
  const std::size_t j0 = s.below(n);
  if (kind == DeCrossover::binomial) {
    for (std::size_t j = 0; j < n; j++)
      if (j == j0 || s.uniform() < cr) u[j] = mutant[j];
  } else {
    std::size_t j = j0, L = 0;
    do {
      u[j] = mutant[j];
      j = (j + 1) % n;
      L++;
    } while (L < n && s.uniform() < cr);
  }
  return u;
}

inline void bring_back(Real& u, const Real& target, const Box& b, bool midpoint, Repair repair, calx::random::Stream& s) {
  if (midpoint) {
    for (std::size_t j = 0; j < u.size(); j++) {
      if (u[j] < b.lower[j]) u[j] = 0.5 * (b.lower[j] + target[j]);
      else if (u[j] > b.upper[j]) u[j] = 0.5 * (b.upper[j] + target[j]);
    }
  } else {
    heuresis::variation::repair(u, b, repair, s);
  }
}

}  // namespace detail

class DifferentialEvolution {
 public:
  using Genotype = Real;
  using Individual = heuresis::search::Individual<Real>;
  static constexpr std::uint32_t init_purpose = heuresis::search::tag("strategies.de.initialise");
  static constexpr std::uint32_t trial_purpose = heuresis::search::tag("strategies.de.trial");

  DifferentialEvolution(Box box, DeOptions options = {}, heuresis::search::Draws draws = heuresis::search::Draws{}) : box_(std::move(box)), options_(options), draws_(draws) {
    box_.validate();
    if (options_.population == 0) options_.population = 10 * box_.size();
    const std::size_t need = options_.mutation == DeMutation::rand_2 ? 6 : 4;
    if (options_.population < need) throw std::invalid_argument("heuresis::strategies::DifferentialEvolution: too small a population for the strategy");
  }

  std::vector<heuresis::search::Candidate<Real>> ask(std::size_t limit) {
    if (!batch_.active()) open();
    return batch_.take(limit);
  }

  void tell(std::span<const heuresis::search::Told> told) {
    for (const auto& t : told)
      if (batch_.accept(t)) close();
  }

  bool stopped() const { return false; }

  // Evaluated solutions from outside replace the worst members, once the population exists.
  void inject(std::span<const Individual> in) {
    for (const auto& x : in) {
      if (population_.empty() || x.genotype.size() != box_.size()) continue;
      std::size_t w = 0;
      for (std::size_t i = 1; i < population_.size(); i++)
        if (!heuresis::search::better(population_[i].evaluation, population_[w].evaluation)) w = i;
      if (heuresis::search::better(x.evaluation, population_[w].evaluation)) population_[w] = x;
    }
  }
  std::vector<Individual> emigrants(std::size_t k) const {
    std::vector<Individual> sorted = population_;
    std::stable_sort(sorted.begin(), sorted.end(), [](const Individual& a, const Individual& b) { return heuresis::search::better(a.evaluation, b.evaluation); });
    sorted.resize(std::min(k, sorted.size()));
    return sorted;
  }

  const std::vector<Individual>& population() const { return population_; }
  std::uint64_t generation() const { return generation_; }

 private:
  std::size_t best_index() const {
    std::size_t b = 0;
    for (std::size_t i = 1; i < population_.size(); i++)
      if (heuresis::search::better(population_[i].evaluation, population_[b].evaluation)) b = i;
    return b;
  }

  void open() {
    const std::size_t NP = options_.population;
    std::vector<heuresis::search::Candidate<Real>> c(NP);
    if (population_.empty()) {
      for (std::size_t i = 0; i < NP; i++) {
        calx::random::Stream s = draws_(init_purpose, 0, i);
        c[i] = {heuresis::search::make_ticket(draws_.island(), serial_++), heuresis::variation::uniform_point(box_, s), {init_purpose, {}}};
      }
    } else {
      const std::size_t best = best_index();
      const double F = options_.F;
      for (std::size_t i = 0; i < NP; i++) {
        calx::random::Stream s = draws_(trial_purpose, generation_, i);
        const Real& x = population_[i].genotype;
        Real v(x.size());
        std::vector<std::size_t> used;
        auto at = [&](std::size_t k) -> const Real& { return population_[k].genotype; };
        switch (options_.mutation) {
          case DeMutation::rand_1: {
            used = detail::distinct(NP, 3, {i}, s);
            for (std::size_t j = 0; j < v.size(); j++) v[j] = at(used[0])[j] + F * (at(used[1])[j] - at(used[2])[j]);
            break;
          }
          case DeMutation::best_1: {
            auto r = detail::distinct(NP, 2, {i, best}, s);
            used = {best, r[0], r[1]};
            for (std::size_t j = 0; j < v.size(); j++) v[j] = at(best)[j] + F * (at(r[0])[j] - at(r[1])[j]);
            break;
          }
          case DeMutation::current_to_best_1: {
            auto r = detail::distinct(NP, 2, {i, best}, s);
            used = {best, r[0], r[1]};
            for (std::size_t j = 0; j < v.size(); j++) v[j] = x[j] + F * (at(best)[j] - x[j]) + F * (at(r[0])[j] - at(r[1])[j]);
            break;
          }
          case DeMutation::rand_2: {
            used = detail::distinct(NP, 5, {i}, s);
            for (std::size_t j = 0; j < v.size(); j++) v[j] = at(used[0])[j] + F * (at(used[1])[j] - at(used[2])[j]) + F * (at(used[3])[j] - at(used[4])[j]);
            break;
          }
        }
        Real u = detail::crossover(x, v, options_.CR, options_.crossover, s);
        detail::bring_back(u, x, box_, options_.midpoint, options_.repair, s);
        heuresis::search::Origin origin{trial_purpose, {population_[i].ticket}};
        for (auto k : used) origin.parents.push_back(population_[k].ticket);
        c[i] = {heuresis::search::make_ticket(draws_.island(), serial_++), std::move(u), std::move(origin)};
      }
    }
    batch_.open(std::move(c));
  }

  void close() {
    auto trials = batch_.individuals();
    batch_.close();
    if (population_.empty()) {
      population_ = std::move(trials);
    } else {
      for (std::size_t i = 0; i < trials.size(); i++)
        if (!heuresis::search::better(population_[i].evaluation, trials[i].evaluation)) population_[i] = std::move(trials[i]);
    }
    generation_++;
  }

  Box box_;
  DeOptions options_;
  heuresis::search::Draws draws_;
  heuresis::search::Batch<Real> batch_;
  std::vector<Individual> population_;
  std::uint64_t serial_ = 0, generation_ = 0;
};

struct ShadeOptions {
  std::size_t population = 100;
  std::size_t memory = 0;  // H; 0: the population size
  double p_max = 0.2;
};

class Shade {
 public:
  using Genotype = Real;
  using Individual = heuresis::search::Individual<Real>;
  static constexpr std::uint32_t init_purpose = heuresis::search::tag("strategies.shade.initialise");
  static constexpr std::uint32_t trial_purpose = heuresis::search::tag("strategies.shade.trial");
  static constexpr std::uint32_t archive_purpose = heuresis::search::tag("strategies.shade.archive");

  Shade(Box box, ShadeOptions options = {}, heuresis::search::Draws draws = heuresis::search::Draws{}) : box_(std::move(box)), options_(options), draws_(draws) {
    box_.validate();
    if (options_.population < 4) throw std::invalid_argument("heuresis::strategies::Shade: a population below 4");
    if (options_.memory == 0) options_.memory = options_.population;
    m_cr_.assign(options_.memory, 0.5);
    m_f_.assign(options_.memory, 0.5);
  }

  std::vector<heuresis::search::Candidate<Real>> ask(std::size_t limit) {
    if (!batch_.active()) open();
    return batch_.take(limit);
  }
  void tell(std::span<const heuresis::search::Told> told) {
    for (const auto& t : told)
      if (batch_.accept(t)) close();
  }
  bool stopped() const { return false; }

  const std::vector<Individual>& population() const { return population_; }
  const std::vector<double>& memory_cr() const { return m_cr_; }
  const std::vector<double>& memory_f() const { return m_f_; }
  std::size_t archive_size() const { return archive_.size(); }

 private:
  struct Note {
    double cr = 0, f = 0;
  };

  void open() {
    const std::size_t NP = options_.population, n = box_.size();
    std::vector<heuresis::search::Candidate<Real>> c(NP);
    std::vector<Note> notes(NP);
    if (population_.empty()) {
      for (std::size_t i = 0; i < NP; i++) {
        calx::random::Stream s = draws_(init_purpose, 0, i);
        c[i] = {heuresis::search::make_ticket(draws_.island(), serial_++), heuresis::variation::uniform_point(box_, s), {init_purpose, {}}};
      }
    } else {
      std::vector<std::size_t> order(NP);
      for (std::size_t i = 0; i < NP; i++) order[i] = i;
      std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return heuresis::search::better(population_[a].evaluation, population_[b].evaluation); });
      const double p_min = 2.0 / static_cast<double>(NP);
      for (std::size_t i = 0; i < NP; i++) {
        calx::random::Stream s = draws_(trial_purpose, generation_, i);
        const std::size_t r = s.below(options_.memory);
        double cr = std::clamp(m_cr_[r] + 0.1 * s.normal(), 0.0, 1.0);
        double f;
        do f = m_f_[r] + 0.1 * std::tan(std::numbers::pi * (s.uniform() - 0.5));
        while (!(f > 0));
        f = std::min(f, 1.0);
        const double p = s.uniform(p_min, std::max(p_min, options_.p_max));
        const std::size_t top = std::max<std::size_t>(1, static_cast<std::size_t>(std::ceil(p * static_cast<double>(NP))));
        const std::size_t pbest = order[s.below(top)];
        std::size_t r1;
        do r1 = s.below(NP);
        while (r1 == i);
        const std::size_t pool = NP + archive_.size();
        std::size_t r2;
        do r2 = s.below(pool);
        while (r2 == i || r2 == r1);
        const Real& x = population_[i].genotype;
        const Real& xp = population_[pbest].genotype;
        const Real& x1 = population_[r1].genotype;
        const Real& x2 = r2 < NP ? population_[r2].genotype : archive_[r2 - NP];
        Real v(n);
        for (std::size_t j = 0; j < n; j++) v[j] = x[j] + f * (xp[j] - x[j]) + f * (x1[j] - x2[j]);
        Real u = detail::crossover(x, v, cr, DeCrossover::binomial, s);
        detail::bring_back(u, x, box_, true, Repair::clip, s);
        heuresis::search::Origin origin{trial_purpose, {population_[i].ticket, population_[pbest].ticket, population_[r1].ticket}};
        if (r2 < NP) origin.parents.push_back(population_[r2].ticket);
        c[i] = {heuresis::search::make_ticket(draws_.island(), serial_++), std::move(u), std::move(origin)};
        notes[i] = {cr, f};
      }
    }
    batch_.open(std::move(c), std::move(notes));
  }

  void close() {
    auto trials = batch_.individuals();
    std::vector<Note> notes(trials.size());
    for (std::size_t i = 0; i < trials.size(); i++) notes[i] = batch_.note(i);
    batch_.close();
    if (population_.empty()) {
      population_ = std::move(trials);
      generation_++;
      return;
    }
    std::vector<double> s_cr, s_f, w;
    for (std::size_t i = 0; i < trials.size(); i++) {
      auto& target = population_[i];
      if (heuresis::search::better(target.evaluation, trials[i].evaluation)) continue;
      if (heuresis::search::better(trials[i].evaluation, target.evaluation)) {
        s_cr.push_back(notes[i].cr);
        s_f.push_back(notes[i].f);
        const double d = std::fabs(target.evaluation.value() - trials[i].evaluation.value());
        w.push_back(std::isfinite(d) ? d : 1.0);
        archive_.push_back(target.genotype);
      }
      target = std::move(trials[i]);
    }
    calx::random::Stream s = draws_(archive_purpose, generation_, 0);
    while (archive_.size() > options_.population) {
      const std::size_t k = s.below(archive_.size());
      archive_[k] = std::move(archive_.back());
      archive_.pop_back();
    }
    if (!s_cr.empty()) {
      double tw = 0, mcr = 0, f2 = 0, f1 = 0;
      for (double v : w) tw += v;
      if (!(tw > 0)) { tw = static_cast<double>(w.size()); std::fill(w.begin(), w.end(), 1.0); }
      for (std::size_t k = 0; k < w.size(); k++) {
        mcr += w[k] / tw * s_cr[k];
        f2 += w[k] / tw * s_f[k] * s_f[k];
        f1 += w[k] / tw * s_f[k];
      }
      m_cr_[slot_] = mcr;
      m_f_[slot_] = f2 / f1;
      slot_ = (slot_ + 1) % options_.memory;
    }
    generation_++;
  }

  Box box_;
  ShadeOptions options_;
  heuresis::search::Draws draws_;
  heuresis::search::Batch<Real, Note> batch_;
  std::vector<Individual> population_;
  std::vector<Real> archive_;
  std::vector<double> m_cr_, m_f_;
  std::size_t slot_ = 0;
  std::uint64_t serial_ = 0, generation_ = 0;
};

}  // namespace heuresis::strategies
