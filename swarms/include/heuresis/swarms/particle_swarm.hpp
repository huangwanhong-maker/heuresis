// Particle swarm optimisation.
//
// A swarm of particles flies through a box, each with a position x, a velocity v and the best point p it has
// found; each is attracted towards p and towards the best l found by its informants (Kennedy and Eberhart 1995,
// "Particle swarm optimization", Proc. IEEE ICNN 4, 1942–1948). Three rules of motion:
//
// - inertia (Shi and Eberhart 1998, "A modified particle swarm optimizer", IEEE CEC, 69–73):
//     v ← w v + c₁ r₁ ⊗ (p − x) + c₂ r₂ ⊗ (l − x),  x ← x + v,  r₁, r₂ uniform on [0, 1]ⁿ per coordinate;
// - constriction (Clerc and Kennedy 2002, "The particle swarm — explosion, stability, and convergence in a
//   multidimensional complex space", IEEE Trans. Evol. Comput. 6, 58–73):
//     v ← χ (v + φ₁ r₁ ⊗ (p − x) + φ₂ r₂ ⊗ (l − x)),  χ = 2/|2 − φ − √(φ² − 4φ)|,  φ = φ₁ + φ₂ > 4,
//   χ = 0.72984… at the customary φ = 4.1, the same swarm as inertia with w = χ, c = χφ/2 = 1.49618;
// - the Standard PSO 2011 (Clerc 2012, "Standard particle swarm optimisation", HAL hal-00764996; Zambrano-Bigiarini,
//   Clerc and Rojas 2013, "Standard particle swarm optimisation 2011 at CEC-2013: a baseline for future PSO
//   improvements", IEEE CEC, 2337–2344, eqs. 3–6): with p′ = x + c(p − x) and l′ = x + c(l − x), the centre
//   G = (x + p′ + l′)/3 (G = (x + p′)/2 when l = p); x′ uniform in the ball of centre G and radius ‖G − x‖;
//   v ← w v + x′ − x, x ← x + v; w = 1/(2 ln 2), c = ½ + ln 2.
//
// The per-coordinate factors r₁, r₂ of the first two rules make them depend on the coordinate system: rotating
// the problem changes the swarm's behaviour. SPSO 2011's move draws a direction uniformly on the sphere and so
// commutes with every rotation and translation; `spso2011_move` takes the direction as an argument, so that the
// checks hold this exactly.
//
// Stability (checked). About a fixed attractor, the second moments (E|x − p|², E|v|², E(x − p)·v) of an SPSO
// 2011 particle follow a linear map, since x′ is uniform in the ball (E|u|² = n/(n + 2) for u uniform in the unit
// ball). At the standard w and c its spectral radius is 0.98 in two dimensions, 1.04 in three and 1.18 in ten:
// from three dimensions on a particle does not settle about its attractor but flies at the scale of the box,
// and the swarm progresses only through its personal bests. The checks hold the simulated growth to the
// spectral radius. (Bonyadi and Michalewicz 2014, "SPSO 2011: analysis of stability; local convergence; and
// rotation sensitivity", GECCO, analysed its stability.) The inertia and constriction rules converge on the
// sphere in ten dimensions.
//
// Informants: the whole swarm, a ring (each particle and its two neighbours), or SPSO 2011's adaptive random
// topology, in which each particle informs itself and K = 3 others drawn at random, redrawn after every
// generation that did not improve the swarm's best. Positions leaving the box are set on the bound and that
// coordinate's velocity reversed and halved (SPSO 2011's confinement). The swarm is generational: its first
// generation is uniform in the box with velocities U(l − x, u − x), each later one is every particle's new
// position; personal bests are replaced only by strictly better points.

#pragma once

#include <heuresis/search/draws.hpp>
#include <heuresis/search/protocol.hpp>
#include <heuresis/variation/real.hpp>

#include <cmath>
#include <numbers>
#include <span>
#include <stdexcept>
#include <vector>

namespace heuresis::swarms {

using heuresis::variation::Box;
using heuresis::variation::Real;

// Clerc and Kennedy's constriction coefficient for φ > 4.
inline double constriction(double phi) {
  if (!(phi > 4)) throw std::invalid_argument("heuresis::swarms::constriction: φ must exceed 4");
  return 2 / std::fabs(2 - phi - std::sqrt(phi * phi - 4 * phi));
}

struct Move {
  Real x, v;
};

// SPSO 2011's move, given a unit direction and the fraction ρ ∈ [0, 1] of the ball's radius (uniform in the ball
// when ρ = U^{1/n}). same_best: l is p.
inline Move spso2011_move(std::span<const double> x, std::span<const double> v, std::span<const double> p, std::span<const double> l, bool same_best, double w, double c,
                          std::span<const double> direction, double fraction) {
  const std::size_t n = x.size();
  Real G(n);
  for (std::size_t i = 0; i < n; i++) G[i] = same_best ? x[i] + c * (p[i] - x[i]) / 2 : x[i] + c * (p[i] + l[i] - 2 * x[i]) / 3;
  double r = 0;
  for (std::size_t i = 0; i < n; i++) r += (G[i] - x[i]) * (G[i] - x[i]);
  r = std::sqrt(r) * fraction;
  Move m{Real(n), Real(n)};
  for (std::size_t i = 0; i < n; i++) {
    const double xp = G[i] + r * direction[i];
    m.v[i] = w * v[i] + xp - x[i];
    m.x[i] = x[i] + m.v[i];
  }
  return m;
}

// The inertia rule's move, given the per-coordinate factors r₁ and r₂ (the constriction rule is the same with
// w = χ, c₁ = χφ₁, c₂ = χφ₂).
inline Move inertia_move(std::span<const double> x, std::span<const double> v, std::span<const double> p, std::span<const double> l, double w, double c1, double c2, std::span<const double> r1,
                         std::span<const double> r2) {
  const std::size_t n = x.size();
  Move m{Real(n), Real(n)};
  for (std::size_t i = 0; i < n; i++) {
    m.v[i] = w * v[i] + c1 * r1[i] * (p[i] - x[i]) + c2 * r2[i] * (l[i] - x[i]);
    m.x[i] = x[i] + m.v[i];
  }
  return m;
}

enum class PsoRule : std::uint8_t { standard_2011, inertia, constriction };
enum class PsoTopology : std::uint8_t { adaptive_random, ring, global };

struct PsoOptions {
  std::size_t swarm = 40;
  PsoRule rule = PsoRule::standard_2011;
  PsoTopology topology = PsoTopology::adaptive_random;
  std::size_t informants = 3;                          // K, for the adaptive random topology
  double w = 1 / (2 * std::numbers::ln2), c = 0.5 + std::numbers::ln2;  // SPSO 2011
  double inertia = 0.7298, c1 = 1.49618, c2 = 1.49618;  // the inertia rule
  double phi = 4.1;                                     // the constriction rule, φ₁ = φ₂ = φ/2
};

class ParticleSwarm {
 public:
  using Genotype = Real;
  using Individual = heuresis::search::Individual<Real>;
  static constexpr std::uint32_t init_purpose = heuresis::search::tag("swarms.pso.initialise");
  static constexpr std::uint32_t move_purpose = heuresis::search::tag("swarms.pso.move");
  static constexpr std::uint32_t topology_purpose = heuresis::search::tag("swarms.pso.topology");

  ParticleSwarm(Box box, PsoOptions options = {}, heuresis::search::Draws draws = heuresis::search::Draws{}) : box_(std::move(box)), options_(options), draws_(draws) {
    box_.validate();
    if (options_.swarm < 2) throw std::invalid_argument("heuresis::swarms::ParticleSwarm: a swarm below 2");
    redraw_topology();
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

  // An evaluated point from outside becomes the personal best of the particle whose best is worst, if better.
  void inject(std::span<const Individual> in) {
    for (const auto& x : in) {
      if (best_.empty() || x.genotype.size() != box_.size()) continue;
      std::size_t w = 0;
      for (std::size_t i = 1; i < best_.size(); i++)
        if (!heuresis::search::better(best_[i].evaluation, best_[w].evaluation)) w = i;
      if (heuresis::search::better(x.evaluation, best_[w].evaluation)) best_[w] = x;
    }
  }
  std::vector<Individual> emigrants(std::size_t k) const {
    std::vector<Individual> sorted = best_;
    std::stable_sort(sorted.begin(), sorted.end(), [](const Individual& a, const Individual& b) { return heuresis::search::better(a.evaluation, b.evaluation); });
    sorted.resize(std::min(k, sorted.size()));
    return sorted;
  }

  const std::vector<Individual>& personal_bests() const { return best_; }
  const std::vector<Real>& positions() const { return x_; }
  const std::vector<Real>& velocities() const { return v_; }
  std::uint64_t generation() const { return generation_; }

 private:
  void redraw_topology() {
    const std::size_t S = options_.swarm;
    neighbours_.assign(S, {});
    for (std::size_t i = 0; i < S; i++) neighbours_[i].push_back(i);
    switch (options_.topology) {
      case PsoTopology::global:
        for (std::size_t i = 0; i < S; i++)
          for (std::size_t j = 0; j < S; j++)
            if (i != j) neighbours_[i].push_back(j);
        break;
      case PsoTopology::ring:
        for (std::size_t i = 0; i < S; i++) {
          neighbours_[i].push_back((i + S - 1) % S);
          neighbours_[i].push_back((i + 1) % S);
        }
        break;
      case PsoTopology::adaptive_random:
        for (std::size_t i = 0; i < S; i++) {
          calx::random::Stream s = draws_(topology_purpose, redraws_, i);
          for (std::size_t k = 0; k < options_.informants; k++) neighbours_[s.below(S)].push_back(i);  // i informs the drawn particle
        }
        break;
    }
    redraws_++;
  }

  std::size_t local_best(std::size_t i) const {
    std::size_t b = i;
    for (auto j : neighbours_[i])
      if (heuresis::search::better(best_[j].evaluation, best_[b].evaluation)) b = j;
    return b;
  }

  void open() {
    const std::size_t S = options_.swarm, n = box_.size();
    std::vector<heuresis::search::Candidate<Real>> c(S);
    if (x_.empty()) {
      x_.resize(S);
      v_.resize(S);
      for (std::size_t i = 0; i < S; i++) {
        calx::random::Stream s = draws_(init_purpose, 0, i);
        x_[i] = heuresis::variation::uniform_point(box_, s);
        v_[i].resize(n);
        for (std::size_t d = 0; d < n; d++) v_[i][d] = s.uniform(box_.lower[d] - x_[i][d], box_.upper[d] - x_[i][d]);
        c[i] = {heuresis::search::make_ticket(draws_.island(), serial_++), x_[i], {init_purpose, {}}};
      }
    } else {
      for (std::size_t i = 0; i < S; i++) {
        calx::random::Stream s = draws_(move_purpose, generation_, i);
        const std::size_t l = local_best(i);
        const Real& p = best_[i].genotype;
        const Real& lb = best_[l].genotype;
        Move m;
        if (options_.rule == PsoRule::standard_2011) {
          Real dir(n);
          double norm = 0;
          for (auto& d : dir) { d = s.normal(); norm += d * d; }
          norm = std::sqrt(norm);
          for (auto& d : dir) d /= norm;
          const double frac = std::pow(s.uniform(), 1 / static_cast<double>(n));
          m = spso2011_move(x_[i], v_[i], p, lb, l == i, options_.w, options_.c, dir, frac);
        } else {
          Real r1(n), r2(n);
          for (std::size_t d = 0; d < n; d++) { r1[d] = s.uniform(); r2[d] = s.uniform(); }
          if (options_.rule == PsoRule::inertia) m = inertia_move(x_[i], v_[i], p, lb, options_.inertia, options_.c1, options_.c2, r1, r2);
          else {
            const double chi = constriction(options_.phi);
            m = inertia_move(x_[i], v_[i], p, lb, chi, chi * options_.phi / 2, chi * options_.phi / 2, r1, r2);
          }
        }
        for (std::size_t d = 0; d < n; d++) {
          if (m.x[d] < box_.lower[d]) { m.x[d] = box_.lower[d]; m.v[d] = -0.5 * m.v[d]; }
          else if (m.x[d] > box_.upper[d]) { m.x[d] = box_.upper[d]; m.v[d] = -0.5 * m.v[d]; }
        }
        x_[i] = std::move(m.x);
        v_[i] = std::move(m.v);
        c[i] = {heuresis::search::make_ticket(draws_.island(), serial_++), x_[i], {move_purpose, {best_[i].ticket, best_[l].ticket}}};
      }
    }
    batch_.open(std::move(c));
  }

  void close() {
    auto told = batch_.individuals();
    batch_.close();
    const bool first = best_.empty();
    const double before = first ? INFINITY : swarm_best();
    if (first) best_ = std::move(told);
    else
      for (std::size_t i = 0; i < told.size(); i++)
        if (heuresis::search::better(told[i].evaluation, best_[i].evaluation)) best_[i] = std::move(told[i]);
    if (options_.topology == PsoTopology::adaptive_random && !first && !(swarm_best() < before)) redraw_topology();
    generation_++;
  }

  double swarm_best() const {
    double b = INFINITY;
    for (const auto& x : best_) b = std::min(b, x.evaluation.feasible() ? x.evaluation.value() : INFINITY);
    return b;
  }

  Box box_;
  PsoOptions options_;
  heuresis::search::Draws draws_;
  heuresis::search::Batch<Real> batch_;
  std::vector<Real> x_, v_;
  std::vector<Individual> best_;
  std::vector<std::vector<std::size_t>> neighbours_;
  std::uint64_t serial_ = 0, generation_ = 0, redraws_ = 0;
};

}  // namespace heuresis::swarms
