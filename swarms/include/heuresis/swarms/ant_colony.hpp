// Ant colony optimisation on cyclic sequences.
//
// Ants build permutations step by step on a graph of n elements, each choosing the next element j after i among
// those not yet used with probability ∝ τ_ij^α η_ij^β: τ the colony's pheromone trails, learnt; η the problem's
// heuristic information, given (the domain's knowledge: for a tour, the inverse of the distance). After a
// generation the trails evaporate, τ ← (1 − ρ)τ, and are reinforced along good permutations (Dorigo, Maniezzo and
// Colorni 1996, "Ant system: optimization by a colony of cooperating agents", IEEE Trans. Syst. Man Cybern. B 26,
// 29–41). The permutation is read as a cycle, its last element followed by its first, and the trails are
// symmetric (τ_ij = τ_ji), as for tours.
//
// - Ant System: every ant deposits 1/f of its permutation on each of its edges.
// - MAX–MIN Ant System (Stützle and Hoos 2000, "MAX–MIN Ant System", Future Gener. Comput. Syst. 16, 889–914,
//   §3, eqs. 6–11): only the generation's best ant deposits (or the best so far, if asked), the trails are kept
//   in [τ_min, τ_max] with τ_max = 1/(ρ f_best) and τ_min = τ_max (1 − p_best^{1/n})/((n/2 − 1) p_best^{1/n}),
//   f_best the best so far, and they start at τ_max, so that exploration is high at first and never vanishes.
// Objectives must be positive (lengths, costs); a failed evaluation deposits nothing.
//
// It is generational: a generation is m ants, ant k's choices drawn from the stream keyed by (generation, k); the
// trails change once per generation, after every ant is told, so the colony does not depend on the order in
// which the evaluations return. Since the trails are fixed within a generation, the weights τ_ij^α η_ij^β are
// computed once per generation (n² powers) rather than at every step of every ant (about m n²/2), with the same
// rounding, so the colony is unchanged bit for bit.

#pragma once

#include <heuresis/search/draws.hpp>
#include <heuresis/search/protocol.hpp>
#include <heuresis/variation/permutation.hpp>

#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <vector>

namespace heuresis::swarms {

using heuresis::variation::Permutation;

enum class AcoVariant : std::uint8_t { ant_system, max_min };

struct AcoOptions {
  AcoVariant variant = AcoVariant::max_min;
  std::size_t ants = 0;  // 0: n
  double alpha = 1, beta = 2;
  double rho = 0;  // 0: 0.5 for the Ant System, 0.02 for MAX–MIN
  double p_best = 0.05;
  bool global_best = false;  // MAX–MIN: the best so far deposits instead of the generation's best
};

class AntColony {
 public:
  using Genotype = Permutation;
  using Individual = heuresis::search::Individual<Permutation>;
  static constexpr std::uint32_t purpose = heuresis::search::tag("swarms.aco.construct");

  // heuristic: the n × n matrix η, row by row, η_ij ≥ 0.
  AntColony(std::size_t n, std::vector<double> heuristic, AcoOptions options = {}, heuresis::search::Draws draws = heuresis::search::Draws{})
      : n_(n), eta_(std::move(heuristic)), options_(options), draws_(draws), tau_(n * n, 1.0) {
    if (n < 3 || eta_.size() != n * n) throw std::invalid_argument("heuresis::swarms::AntColony: n ≥ 3 and an n × n heuristic");
    for (double e : eta_)
      if (!(e >= 0) || !std::isfinite(e)) throw std::invalid_argument("heuresis::swarms::AntColony: the heuristic must be finite and nonnegative");
    if (options_.ants == 0) options_.ants = n;
    if (options_.rho == 0) options_.rho = options_.variant == AcoVariant::ant_system ? 0.5 : 0.02;
    weight_.resize(n * n);
    for (std::size_t k = 0; k < n * n; k++) weight_[k] = std::pow(eta_[k], options_.beta);
    choice_.resize(n * n);
  }

  std::vector<heuresis::search::Candidate<Permutation>> ask(std::size_t limit) {
    if (!batch_.active()) open();
    return batch_.take(limit);
  }
  void tell(std::span<const heuresis::search::Told> told) {
    for (const auto& t : told)
      if (batch_.accept(t)) close();
  }
  bool stopped() const { return false; }

  double trail(std::size_t i, std::size_t j) const { return tau_[i * n_ + j]; }
  const std::vector<double>& trails() const { return tau_; }
  double tau_min() const { return tau_min_; }
  double tau_max() const { return tau_max_; }
  const std::optional<Individual>& best() const { return best_; }

 private:
  void open() {
    std::vector<heuresis::search::Candidate<Permutation>> c(options_.ants);
    std::vector<double> w(n_);
    for (std::size_t k = 0; k < n_ * n_; k++) choice_[k] = std::pow(tau_[k], options_.alpha) * weight_[k];
    for (std::size_t k = 0; k < options_.ants; k++) {
      calx::random::Stream s = draws_(purpose, generation_, k);
      Permutation p;
      p.reserve(n_);
      std::vector<char> used(n_, 0);
      std::size_t at = s.below(n_);
      p.push_back(static_cast<std::uint32_t>(at));
      used[at] = 1;
      while (p.size() < n_) {
        double total = 0;
        for (std::size_t j = 0; j < n_; j++) total += (w[j] = used[j] ? 0 : choice_[at * n_ + j]);
        std::size_t next = n_;
        if (total > 0 && std::isfinite(total)) {
          const double u = s.uniform() * total;
          double acc = 0;
          for (std::size_t j = 0; j < n_ && next == n_; j++)
            if (!used[j] && (acc += w[j]) >= u) next = j;
        }
        if (next == n_) {  // no trail or heuristic leads on, or rounding left the pointer past the end: uniform
          std::size_t r = s.below(n_ - p.size());
          for (std::size_t j = 0; j < n_; j++)
            if (!used[j] && r-- == 0) { next = j; break; }
        }
        p.push_back(static_cast<std::uint32_t>(next));
        used[next] = 1;
        at = next;
      }
      c[k] = {heuresis::search::make_ticket(draws_.island(), serial_++), std::move(p), {purpose, {}}};
    }
    batch_.open(std::move(c));
  }

  void deposit(const Permutation& p, double amount) {
    for (std::size_t i = 0; i < n_; i++) {
      const std::size_t a = p[i], b = p[(i + 1) % n_];
      tau_[a * n_ + b] += amount;
      tau_[b * n_ + a] += amount;
    }
  }

  void close() {
    auto ants = batch_.individuals();
    batch_.close();
    std::size_t gen_best = ants.size();
    for (std::size_t k = 0; k < ants.size(); k++) {
      if (!ants[k].evaluation.feasible() || !(ants[k].evaluation.value() > 0)) continue;
      if (gen_best == ants.size() || heuresis::search::better(ants[k].evaluation, ants[gen_best].evaluation)) gen_best = k;
    }
    if (gen_best < ants.size() && (!best_ || heuresis::search::better(ants[gen_best].evaluation, best_->evaluation))) best_ = ants[gen_best];
    const double keep = 1 - options_.rho;
    if (options_.variant == AcoVariant::ant_system) {
      for (auto& t : tau_) t *= keep;
      for (const auto& a : ants)
        if (a.evaluation.feasible() && a.evaluation.value() > 0) deposit(a.genotype, 1 / a.evaluation.value());
    } else if (best_) {
      const double N = static_cast<double>(n_), root = std::pow(options_.p_best, 1 / N);
      tau_max_ = 1 / (options_.rho * best_->evaluation.value());
      tau_min_ = std::min(tau_max_, tau_max_ * (1 - root) / ((N / 2 - 1) * root));
      if (!started_) {
        std::fill(tau_.begin(), tau_.end(), tau_max_);
        started_ = true;
      }
      for (auto& t : tau_) t *= keep;
      const Individual& by = options_.global_best ? *best_ : ants[gen_best];
      deposit(by.genotype, 1 / by.evaluation.value());
      for (auto& t : tau_) t = std::clamp(t, tau_min_, tau_max_);
    }
    generation_++;
  }

  std::size_t n_;
  std::vector<double> eta_, weight_, choice_;  // η, η^β, and τ^α η^β for the generation being built
  AcoOptions options_;
  heuresis::search::Draws draws_;
  std::vector<double> tau_;
  heuresis::search::Batch<Permutation> batch_;
  std::optional<Individual> best_;
  double tau_min_ = 0, tau_max_ = 0;
  bool started_ = false;
  std::uint64_t serial_ = 0, generation_ = 0;
};

}  // namespace heuresis::swarms
