// The quantum-inspired evolutionary algorithm, and why it is a search distribution.
//
// Han and Kim's algorithm (2002, "Quantum-inspired evolutionary algorithm for a class of combinatorial
// optimization", IEEE Trans. Evol. Comput. 6, 580–593) keeps each individual as a string of Q-bits, qubits
// αᵢ|0⟩ + βᵢ|1⟩ with real amplitudes αᵢ = cos φᵢ, βᵢ = sin φᵢ. A generation observes every individual once,
// bit i reading 1 with probability |βᵢ|²; each individual remembers the best string it has observed, bᵢ; a rotation
// gate turns φᵢ by an angle read from a lookup table indexed by (xᵢ, bᵢ, whether x is at least as good as b), in the
// direction that moves the probability of reading 1 towards the table's sign; every few generations the best string
// of all replaces every individual's (global migration). The table's default is the knapsack setting of their
// Table I: rotate by 0.01π towards bᵢ when xᵢ ≠ bᵢ and x is worse than b, otherwise not at all.
//
// Theorem (the observation law is a product law). The register of one individual is the product state
// ⊗ᵢ (cos φᵢ|0⟩ + sin φᵢ|1⟩), whose amplitude on a basis string x is Πᵢ (xᵢ ? sin φᵢ : cos φᵢ). Measured in the
// computational basis it gives x with probability Πᵢ pᵢ^{xᵢ}(1 − pᵢ)^{1−xᵢ}, pᵢ = sin² φᵢ: the product Bernoulli law
// (distributions/bernoulli.hpp). No entangling gate is ever applied, so the algorithm never leaves the product
// family, and it is an estimation-of-distribution algorithm of that family whose rule rotates angles instead of
// moving frequencies: QIEA's "quantum" register needs no quantum device, and its observations are draws a CPU makes
// exactly. (A register in an entangled state is not a product law: the Bell state's law has
// P(00)P(11) − P(01)P(10) = ¼, while every product law has 0; the checks verify both.) The algorithm here observes
// bit i by the same comparison U < pᵢ, from the same keyed stream, as the product-Bernoulli family's reference
// sampler, so its observations are that family's draws exactly.

#pragma once

#include <heuresis/distributions/bernoulli.hpp>
#include <heuresis/search/draws.hpp>
#include <heuresis/search/protocol.hpp>

#include <array>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <vector>

namespace heuresis::quantum {

using heuresis::variation::Bits;

// The probabilities of reading 1, sin² φᵢ.
inline std::vector<double> observation_probabilities(const std::vector<double>& angles) {
  std::vector<double> p(angles.size());
  for (std::size_t i = 0; i < p.size(); i++) p[i] = std::sin(angles[i]) * std::sin(angles[i]);
  return p;
}

struct QuantumInspiredOptions {
  std::size_t individuals = 10;
  // Signed rotations towards |1⟩, indexed by 4·xᵢ + 2·bᵢ + (x at least as good as b), in radians.
  std::array<double, 8> table{0, 0, 0.01 * std::numbers::pi, 0, -0.01 * std::numbers::pi, 0, 0, 0};
  std::size_t global_period = 100;  // generations between global migrations; 0: never
};

class QuantumInspired {
 public:
  using Genotype = Bits;

  QuantumInspired(std::size_t n, QuantumInspiredOptions options = {}, heuresis::search::Draws draws = heuresis::search::Draws{})
      : n_(n), options_(options), draws_(draws), angles_(options.individuals, std::vector<double>(n, std::numbers::pi / 4)) {
    if (options_.individuals == 0 || n == 0) throw std::invalid_argument("heuresis::quantum::QuantumInspired: at least one individual of at least one Q-bit");
  }

  std::vector<heuresis::search::Candidate<Bits>> ask(std::size_t limit) {
    if (!batch_.active()) {
      std::vector<heuresis::search::Candidate<Bits>> g;
      for (std::size_t k = 0; k < options_.individuals; k++) {
        calx::random::Stream s = draws_(observe_tag, generation_, k);
        Bits x(n_);
        for (std::size_t i = 0; i < n_; i++) {
          const double beta = std::sin(angles_[k][i]);
          x[i] = s.uniform() < beta * beta ? 1 : 0;  // the measurement of Q-bit i
        }
        g.push_back({heuresis::search::make_ticket(draws_.island(), serial_++), std::move(x), heuresis::search::Origin{observe_tag, {}}});
      }
      batch_.open(std::move(g));
    }
    return batch_.take(limit);
  }

  void tell(std::span<const heuresis::search::Told> told) {
    for (const auto& t : told) {
      if (!batch_.accept(t)) continue;
      const bool first = best_.empty();
      if (first) best_.resize(options_.individuals);
      for (std::size_t k = 0; k < options_.individuals; k++) {
        const auto& x = batch_.candidate(k).genotype;
        const auto& e = batch_.evaluation(k);
        if (!first) rotate(k, x, !heuresis::search::better(best_[k].evaluation, e));
        if (first || heuresis::search::better(e, best_[k].evaluation)) best_[k] = {x, e};
      }
      if (options_.global_period && (generation_ + 1) % options_.global_period == 0) {
        std::size_t g = 0;
        for (std::size_t k = 1; k < best_.size(); k++)
          if (heuresis::search::better(best_[k].evaluation, best_[g].evaluation)) g = k;
        for (auto& b : best_) b = best_[g];
      }
      batch_.close();
      generation_++;
    }
  }

  bool stopped() const { return false; }

  const std::vector<std::vector<double>>& angles() const { return angles_; }
  std::uint64_t generation() const { return generation_; }
  static constexpr std::uint32_t observe_tag = heuresis::search::tag("quantum.qiea.observe");

 private:
  // The rotation gate on individual k towards its best string, by the table.
  void rotate(std::size_t k, const Bits& x, bool at_least_as_good) {
    const Bits& b = best_[k].genotype;
    for (std::size_t i = 0; i < n_; i++) {
      const double towards_one = options_.table[4 * x[i] + 2 * b[i] + (at_least_as_good ? 1 : 0)];
      if (towards_one == 0) continue;
      double& phi = angles_[k][i];
      // d sin²φ/dφ = sin 2φ: turning φ by +δ raises P(1) where sin 2φ > 0.
      phi += towards_one * (std::sin(2 * phi) >= 0 ? 1.0 : -1.0);
    }
  }

  struct Best {
    Bits genotype;
    heuresis::search::Evaluation evaluation;
  };

  std::size_t n_;
  QuantumInspiredOptions options_;
  heuresis::search::Draws draws_;
  std::vector<std::vector<double>> angles_;
  std::vector<Best> best_;
  heuresis::search::Batch<Bits> batch_;
  std::uint64_t generation_ = 0, serial_ = 0;
};

}  // namespace heuresis::quantum
