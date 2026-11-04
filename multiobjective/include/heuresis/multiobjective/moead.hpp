// MOEA/D: the front as a family of scalar problems solved together by neighbours (Zhang and Li 2007, "MOEA/D: a
// multiobjective evolutionary algorithm based on decomposition", IEEE Trans. Evol. Comput. 11, 712–731).
//
// N weights w¹, …, wᴺ (decomposition.hpp) define N scalar subproblems g(· | wⁱ, z*), each holding one solution xⁱ.
// The neighbourhood B(i) is the T weights nearest wⁱ (wⁱ itself first); neighbouring subproblems have neighbouring
// optima, so they share parents and solutions. For each i, a child is bred from two parents drawn from B(i); the
// ideal point z* takes its values where it improves; and every j ∈ B(i) whose solution the child improves on
// g(· | wʲ, z*) takes the child (their Algorithm 1, steps 2.1–2.5). At most n_r neighbours are replaced by one
// child, as in MOEA/D-DE (Li and Zhang 2009, IEEE Trans. Evol. Comput. 13, 284–302), to keep diversity; n_r = T
// gives the original.
//
// Generational under the ask–tell protocol: the N children of a generation are bred from the solutions as they
// stood at its start, then applied in ticket order (subproblem order), each updating z* and then its neighbours, as
// the original does one child at a time. This is the form of MOEA/D whose evaluations can run in parallel; its
// states do not depend on the order in which evaluations return. Comparisons are Deb's: a feasible child replaces
// an infeasible solution, two infeasible ones compare by violation, two feasible by g; a failure replaces nothing.

#pragma once

#include <heuresis/multiobjective/decomposition.hpp>
#include <heuresis/multiobjective/dominance.hpp>
#include <heuresis/multiobjective/kernels.hpp>
#include <heuresis/search/draws.hpp>

#include <algorithm>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

namespace heuresis::multiobjective {

template <class G>
class Moead {
 public:
  using Genotype = G;
  using Individual = heuresis::search::Individual<G>;
  using Candidate = heuresis::search::Candidate<G>;

  struct Options {
    std::size_t neighbours = 20;     // T
    std::size_t replacements = 0;    // n_r; 0 means T
    Scalarisation scalarisation = Scalarisation::tchebycheff;
    double theta = 5;                // PBI's penalty
  };

  Moead(Kernels<G> kernels, std::vector<Point> weights, Options options, heuresis::search::Draws draws = heuresis::search::Draws{})
      : kernels_(std::move(kernels)), weights_(std::move(weights)), options_(options), draws_(draws) {
    const std::size_t n = weights_.size();
    if (n < 2) throw std::invalid_argument("heuresis::multiobjective::Moead: at least two weights");
    options_.neighbours = std::clamp<std::size_t>(options_.neighbours, 2, n);
    if (options_.replacements == 0) options_.replacements = options_.neighbours;
    neighbours_.resize(n);
    for (std::size_t i = 0; i < n; i++) {
      std::vector<std::size_t> o(n);
      std::iota(o.begin(), o.end(), std::size_t{0});
      std::vector<double> d(n);
      for (std::size_t j = 0; j < n; j++) {
        double s = 0;
        for (std::size_t k = 0; k < weights_[i].size(); k++) s += (weights_[i][k] - weights_[j][k]) * (weights_[i][k] - weights_[j][k]);
        d[j] = s;
      }
      std::stable_sort(o.begin(), o.end(), [&](std::size_t a, std::size_t b) { return d[a] < d[b]; });
      o.resize(options_.neighbours);
      neighbours_[i] = std::move(o);
    }
  }

  std::vector<Candidate> ask(std::size_t limit) {
    if (!batch_.active()) open();
    return batch_.take(limit);
  }

  void tell(std::span<const heuresis::search::Told> told) {
    for (const auto& t : told)
      if (batch_.accept(t)) {
        auto children = batch_.individuals();
        batch_.close();
        if (solutions_.empty()) {
          solutions_ = std::move(children);
          for (const auto& x : solutions_) improve_ideal(x.evaluation);
        } else {
          for (std::size_t i = 0; i < children.size(); i++) apply(i, children[i]);
        }
        generation_++;
      }
  }

  bool stopped() const { return false; }

  const std::vector<Individual>& solutions() const { return solutions_; }
  const Point& ideal() const { return ideal_; }
  std::uint64_t generation() const { return generation_; }
  std::vector<Point> front() const {
    std::vector<Evaluation> e;
    for (const auto& x : solutions_) e.push_back(x.evaluation);
    const auto f = non_dominated_sort(e);
    std::vector<Point> out;
    if (!f.fronts.empty())
      for (auto i : f.fronts[0])
        if (!e[i].failed()) out.push_back(e[i].objectives);
    return out;
  }

  static constexpr std::uint32_t sample_tag = heuresis::search::tag("multiobjective.moead.sample");
  static constexpr std::uint32_t breed_tag = heuresis::search::tag("multiobjective.moead.breed");

 private:
  void open() {
    std::vector<Candidate> c;
    for (std::size_t i = 0; i < weights_.size(); i++) {
      if (solutions_.empty()) {
        calx::random::Stream s = draws_(sample_tag, generation_, i);
        c.push_back({heuresis::search::make_ticket(draws_.island(), serial_++), kernels_.initialise(s), {sample_tag, {}}});
      } else {
        calx::random::Stream s = draws_(breed_tag, generation_, i);
        const auto& b = neighbours_[i];
        const std::size_t k = b[s.below(b.size())], l = b[s.below(b.size())];
        auto children = kernels_.crossover(solutions_[k].genotype, solutions_[l].genotype, s);
        c.push_back({heuresis::search::make_ticket(draws_.island(), serial_++), kernels_.mutate(children.first, s), {breed_tag, {solutions_[k].ticket, solutions_[l].ticket}}});
      }
    }
    batch_.open(std::move(c));
  }

  void improve_ideal(const Evaluation& e) {
    if (e.failed()) return;
    if (ideal_.empty()) ideal_.assign(e.objectives.size(), std::numeric_limits<double>::infinity());
    for (std::size_t k = 0; k < ideal_.size(); k++) ideal_[k] = std::min(ideal_[k], e.objectives[k]);
  }

  // Whether a is better than b on subproblem j.
  bool better_on(const Evaluation& a, const Evaluation& b, std::size_t j) const {
    if (a.failed() || b.failed()) return !a.failed() && b.failed();
    const double va = a.violation(), vb = b.violation();
    if (va > 0 || vb > 0) return va < vb;
    return scalarise(a.objectives, weights_[j], ideal_, options_.scalarisation, options_.theta) <= scalarise(b.objectives, weights_[j], ideal_, options_.scalarisation, options_.theta);
  }

  void apply(std::size_t i, const Individual& child) {
    if (child.evaluation.failed()) return;
    improve_ideal(child.evaluation);
    std::size_t replaced = 0;
    for (std::size_t j : neighbours_[i]) {
      if (replaced >= options_.replacements) break;
      if (better_on(child.evaluation, solutions_[j].evaluation, j)) {
        solutions_[j] = child;
        replaced++;
      }
    }
  }

  Kernels<G> kernels_;
  std::vector<Point> weights_;
  Options options_;
  heuresis::search::Draws draws_;
  std::vector<std::vector<std::size_t>> neighbours_;
  heuresis::search::Batch<G> batch_;
  std::vector<Individual> solutions_;
  Point ideal_;
  std::uint64_t generation_ = 0, serial_ = 0;
};

}  // namespace heuresis::multiobjective
