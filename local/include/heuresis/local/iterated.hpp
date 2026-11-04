// Iterated local search: a walk on the local optima.
//
// Climb to a local optimum x*; perturb it by a kick larger than a neighbourhood's step; climb again to y*; keep
// y* as the incumbent when the acceptance criterion says so (better, or not worse); repeat (Lourenço, Martin and
// Stützle 2003, "Iterated local search", in Handbook of Metaheuristics, Kluwer, 321–353). The kick must leave
// the basin of x* without destroying what it holds; for tours the double bridge (Martin, Otto and Felten 1991,
// "Large-step Markov chains for the traveling salesman problem", Complex Systems 5, 299–326) cuts the tour into
// four segments A B C D and rejoins them A C B D, a move no sequence of improving 2-opt moves undoes.
//
// It never stops by itself; a budget ends it. Its best local optimum is `best()`.

#pragma once

#include <heuresis/local/climbing.hpp>

namespace heuresis::local {

enum class Acceptance : std::uint8_t { better, not_worse };

template <class G>
class IteratedLocalSearch {
 public:
  using Genotype = G;
  using Kick = std::function<G(const G&, calx::random::Stream&)>;
  static constexpr std::uint32_t kick_tag = tag("local.iterated.kick");

  IteratedLocalSearch(G start, Neighbourhood<G> neighbourhood, Kick kick, Pivot pivot = Pivot::best, Acceptance acceptance = Acceptance::not_worse, Draws draws = Draws{})
      : climber_(std::move(start), std::move(neighbourhood), pivot, 1, draws), kick_(std::move(kick)), acceptance_(acceptance), draws_(draws) {}

  std::vector<Candidate<G>> ask(std::size_t limit) { return climber_.ask(limit); }

  void tell(std::span<const Told> told) {
    climber_.tell(told);
    if (!climber_.stopped()) return;
    const Individual<G>& y = *climber_.current();
    if (!incumbent_ || better(y.evaluation, incumbent_->evaluation) || (acceptance_ == Acceptance::not_worse && !better(incumbent_->evaluation, y.evaluation))) incumbent_ = y;
    if (!best_ || better(y.evaluation, best_->evaluation)) best_ = y;
    calx::random::Stream s = draws_(kick_tag, kicks_++, 0);
    climber_.restart(kick_(incumbent_->genotype, s), Origin{kick_tag, {incumbent_->ticket}});
  }

  bool stopped() const { return false; }
  const std::optional<Individual<G>>& best() const { return best_; }
  std::uint64_t kicks() const { return kicks_; }

 private:
  HillClimber<G> climber_;
  Kick kick_;
  Acceptance acceptance_;
  Draws draws_;
  std::optional<Individual<G>> incumbent_, best_;
  std::uint64_t kicks_ = 0;
};

// The double bridge: A B C D → A C B D, at three distinct cut points.
inline Permutation double_bridge(const Permutation& p, calx::random::Stream& s) {
  const std::size_t n = p.size();
  if (n < 8) return heuresis::variation::scramble_mutation(p, s);
  std::size_t c[3];
  for (;;) {
    for (auto& v : c) v = 1 + s.below(n - 1);
    std::sort(c, c + 3);
    if (c[0] < c[1] && c[1] < c[2]) break;
  }
  Permutation q;
  q.reserve(n);
  q.insert(q.end(), p.begin(), p.begin() + static_cast<std::ptrdiff_t>(c[0]));
  q.insert(q.end(), p.begin() + static_cast<std::ptrdiff_t>(c[1]), p.begin() + static_cast<std::ptrdiff_t>(c[2]));
  q.insert(q.end(), p.begin() + static_cast<std::ptrdiff_t>(c[0]), p.begin() + static_cast<std::ptrdiff_t>(c[1]));
  q.insert(q.end(), p.begin() + static_cast<std::ptrdiff_t>(c[2]), p.end());
  return q;
}

}  // namespace heuresis::local
