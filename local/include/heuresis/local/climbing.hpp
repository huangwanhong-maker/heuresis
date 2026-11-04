// Hill climbing: moving to an improving neighbour until none improves.
//
// From the current point x, a best-improvement climber evaluates all of N(x) as one generation and moves to the
// best neighbour when it is strictly better; a first-improvement climber scans N(x) in an order drawn afresh at
// each step (keyed by the step), a chunk of neighbours at a time, and moves to the first strictly better one in
// scan order (Hansen and Mladenović 2006, "First vs. best improvement: an empirical study", Discrete Appl. Math.
// 154, 802–817). Either stops at a local optimum, and stopped() says so; restart(x) begins again from another
// point while keeping the run's tickets and draws, which is how iterated local search uses it.
//
// Both are generational in the protocol's sense, so their moves do not depend on the order in which a chunk's
// evaluations return. Randomised local search, which accepts a random neighbour that is not worse, is the
// (1+1) EA with one bit flipped (genetic/elitist.hpp).

#pragma once

#include <heuresis/local/neighbourhood.hpp>
#include <heuresis/search/search.hpp>

#include <numeric>
#include <optional>
#include <span>
#include <vector>

namespace heuresis::local {

using heuresis::search::Candidate;
using heuresis::search::Draws;
using heuresis::search::Individual;
using heuresis::search::Origin;
using heuresis::search::Told;
using heuresis::search::better;
using heuresis::search::make_ticket;
using heuresis::search::tag;

enum class Pivot : std::uint8_t { best, first };

template <class G>
class HillClimber {
 public:
  using Genotype = G;
  static constexpr std::uint32_t start_tag = tag("local.climber.start");
  static constexpr std::uint32_t move_tag = tag("local.climber.move");
  static constexpr std::uint32_t order_tag = tag("local.climber.order");

  HillClimber(G start, Neighbourhood<G> neighbourhood, Pivot pivot = Pivot::best, std::size_t chunk = 1, Draws draws = Draws{})
      : nb_(std::move(neighbourhood)), pivot_(pivot), chunk_(std::max<std::size_t>(1, chunk)), draws_(draws) {
    restart(std::move(start), Origin{start_tag, {}});
  }

  // Begins again from an unevaluated point.
  void restart(G start, Origin origin) {
    if (batch_.active()) throw std::logic_error("heuresis::local::HillClimber: restart while a generation is out");
    start_ = Candidate<G>{0, std::move(start), std::move(origin)};
    current_.reset();
    optimum_ = false;
    scan_.clear();
    restarts_++;
  }

  std::vector<Candidate<G>> ask(std::size_t limit) {
    if (optimum_) return {};
    if (!batch_.active()) {
      std::vector<Candidate<G>> g;
      if (start_) {
        start_->ticket = make_ticket(draws_.island(), serial_++);
        g.push_back(std::move(*start_));
        start_.reset();
      } else if (pivot_ == Pivot::best) {
        const std::size_t m = nb_.size(current_->genotype);
        for (std::size_t k = 0; k < m; k++) g.push_back({make_ticket(draws_.island(), serial_++), nb_.apply(current_->genotype, k), {move_tag, {current_->ticket}}});
      } else {
        if (scan_.empty()) {
          scan_.resize(nb_.size(current_->genotype));
          std::iota(scan_.begin(), scan_.end(), std::size_t{0});
          calx::random::Stream s = draws_(order_tag, step_, restarts_);
          for (std::size_t i = scan_.size(); i > 1; i--) std::swap(scan_[i - 1], scan_[s.below(i)]);
          scanned_ = 0;
        }
        for (std::size_t c = 0; c < chunk_ && scanned_ < scan_.size(); c++, scanned_++)
          g.push_back({make_ticket(draws_.island(), serial_++), nb_.apply(current_->genotype, scan_[scanned_]), {move_tag, {current_->ticket}}});
      }
      if (g.empty()) {  // a point without neighbours is a local optimum
        optimum_ = true;
        return {};
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
      evaluations_ += xs.size();
      if (!current_) {
        current_ = std::move(xs[0]);
        continue;
      }
      std::optional<std::size_t> move;
      for (std::size_t i = 0; i < xs.size(); i++) {
        if (!better(xs[i].evaluation, current_->evaluation)) continue;
        if (pivot_ == Pivot::first) { move = i; break; }
        if (!move || better(xs[i].evaluation, xs[*move].evaluation)) move = i;
      }
      if (move) {
        current_ = std::move(xs[*move]);
        step_++;
        scan_.clear();
      } else if (pivot_ == Pivot::best || scanned_ == scan_.size()) {
        optimum_ = true;
      }
    }
  }

  bool stopped() const { return optimum_; }
  const std::optional<Individual<G>>& current() const { return current_; }
  std::uint64_t steps() const { return step_; }
  std::size_t evaluations() const { return evaluations_; }

  // An injected point better than the current one is moved to.
  void inject(std::span<const Individual<G>> in) {
    for (const auto& x : in)
      if (current_ && better(x.evaluation, current_->evaluation)) {
        current_ = x;
        optimum_ = false;
        scan_.clear();
      }
  }

 private:
  Neighbourhood<G> nb_;
  Pivot pivot_;
  std::size_t chunk_;
  Draws draws_;
  heuresis::search::Batch<G> batch_;
  std::optional<Candidate<G>> start_;
  std::optional<Individual<G>> current_;
  std::vector<std::size_t> scan_;
  std::size_t scanned_ = 0, evaluations_ = 0;
  std::uint64_t step_ = 0, serial_ = 0, restarts_ = 0;
  bool optimum_ = false;
};

}  // namespace heuresis::local
