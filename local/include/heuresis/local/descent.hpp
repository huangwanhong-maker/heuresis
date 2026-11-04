// Variable neighbourhood descent: best improvement through a sequence of neighbourhoods.
//
// With neighbourhoods N₁, …, N_m, usually nested or growing: search N_k(x) for its best neighbour; if it is
// strictly better, move there and return to N₁; otherwise go on to N_{k+1}. It stops at a point that is a local
// optimum for every N_k at once, which few points are (Mladenović and Hansen 1997, "Variable neighborhood
// search", Comput. Oper. Res. 24, 1097–1100; Hansen and Mladenović 2001, "Variable neighborhood search:
// principles and applications", Eur. J. Oper. Res. 130, 449–467, §2).

#pragma once

#include <heuresis/local/neighbourhood.hpp>
#include <heuresis/search/search.hpp>

#include <optional>
#include <span>
#include <vector>

namespace heuresis::local {

template <class G>
class VariableNeighbourhoodDescent {
 public:
  using Genotype = G;
  static constexpr std::uint32_t start_tag = heuresis::search::tag("local.vnd.start");
  static constexpr std::uint32_t move_tag = heuresis::search::tag("local.vnd.move");

  VariableNeighbourhoodDescent(G start, std::vector<Neighbourhood<G>> neighbourhoods, heuresis::search::Draws draws = heuresis::search::Draws{})
      : nbs_(std::move(neighbourhoods)), draws_(draws), start_(std::move(start)) {
    if (nbs_.empty()) throw std::invalid_argument("heuresis::local::VariableNeighbourhoodDescent: no neighbourhood");
  }

  std::vector<heuresis::search::Candidate<G>> ask(std::size_t limit) {
    using heuresis::search::make_ticket;
    if (done_) return {};
    if (!batch_.active()) {
      std::vector<heuresis::search::Candidate<G>> g;
      if (start_) {
        g.push_back({make_ticket(draws_.island(), serial_++), std::move(*start_), {start_tag, {}}});
        start_.reset();
      } else {
        while (level_ < nbs_.size() && nbs_[level_].size(current_->genotype) == 0) level_++;
        if (level_ == nbs_.size()) { done_ = true; return {}; }
        const std::size_t m = nbs_[level_].size(current_->genotype);
        for (std::size_t k = 0; k < m; k++) g.push_back({make_ticket(draws_.island(), serial_++), nbs_[level_].apply(current_->genotype, k), {move_tag, {current_->ticket}}});
      }
      batch_.open(std::move(g));
    }
    return batch_.take(limit);
  }

  void tell(std::span<const heuresis::search::Told> told) {
    for (const auto& t : told) {
      if (!batch_.accept(t)) continue;
      auto xs = batch_.individuals();
      batch_.close();
      if (!current_) { current_ = std::move(xs[0]); continue; }
      const auto order = heuresis::search::ranking(batch_evaluations(xs));
      if (heuresis::search::better(xs[order[0]].evaluation, current_->evaluation)) {
        current_ = std::move(xs[order[0]]);
        level_ = 0;
      } else if (++level_ == nbs_.size()) {
        done_ = true;
      }
    }
  }

  bool stopped() const { return done_; }
  const std::optional<heuresis::search::Individual<G>>& current() const { return current_; }

 private:
  static std::vector<heuresis::search::Evaluation> batch_evaluations(const std::vector<heuresis::search::Individual<G>>& xs) {
    std::vector<heuresis::search::Evaluation> e;
    for (const auto& x : xs) e.push_back(x.evaluation);
    return e;
  }

  std::vector<Neighbourhood<G>> nbs_;
  heuresis::search::Draws draws_;
  std::optional<G> start_;
  heuresis::search::Batch<G> batch_;
  std::optional<heuresis::search::Individual<G>> current_;
  std::size_t level_ = 0;
  std::uint64_t serial_ = 0;
  bool done_ = false;
};

}  // namespace heuresis::local
