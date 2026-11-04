// Tabu search: always moving to the best admissible neighbour, forbidding the undoing of recent moves.
//
// Each iteration evaluates the whole neighbourhood and moves to its best admissible neighbour, even when that is
// worse than the current point, so the search walks out of local optima; the attribute of each move made (a
// bit's position) is tabu for `tenure` iterations, which keeps it from walking straight back. A tabu move is
// admissible all the same when it would give a point better than any seen, the aspiration criterion (Glover
// 1989, "Tabu search — Part I", ORSA J. Comput. 1, 190–206; Glover 1990, "Tabu search — Part II", ORSA J. Comput.
// 2, 4–32). When every move is tabu and none aspires, the move whose tabu ends soonest is made. Ties fall to the
// lower move index.
//
// The neighbourhood must give attributes. It is generational in the protocol's sense; it never stops by itself.

#pragma once

#include <heuresis/local/neighbourhood.hpp>
#include <heuresis/search/search.hpp>

#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

namespace heuresis::local {

template <class G>
class TabuSearch {
 public:
  using Genotype = G;
  static constexpr std::uint32_t start_tag = heuresis::search::tag("local.tabu.start");
  static constexpr std::uint32_t move_tag = heuresis::search::tag("local.tabu.move");

  TabuSearch(G start, Neighbourhood<G> neighbourhood, std::size_t tenure, heuresis::search::Draws draws = heuresis::search::Draws{})
      : nb_(std::move(neighbourhood)), tenure_(tenure), draws_(draws), start_(std::move(start)) {
    if (!nb_.attribute) throw std::invalid_argument("heuresis::local::TabuSearch: the neighbourhood gives no attributes");
  }

  std::vector<heuresis::search::Candidate<G>> ask(std::size_t limit) {
    using heuresis::search::make_ticket;
    if (!batch_.active()) {
      std::vector<heuresis::search::Candidate<G>> g;
      std::vector<std::size_t> moves;
      if (start_) {
        g.push_back({make_ticket(draws_.island(), serial_++), std::move(*start_), {start_tag, {}}});
        start_.reset();
      } else {
        const std::size_t m = nb_.size(current_->genotype);
        for (std::size_t k = 0; k < m; k++) {
          g.push_back({make_ticket(draws_.island(), serial_++), nb_.apply(current_->genotype, k), {move_tag, {current_->ticket}}});
          moves.push_back(k);
        }
      }
      if (g.empty()) return {};
      batch_.open(std::move(g), std::move(moves));
    }
    return batch_.take(limit);
  }

  void tell(std::span<const heuresis::search::Told> told) {
    using heuresis::search::better;
    for (const auto& t : told) {
      if (!batch_.accept(t)) continue;
      auto xs = batch_.individuals();
      if (!current_) {
        current_ = xs[0];
        best_ = xs[0];
        batch_.close();
        continue;
      }
      std::optional<std::size_t> chosen, soonest;
      for (std::size_t i = 0; i < xs.size(); i++) {
        const std::size_t a = nb_.attribute(current_->genotype, batch_.note(i));
        const auto it = until_.find(a);
        const bool tabu = it != until_.end() && it->second > iteration_;
        const bool aspires = better(xs[i].evaluation, best_->evaluation);
        if (!tabu || aspires) {
          if (!chosen || better(xs[i].evaluation, xs[*chosen].evaluation)) chosen = i;
        } else if (!soonest || it->second < until_[nb_.attribute(current_->genotype, batch_.note(*soonest))]) {
          soonest = i;
        }
      }
      const std::size_t i = chosen ? *chosen : *soonest;
      until_[nb_.attribute(current_->genotype, batch_.note(i))] = iteration_ + 1 + tenure_;
      batch_.close();
      current_ = std::move(xs[i]);
      if (better(current_->evaluation, best_->evaluation)) best_ = current_;
      iteration_++;
    }
  }

  bool stopped() const { return false; }
  const std::optional<heuresis::search::Individual<G>>& current() const { return current_; }
  const std::optional<heuresis::search::Individual<G>>& best() const { return best_; }

 private:
  Neighbourhood<G> nb_;
  std::size_t tenure_;
  heuresis::search::Draws draws_;
  std::optional<G> start_;
  heuresis::search::Batch<G, std::size_t> batch_;
  std::optional<heuresis::search::Individual<G>> current_, best_;
  std::unordered_map<std::size_t, std::uint64_t> until_;  // attribute → the first iteration at which it is free
  std::uint64_t iteration_ = 0, serial_ = 0;
};

}  // namespace heuresis::local
