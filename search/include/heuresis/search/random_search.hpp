// Random search: independent samples from a fixed distribution on the space, the baseline every search is
// measured against.
//
// By the No Free Lunch theorems (Wolpert and Macready 1997, "No free lunch theorems for optimization", IEEE
// Trans. Evol. Comput. 1, 67–82, Theorem 1), averaged over all objectives on a finite space, every algorithm that
// does not revisit points has the same distribution of observed values as this one. A search is better than
// random search only on a class of problems whose structure it exploits; the checks of each algorithm say on
// which.
//
// It is asynchronous: it offers candidates at any time, each sampled from the stream keyed by its serial number, so the
// k-th candidate is the same however the run is cut into batches or spread over workers. It offers at most `breadth`
// per ask (by default 1, as for every asynchronous algorithm, search/protocol.hpp): under run() with a finite budget an
// ask's limit is the whole remaining budget, which would otherwise be sampled and held at once. A breadth as wide as
// the executor (a pool's threads, a device's batch) keeps it busy.

#pragma once

#include <heuresis/search/draws.hpp>
#include <heuresis/search/protocol.hpp>

#include <algorithm>
#include <functional>
#include <utility>

namespace heuresis::search {

template <class G>
class RandomSearch {
 public:
  using Genotype = G;
  using Sampler = std::function<G(calx::random::Stream&)>;

  explicit RandomSearch(Sampler sample, Draws draws = Draws{}, std::size_t breadth = 1) : sample_(std::move(sample)), draws_(draws), breadth_(std::max<std::size_t>(1, breadth)) {}

  std::vector<Candidate<G>> ask(std::size_t limit) {
    limit = std::min(limit, breadth_);
    std::vector<Candidate<G>> out;
    out.reserve(limit);
    for (std::size_t i = 0; i < limit; i++) {
      calx::random::Stream s = draws_(purpose, 0, next_);
      Candidate<G> c{make_ticket(draws_.island(), next_++), sample_(s), Origin{purpose, {}}};
      outstanding_.add(c);
      out.push_back(std::move(c));
    }
    return out;
  }
  void tell(std::span<const Told> told) {
    for (const auto& t : told) outstanding_.take(t.ticket);
  }
  bool stopped() const { return false; }
  void inject(std::span<const Individual<G>>) {}
  std::size_t breadth() const { return breadth_; }

  static constexpr std::uint32_t purpose = tag("search.random");

 private:
  Sampler sample_;
  Draws draws_;
  std::size_t breadth_;
  std::uint64_t next_ = 0;
  Pending<G> outstanding_;
};

}  // namespace heuresis::search
