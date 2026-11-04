// Random search: independent samples from a fixed distribution on the space, the baseline every search is
// measured against.
//
// By the No Free Lunch theorems (Wolpert and Macready 1997, "No free lunch theorems for optimization", IEEE
// Trans. Evol. Comput. 1, 67–82, Theorem 1), averaged over all objectives on a finite space, every algorithm that
// does not revisit points has the same distribution of observed values as this one. A search is better than
// random search only on a class of problems whose structure it exploits; the checks of each algorithm say on
// which.
//
// It is asynchronous: it offers any number of candidates at any time, each sampled from the stream keyed by its
// serial number, so the k-th candidate is the same however the run is cut into batches or spread over workers.

#pragma once

#include <heuresis/search/draws.hpp>
#include <heuresis/search/protocol.hpp>

#include <functional>
#include <utility>

namespace heuresis::search {

template <class G>
class RandomSearch {
 public:
  using Genotype = G;
  using Sampler = std::function<G(calx::random::Stream&)>;

  explicit RandomSearch(Sampler sample, Draws draws = Draws{}) : sample_(std::move(sample)), draws_(draws) {}

  std::vector<Candidate<G>> ask(std::size_t limit) {
    if (limit == unlimited) limit = 1;
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

  static constexpr std::uint32_t purpose = tag("search.random");

 private:
  Sampler sample_;
  Draws draws_;
  std::uint64_t next_ = 0;
  Pending<G> outstanding_;
};

}  // namespace heuresis::search
