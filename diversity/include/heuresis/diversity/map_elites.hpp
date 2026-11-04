// MAP-Elites: the illumination of a space of behaviours by an archive of elites (Mouret and Clune 2015,
// "Illuminating search spaces by mapping elites", arXiv:1504.04909, Algorithm 1).
//
// After a number of random samples, every candidate is bred from elites drawn uniformly from the filled cells of the
// archive: a mutation of one, or, with probability p_v, a two-parent variation of two followed by the mutation (the
// iso+line operator of Vassiliades and Mouret 2018 on real vectors, variation/real.hpp). Each evaluation is offered
// to the archive, which keeps it if it fills or improves its cell. Coverage and QD-score never decrease
// (archive.hpp). With a grid it is MAP-Elites; with Voronoi cells, CVT-MAP-Elites.
//
// Asynchronous under the ask–tell protocol: candidates are bred from the archive as it stands, the k-th from the
// stream keyed by k, and each tell offers one individual; the run is reproducible from the order of arrival. It
// offers at most `breadth` bred candidates per ask (the batch of Mouret and Clune's parallel form; one by
// default); the initial samples are offered together.

#pragma once

#include <heuresis/diversity/archive.hpp>
#include <heuresis/search/draws.hpp>

#include <calx/random/stream.hpp>

#include <algorithm>
#include <functional>
#include <span>
#include <utility>
#include <vector>

namespace heuresis::diversity {

template <class G>
struct Kernels {
  std::function<G(calx::random::Stream&)> initialise;
  std::function<G(const G&, calx::random::Stream&)> mutate;
  std::function<G(const G&, const G&, calx::random::Stream&)> vary;  // optional: two parents, one child
};

template <class G, class Tessellation>
class MapElites {
 public:
  using Genotype = G;
  using Individual = heuresis::search::Individual<G>;
  using Candidate = heuresis::search::Candidate<G>;

  struct Options {
    std::size_t initial = 100;        // random samples before breeding
    double variation_probability = 0;  // p_v: the share bred by `vary`
    std::size_t breadth = 1;          // bred candidates per ask
  };

  MapElites(Kernels<G> kernels, Tessellation tessellation, Options options, heuresis::search::Draws draws = heuresis::search::Draws{})
      : kernels_(std::move(kernels)), archive_(std::move(tessellation)), options_(options), draws_(draws) {
    if (options_.breadth == 0) options_.breadth = 1;
  }

  std::vector<Candidate> ask(std::size_t limit) {
    std::vector<Candidate> out;
    std::size_t bred = 0;
    while (out.size() < limit) {
      if (serial_ < options_.initial) {
        calx::random::Stream s = draws_(sample_tag, 0, serial_);
        out.push_back({heuresis::search::make_ticket(draws_.island(), serial_), kernels_.initialise(s), {sample_tag, {}}});
      } else {
        if (archive_.size() == 0 || bred == options_.breadth) break;
        bred++;
        calx::random::Stream s = draws_(breed_tag, 0, serial_);
        const auto& filled = archive_.filled();
        const auto& a = *archive_.elite(filled[s.below(filled.size())]);
        if (kernels_.vary && s.uniform() < options_.variation_probability) {
          const auto& b = *archive_.elite(filled[s.below(filled.size())]);
          out.push_back({heuresis::search::make_ticket(draws_.island(), serial_), kernels_.mutate(kernels_.vary(a.genotype, b.genotype, s), s), {breed_tag, {a.ticket, b.ticket}}});
        } else {
          out.push_back({heuresis::search::make_ticket(draws_.island(), serial_), kernels_.mutate(a.genotype, s), {breed_tag, {a.ticket}}});
        }
      }
      serial_++;
      pending_.add(out.back());
    }
    return out;
  }

  void tell(std::span<const heuresis::search::Told> told) {
    for (const auto& t : told) {
      auto [c, note] = pending_.take(t.ticket);
      (void)note;
      archive_.offer({c.ticket, std::move(c.genotype), t.evaluation, std::move(c.origin)});
    }
  }

  bool stopped() const { return false; }

  void inject(std::span<const Individual> in) {
    for (const auto& x : in) archive_.offer(x);
  }
  // The k best elites by Deb's rule, ties to the cell filled first.
  std::vector<Individual> emigrants(std::size_t k) const {
    std::vector<Individual> all;
    for (auto c : archive_.filled()) all.push_back(*archive_.elite(c));
    std::stable_sort(all.begin(), all.end(), [](const Individual& a, const Individual& b) { return heuresis::search::better(a.evaluation, b.evaluation); });
    all.resize(std::min(k, all.size()));
    return all;
  }

  const Archive<G, Tessellation>& archive() const { return archive_; }

  static constexpr std::uint32_t sample_tag = heuresis::search::tag("diversity.map_elites.sample");
  static constexpr std::uint32_t breed_tag = heuresis::search::tag("diversity.map_elites.breed");

 private:
  Kernels<G> kernels_;
  Archive<G, Tessellation> archive_;
  Options options_;
  heuresis::search::Draws draws_;
  heuresis::search::Pending<G> pending_;
  std::uint64_t serial_ = 0;
};

}  // namespace heuresis::diversity
