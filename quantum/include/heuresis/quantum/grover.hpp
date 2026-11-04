// Grover adaptive search: minimisation by amplitude amplification, on a device or on the exact law of one.
//
// A space of N points x with values f(x). Given a threshold y, the oracle marks the t points with f(x) < y; r Grover
// iterations from the uniform superposition leave the register in the plane spanned by the uniform states on the
// marked and on the unmarked points, at angle (2r + 1)θ from the latter, sin²θ = t/N (Grover 1996, "A fast quantum
// mechanical algorithm for database search", STOC, 212–219; Boyer, Brassard, Høyer and Tapp 1998, "Tight bounds on
// quantum searching", Fortschr. Phys. 46, 493–505). A measurement therefore returns
//
//   a marked point with probability sin²((2r + 1)θ), uniformly among the marked; otherwise an unmarked point,
//   uniformly among the unmarked.
//
// This law is exact, and it is all a search sees of the device. `EmulatedGrover` draws from it: to know the marked
// set it evaluates f at every point once, classically (N evaluations, reported apart from the oracle calls a device
// would spend). It emulates the law of a quantum computer, not a quantum computer: it says what a device would return
// and what it would cost in oracle calls, and is the reference against which a device or a simulator of registers and
// gates is tested (decision 0003 §2.8). Registers, gates and measurement as such belong to physis's `quantum` when it
// holds them; any type with `size()` and `measure(threshold, r, stream)` is a `GroverDevice`.
//
// Grover adaptive search (Dürr and Høyer 1996, "A quantum algorithm for finding the minimum", arXiv:quant-ph/9607014;
// Baritompa, Bulger and Wood 2005, "Grover's quantum algorithm applied to global optimization", SIAM J. Optim. 15,
// 1170–1184): begin with a uniformly measured point as the threshold; at each step choose r uniformly among the
// integers below m, measure, evaluate the outcome classically; on an improvement take it as the threshold and reset
// m = 1, otherwise m ← min(λm, √N), with λ = 6/5 (any 1 < λ < 4/3 serves; Boyer et al. 1998). Dürr and Høyer prove
// that the expected number of oracle calls before the threshold holds the minimum is at most (45/4)√N + (7/10) lg² N,
// so that stopping after 22.5√N + 1.4 lg² N finds it with probability at least ½; a classical search needs Θ(N).

#pragma once

#include <heuresis/search/draws.hpp>
#include <heuresis/search/protocol.hpp>

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <vector>

namespace heuresis::quantum {

struct Measurement {
  std::uint64_t index = 0;         // the point measured
  std::uint64_t oracle_calls = 0;  // Grover iterations spent
};

template <class D>
concept GroverDevice = requires(D& d, const D& c, double threshold, std::uint64_t r, calx::random::Stream& s) {
  { c.size() } -> std::convertible_to<std::uint64_t>;
  { d.measure(threshold, r, s) } -> std::same_as<Measurement>;
};

// sin²((2r + 1)θ), sin²θ = t/N.
inline double grover_success(std::uint64_t t, std::uint64_t N, std::uint64_t r) {
  if (N == 0 || t > N) throw std::invalid_argument("heuresis::quantum::grover_success: 0 ≤ t ≤ N, N ≥ 1");
  const double theta = std::asin(std::sqrt(static_cast<double>(t) / static_cast<double>(N)));
  const double s = std::sin((2 * static_cast<double>(r) + 1) * theta);
  return s * s;
}

class EmulatedGrover {
 public:
  EmulatedGrover(std::uint64_t N, const std::function<double(std::uint64_t)>& f) : values_(N), order_(N) {
    if (N == 0) throw std::invalid_argument("heuresis::quantum::EmulatedGrover: an empty space");
    for (std::uint64_t i = 0; i < N; i++) values_[i] = f(i);
    std::iota(order_.begin(), order_.end(), std::uint64_t{0});
    std::stable_sort(order_.begin(), order_.end(), [&](std::uint64_t a, std::uint64_t b) { return values_[a] < values_[b]; });
  }

  std::uint64_t size() const { return values_.size(); }
  // The number of points with f < threshold.
  std::uint64_t marked(double threshold) const {
    return static_cast<std::uint64_t>(std::lower_bound(order_.begin(), order_.end(), threshold, [&](std::uint64_t i, double y) { return values_[i] < y; }) - order_.begin());
  }
  Measurement measure(double threshold, std::uint64_t r, calx::random::Stream& s) {
    const std::uint64_t N = size(), t = marked(threshold);
    const bool hit = t > 0 && s.uniform() < grover_success(t, N, r);
    const std::uint64_t index = hit ? order_[s.below(t)] : (t < N ? order_[t + s.below(N - t)] : order_[s.below(N)]);
    return {index, r};
  }
  // What the emulation cost classically: one evaluation of every point.
  std::uint64_t classical_evaluations() const { return values_.size(); }
  double minimum() const { return values_[order_[0]]; }

 private:
  std::vector<double> values_;
  std::vector<std::uint64_t> order_;  // the points by increasing value, ties by index
};

struct GroverSearchOptions {
  double lambda = 6.0 / 5.0;
  double cutoff = 0;  // stop after this many oracle calls; 0: never (dh_cutoff gives Dürr and Høyer's)
};

// Dürr and Høyer's cutoff, 22.5√N + 1.4 lg² N.
inline double dh_cutoff(std::uint64_t N) {
  const double lg = std::log2(static_cast<double>(N));
  return 22.5 * std::sqrt(static_cast<double>(N)) + 1.4 * lg * lg;
}
// Their bound on the expected oracle calls before the minimum is held, (45/4)√N + (7/10) lg² N.
inline double dh_expected_bound(std::uint64_t N) {
  const double lg = std::log2(static_cast<double>(N));
  return 11.25 * std::sqrt(static_cast<double>(N)) + 0.7 * lg * lg;
}

// Grover adaptive search on a device: a candidate is a measured point, its evaluation the classical check.
template <GroverDevice D>
class GroverAdaptiveSearch {
 public:
  using Genotype = std::uint64_t;

  GroverAdaptiveSearch(D& device, GroverSearchOptions options = {}, heuresis::search::Draws draws = heuresis::search::Draws{}) : device_(&device), options_(options), draws_(draws) {
    if (!(options_.lambda > 1 && options_.lambda < 4.0 / 3.0)) throw std::invalid_argument("heuresis::quantum::GroverAdaptiveSearch: 1 < λ < 4/3");
  }

  std::vector<heuresis::search::Candidate<std::uint64_t>> ask(std::size_t limit) {
    if (limit == 0 || outstanding_ || stopped()) return {};
    calx::random::Stream s = draws_(measure_tag, step_, 0);
    std::uint64_t r = 0;
    if (has_threshold_) r = s.below(static_cast<std::uint64_t>(std::ceil(m_)));
    const double threshold = has_threshold_ ? threshold_ : std::numeric_limits<double>::infinity();
    const Measurement got = device_->measure(threshold, r, s);
    oracle_calls_ += got.oracle_calls;
    outstanding_ = heuresis::search::make_ticket(draws_.island(), step_++);
    return {{*outstanding_, got.index, heuresis::search::Origin{measure_tag, {}}}};
  }

  void tell(std::span<const heuresis::search::Told> told) {
    for (const auto& t : told) {
      if (!outstanding_ || t.ticket != *outstanding_) throw std::invalid_argument("heuresis::quantum::GroverAdaptiveSearch: a ticket that is not outstanding");
      outstanding_.reset();
      const double v = t.evaluation.feasible() ? t.evaluation.value() : std::numeric_limits<double>::infinity();
      if (!has_threshold_ || v < threshold_) {
        threshold_ = v;
        has_threshold_ = true;
        m_ = 1;
      } else {
        m_ = std::min(options_.lambda * m_, std::sqrt(static_cast<double>(device_->size())));
      }
    }
  }

  bool stopped() const { return options_.cutoff > 0 && static_cast<double>(oracle_calls_) >= options_.cutoff; }

  std::uint64_t oracle_calls() const { return oracle_calls_; }
  std::uint64_t measurements() const { return step_; }
  double threshold() const { return threshold_; }
  static constexpr std::uint32_t measure_tag = heuresis::search::tag("quantum.grover.measure");

 private:
  D* device_;
  GroverSearchOptions options_;
  heuresis::search::Draws draws_;
  std::optional<heuresis::search::Ticket> outstanding_;
  bool has_threshold_ = false;
  double threshold_ = std::numeric_limits<double>::infinity();
  double m_ = 1;
  std::uint64_t oracle_calls_ = 0, step_ = 0;
};

}  // namespace heuresis::quantum
