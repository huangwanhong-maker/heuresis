// Restarts of CMA-ES with growing populations: IPOP and BIPOP.
//
// A larger population makes CMA-ES more global: on multimodal functions with a global structure (Rastrigin's,
// Schwefel's) the probability of finding the global optimum grows with λ (Hansen and Kern 2004, "Evaluating the
// CMA evolution strategy on multimodal test functions", PPSN VIII). IPOP restarts CMA-ES whenever it stops, from a
// uniform point of a box, with λ multiplied by 2 each time (Auger and Hansen 2005, "A restart CMA evolution
// strategy with increasing population size", CEC, 1769–1776). BIPOP interleaves two regimes (Hansen 2009,
// "Benchmarking a BI-population CMA-ES on the BBOB-2009 function testbed", GECCO workshop, §2): the large one is
// IPOP's (the first run, at the default λ, counts as its); in the small one λ_s = ⌊λ_def (λ_l/(2λ_def))^{U²}⌋ and
// σ_s = σ₀ 10^{−2U′}, U and U′ independent and uniform on (0, 1), λ_l the large regime's last population; the
// regime that has spent fewer evaluations so far runs next.
//
// The wrapper is itself an ask–tell algorithm. Its tickets continue from one run to the next; each run draws from
// streams keyed by the run's seed folded with the restart's number, so the whole sequence of runs is reproducible
// and independent of where its evaluations are made.

#pragma once

#include <heuresis/strategies/cma_es.hpp>

#include <cmath>
#include <memory>

namespace heuresis::strategies {

enum class RestartRegime : std::uint8_t { ipop, bipop };

struct RestartOptions {
  RestartRegime regime = RestartRegime::ipop;
  Box start;                   // the initial mean of each run is uniform in it
  double sigma0 = 1;
  double increase = 2;         // the factor of λ between large runs
  std::size_t max_large_runs = 10;  // runs of the large regime (all of IPOP's), the first included
  CmaOptions cma;              // for every run; its λ is the default λ_def (0: the tutorial's)
};

class CmaRestarts {
 public:
  using Genotype = Real;
  static constexpr std::uint32_t start_purpose = heuresis::search::tag("strategies.restarts.start");
  static constexpr std::uint32_t regime_purpose = heuresis::search::tag("strategies.restarts.regime");

  CmaRestarts(RestartOptions options, heuresis::search::Draws draws = heuresis::search::Draws{}) : options_(std::move(options)), draws_(draws) {
    options_.start.validate();
    default_lambda_ = CmaParameters::tutorial(options_.start.size(), options_.cma.lambda).lambda;
    large_lambda_ = default_lambda_;
    begin(default_lambda_, options_.sigma0, true);
  }

  std::vector<heuresis::search::Candidate<Real>> ask(std::size_t limit) {
    if (current_->stopped()) advance();
    if (finished_) return {};
    return current_->ask(limit);
  }

  void tell(std::span<const heuresis::search::Told> told) {
    current_->tell(told);
    const double spent = static_cast<double>(told.size());
    (in_large_ ? large_spent_ : small_spent_) += spent;
  }

  bool stopped() const { return finished_ || (large_runs_ >= options_.max_large_runs && current_->stopped()); }

  std::size_t runs() const { return runs_; }
  std::size_t lambda() const { return current_->parameters().lambda; }
  const Cma& current() const { return *current_; }

 private:
  void begin(std::size_t lambda, double sigma, bool large) {
    calx::random::Stream s = draws_(start_purpose, runs_, 0);
    Real m = heuresis::variation::uniform_point(options_.start, s);
    CmaOptions o = options_.cma;
    o.lambda = lambda;
    const std::uint64_t serial = current_ ? current_->next_serial() : 0;
    const heuresis::search::Draws run_draws(heuresis::search::mix(draws_.seed() ^ heuresis::search::mix(runs_ + 1)), draws_.island());
    current_ = std::make_unique<Cma>(std::move(m), sigma, o, run_draws, serial);
    in_large_ = large;
    large_runs_ += large;
    runs_++;
  }

  void advance() {
    const bool large = options_.regime == RestartRegime::ipop || large_spent_ <= small_spent_;
    if (large && large_runs_ >= options_.max_large_runs) { finished_ = true; return; }
    if (large) {
      large_lambda_ = static_cast<std::size_t>(std::llround(static_cast<double>(large_lambda_) * options_.increase));
      begin(large_lambda_, options_.sigma0, true);
    } else {
      calx::random::Stream s = draws_(regime_purpose, runs_, 0);
      const double u = s.uniform();
      const double ratio = static_cast<double>(large_lambda_) / (2 * static_cast<double>(default_lambda_));
      const auto lambda = std::max<std::size_t>(2, static_cast<std::size_t>(std::floor(static_cast<double>(default_lambda_) * std::pow(std::max(1.0, ratio), u * u))));
      begin(lambda, options_.sigma0 * std::pow(10.0, -2 * s.uniform()), false);
    }
  }

  RestartOptions options_;
  heuresis::search::Draws draws_;
  std::unique_ptr<Cma> current_;
  std::size_t default_lambda_ = 0, large_lambda_ = 0, runs_ = 0, large_runs_ = 0;
  double large_spent_ = 0, small_spent_ = 0;
  bool in_large_ = true, finished_ = false;
};

}  // namespace heuresis::strategies
