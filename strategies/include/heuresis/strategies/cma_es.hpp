// The covariance matrix adaptation evolution strategy, CMA-ES.
//
// A search distribution N(m, σ²C) on ℝⁿ whose mean, step size and covariance move towards what was good, in the
// form of Hansen (2016), "The CMA Evolution Strategy: A Tutorial", arXiv:1604.00772 (the tutorial below; its
// Table 1 gives the default parameters). One generation:
//
//   sample   y_k = B D z_k, x_k = m + σ y_k,  z_k ~ N(0, I), k = 1..λ;          C = B D² Bᵀ
//   rank     x_{1:λ} best first (Deb's rule, ties to the earlier ticket);   y_{i:λ} = (x_{i:λ} − m)/σ
//   mean     m ← m + c_m σ ⟨y⟩_w,  ⟨y⟩_w = Σ_{i≤μ} w_i y_{i:λ}
//   paths    p_σ ← (1 − c_σ) p_σ + √(c_σ(2 − c_σ)μ_eff) C^{−1/2}⟨y⟩_w
//            h_σ = [‖p_σ‖/√(1 − (1 − c_σ)^{2(g+1)}) < (1.4 + 2/(n + 1)) E‖N(0, I)‖]
//            p_c ← (1 − c_c) p_c + h_σ √(c_c(2 − c_c)μ_eff) ⟨y⟩_w
//   matrix   C ← (1 + c_1 δ(h_σ) − c_1 − c_μ Σ_j w_j) C + c_1 p_c p_cᵀ + c_μ Σ_{i=1}^{λ} w_i° y_{i:λ} y_{i:λ}ᵀ,
//            δ(h_σ) = (1 − h_σ) c_c (2 − c_c),  w_i° = w_i if w_i ≥ 0, else w_i n/‖C^{−1/2} y_{i:λ}‖²
//   step     σ ← σ exp((c_σ/d_σ)(‖p_σ‖/E‖N(0, I)‖ − 1))
//
// (the tutorial's eqs. 39–47, its Table 1 for the parameters). The negative weights w_i < 0 of the λ − μ worst
// make the update "active" (Jastrebski and Arnold 2006, "Improving evolution strategies through active
// covariance matrix adaptation", CEC; the tutorial's form, with their sum bounded so that C stays positive
// definite); without them the update is the classical one of Hansen and Ostermeier (2001, "Completely
// derandomized self-adaptation in evolution strategies", Evol. Comput. 9, 159–195). The eigendecomposition is
// lazy: done every max(1, ⌊1/(10n(c_1 + c_μ))⌋) generations (the tutorial, §B.2), C^{−1/2} = B D⁻¹ Bᵀ from it.
//
// The separable variant keeps C diagonal, at O(n) per sample, with the learning rates c_1 and c_μ multiplied by
// (n + 2)/3 (Ros and Hansen 2008, "A simple modification in CMA-ES achieving linear time and space complexity",
// PPSN X, 296–305).
//
// Invariance (the theory note, §3; checked exactly): the ranking alone reaches the update, so the run is the
// same on g ∘ f for any strictly increasing g; and the update commutes with every affine map x ↦ Ax + b of the
// space, the state (m, C, p_c) mapped with it, σ and ‖p_σ‖ unchanged. The update is a pure function
// (`cma_update`), so that a sample from anywhere — pycma's, in the checks — can be told to it.
//
// Bounds: by default the distribution lives on ℝⁿ and its samples are mapped onto the box by Hansen's
// linear–quadratic transformation (variation/box_transform.hpp), the mapped point evaluated, the sample told to the update,
// so that an optimum on the boundary is an interior optimum of the transformed problem; or, if asked, a sample is
// repaired (reflected, clipped, …) and the repaired point told as if sampled ("Lamarckian" repair), which lets σ
// grow without bound against a wall where the optimum is a corner (the checks show both). Injected solutions
// (from another island, an archive, a person) are ranked with the next generation, their steps clipped to
// Mahalanobis length √n + 2n/(n + 2) (Hansen 2011, "Injecting external solutions into CMA-ES", INRIA RR-7748).
//
// Termination (the tutorial, §B.3, as `stopped()`): TolX (σ p_c and σ√C_ii below 10⁻¹² σ₀), TolFun (the range of
// the best values of the last 10 + ⌈30n/λ⌉ generations and of the last generation's values below 10⁻¹²),
// ConditionCov (above 10¹⁴), NoEffectAxis, NoEffectCoord, Stagnation, TolUpSigma, and a generation limit.

#pragma once

#include <heuresis/search/draws.hpp>
#include <heuresis/search/protocol.hpp>
#include <heuresis/variation/box_transform.hpp>
#include <heuresis/variation/real.hpp>

#include <calx/linalg/dense.hpp>
#include <calx/linalg/eigen.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

namespace heuresis::strategies {

using heuresis::variation::Box;
using heuresis::variation::BoxTransform;
using heuresis::variation::Real;
using heuresis::variation::Repair;

struct CmaParameters {
  std::size_t n = 0, lambda = 0, mu = 0;
  std::vector<double> weights;  // λ weights, best first: positive for the μ best, then ≤ 0
  double mueff = 0, cc = 0, cs = 0, c1 = 0, cmu = 0, damps = 0, cm = 1;
  double chi_n = 0;  // E‖N(0, I)‖

  double weight_sum() const {
    double s = 0;
    for (double w : weights) s += w;
    return s;
  }

  // The defaults of the tutorial's Table 1; lambda = 0 gives 4 + ⌊3 ln n⌋.
  static CmaParameters tutorial(std::size_t n, std::size_t lambda = 0, bool active = true) {
    if (n == 0) throw std::invalid_argument("heuresis::strategies::CmaParameters: dimension 0");
    CmaParameters p;
    p.n = n;
    const double N = static_cast<double>(n);
    p.lambda = lambda ? lambda : 4 + static_cast<std::size_t>(std::floor(3 * std::log(N)));
    if (p.lambda < 2) throw std::invalid_argument("heuresis::strategies::CmaParameters: λ below 2");
    p.mu = p.lambda / 2;
    std::vector<double> raw(p.lambda);
    for (std::size_t i = 0; i < p.lambda; i++) raw[i] = std::log((static_cast<double>(p.lambda) + 1) / 2) - std::log(static_cast<double>(i + 1));
    double sp = 0, sp2 = 0, sn = 0, sn2 = 0;
    for (std::size_t i = 0; i < p.mu; i++) { sp += raw[i]; sp2 += raw[i] * raw[i]; }
    for (std::size_t i = p.mu; i < p.lambda; i++)
      if (raw[i] < 0) { sn += -raw[i]; sn2 += raw[i] * raw[i]; }
    p.mueff = sp * sp / sp2;
    const double mueff_minus = sn2 > 0 ? sn * sn / sn2 : 0;
    p.cc = (4 + p.mueff / N) / (N + 4 + 2 * p.mueff / N);
    p.cs = (p.mueff + 2) / (N + p.mueff + 5);
    p.c1 = 2 / ((N + 1.3) * (N + 1.3) + p.mueff);
    p.cmu = std::min(1 - p.c1, 2 * (p.mueff - 2 + 1 / p.mueff) / ((N + 2) * (N + 2) + p.mueff));
    p.damps = 1 + 2 * std::max(0.0, std::sqrt((p.mueff - 1) / (N + 1)) - 1) + p.cs;
    p.chi_n = std::sqrt(N) * (1 - 1 / (4 * N) + 1 / (21 * N * N));
    const double alpha_mu = 1 + p.c1 / p.cmu, alpha_mueff = 1 + 2 * mueff_minus / (p.mueff + 2), alpha_posdef = (1 - p.c1 - p.cmu) / (N * p.cmu);
    const double negative_scale = std::min({alpha_mu, alpha_mueff, alpha_posdef});
    p.weights.assign(p.lambda, 0.0);
    for (std::size_t i = 0; i < p.lambda; i++) {
      if (i < p.mu) p.weights[i] = raw[i] / sp;
      else if (active && raw[i] < 0) p.weights[i] = negative_scale * raw[i] / sn;
    }
    return p;
  }

  // The separable variant's learning rates (Ros and Hansen 2008).
  CmaParameters separable() const {
    CmaParameters p = *this;
    const double f = (static_cast<double>(n) + 2) / 3;
    p.c1 = c1 * f;
    p.cmu = std::min(1 - p.c1, cmu * f);
    return p;
  }

  // The tutorial's gap between eigendecompositions.
  std::size_t eigen_gap() const { return std::max<std::size_t>(1, static_cast<std::size_t>(std::floor(1 / (10 * static_cast<double>(n) * (c1 + cmu))))); }
};

struct CmaState {
  Real mean;
  double sigma = 1;
  calx::linalg::Matrix C;             // the covariance; diagonal in the separable variant
  Real pc, ps;
  calx::linalg::Matrix B;             // eigenvectors (columns) at the last decomposition
  Real D;                             // square roots of the eigenvalues
  calx::linalg::Matrix inverse_root;  // B D⁻¹ Bᵀ, at the last decomposition
  std::uint64_t generation = 0;       // generations completed
  std::uint64_t decomposed = 0;       // the generation of the last decomposition
  bool hsig = true;                   // h_σ of the last update
  bool separable = false;

  static CmaState initial(Real mean, double sigma, bool separable = false) {
    CmaState s;
    const std::size_t n = mean.size();
    s.mean = std::move(mean);
    s.sigma = sigma;
    s.C = calx::linalg::Matrix::identity(n);
    s.pc.assign(n, 0.0);
    s.ps.assign(n, 0.0);
    s.separable = separable;
    s.decompose();
    return s;
  }

  std::size_t size() const { return mean.size(); }

  // C = B D² Bᵀ, and C^{−1/2}; eigenvalues below the smallest positive normal number are raised to it.
  void decompose() {
    const std::size_t n = size();
    B = calx::linalg::Matrix::identity(n);
    D.assign(n, 1.0);
    if (separable) {
      for (std::size_t i = 0; i < n; i++) D[i] = std::sqrt(std::max(C(i, i), std::numeric_limits<double>::min()));
    } else {
      for (std::size_t i = 0; i < n; i++)
        for (std::size_t j = 0; j < i; j++) C(i, j) = C(j, i) = 0.5 * (C(i, j) + C(j, i));
      const auto e = calx::linalg::symmetric_eigen(C);
      B = e.vectors;
      for (std::size_t i = 0; i < n; i++) D[i] = std::sqrt(std::max(e.values[i], std::numeric_limits<double>::min()));
    }
    inverse_root = calx::linalg::Matrix(n);
    for (std::size_t i = 0; i < n; i++)
      for (std::size_t j = 0; j < n; j++) {
        double v = 0;
        for (std::size_t k = 0; k < n; k++) v += B(i, k) * B(j, k) / D[k];
        inverse_root(i, j) = v;
      }
    decomposed = generation;
  }

  // y ↦ C^{−1/2} y by the last decomposition.
  Real whiten(std::span<const double> y) const {
    const std::size_t n = size();
    Real z(n, 0.0);
    for (std::size_t i = 0; i < n; i++)
      for (std::size_t j = 0; j < n; j++) z[i] += inverse_root(i, j) * y[j];
    return z;
  }

  // z ↦ B D z, a step of the distribution from a standard normal vector.
  void colour(std::span<const double> z, std::span<double> y) const {
    const std::size_t n = size();
    for (std::size_t i = 0; i < n; i++) {
      double v = 0;
      for (std::size_t k = 0; k < n; k++) v += B(i, k) * D[k] * z[k];
      y[i] = v;
    }
  }

  double condition() const {
    const auto [lo, hi] = std::minmax_element(D.begin(), D.end());
    return (*hi * *hi) / (*lo * *lo);
  }
};

inline double norm(std::span<const double> v) {
  double s = 0;
  for (double x : v) s += x * x;
  return std::sqrt(s);
}

// One generation's transition: the λ points ranked best first, row by row (λ × n). Pure, deterministic.
inline void cma_update(const CmaParameters& p, CmaState& s, std::span<const double> ranked) {
  const std::size_t n = p.n, lambda = p.lambda;
  if (s.size() != n || ranked.size() != lambda * n) throw std::invalid_argument("heuresis::strategies::cma_update: points of the wrong shape");
  const std::uint64_t g = s.generation + 1;
  const Real old = s.mean;
  std::vector<double> y(lambda * n);
  for (std::size_t k = 0; k < lambda; k++)
    for (std::size_t i = 0; i < n; i++) y[k * n + i] = (ranked[k * n + i] - old[i]) / s.sigma;
  Real yw(n, 0.0);
  for (std::size_t k = 0; k < lambda; k++)
    if (p.weights[k] > 0)
      for (std::size_t i = 0; i < n; i++) yw[i] += p.weights[k] * y[k * n + i];
  for (std::size_t i = 0; i < n; i++) s.mean[i] = old[i] + p.cm * s.sigma * yw[i];

  const Real z = s.whiten(yw);
  const double a_s = std::sqrt(p.cs * (2 - p.cs) * p.mueff);
  for (std::size_t i = 0; i < n; i++) s.ps[i] = (1 - p.cs) * s.ps[i] + a_s * z[i];
  const double ps_norm = norm(s.ps);
  s.hsig = ps_norm / std::sqrt(1 - std::pow(1 - p.cs, 2.0 * static_cast<double>(g))) < (1.4 + 2 / (static_cast<double>(n) + 1)) * p.chi_n;
  const double h = s.hsig ? 1.0 : 0.0, a_c = std::sqrt(p.cc * (2 - p.cc) * p.mueff);
  for (std::size_t i = 0; i < n; i++) s.pc[i] = (1 - p.cc) * s.pc[i] + h * a_c * yw[i];

  std::vector<double> w(lambda);
  for (std::size_t k = 0; k < lambda; k++) {
    w[k] = p.weights[k];
    if (w[k] < 0) {
      const double m = norm(s.whiten(std::span<const double>(y.data() + k * n, n)));
      w[k] *= static_cast<double>(n) / (m * m);
    }
  }
  const double delta = (1 - h) * p.cc * (2 - p.cc);
  const double decay = 1 + p.c1 * delta - p.c1 - p.cmu * p.weight_sum();
  for (std::size_t i = 0; i < n; i++)
    for (std::size_t j = s.separable ? i : 0; j <= i; j++) {
      double v = decay * s.C(i, j) + p.c1 * s.pc[i] * s.pc[j];
      for (std::size_t k = 0; k < lambda; k++)
        if (w[k] != 0) v += p.cmu * w[k] * y[k * n + i] * y[k * n + j];
      s.C(i, j) = s.C(j, i) = v;
    }
  s.sigma *= std::exp((p.cs / p.damps) * (ps_norm / p.chi_n - 1));
  s.generation = g;
}

enum class CmaBounds : std::uint8_t { transform, repair };

enum class CmaStop : std::uint8_t { none, generations, tol_x, tol_fun, condition, no_effect_axis, no_effect_coordinate, stagnation, tol_up_sigma };

struct CmaOptions {
  std::size_t lambda = 0;  // 0: 4 + ⌊3 ln n⌋
  bool active = true;
  bool separable = false;
  std::optional<Box> bounds;
  CmaBounds bound_handling = CmaBounds::transform;
  Repair repair = Repair::reflect;  // for CmaBounds::repair
  std::size_t eigen_gap = 0;  // 0: the tutorial's gap
  std::optional<CmaParameters> parameters;  // overrides the defaults entirely
  double tol_x = 1e-12;                     // relative to σ₀
  double tol_fun = 1e-12;
  double max_condition = 1e14;
  double tol_up_sigma = 1e20;
  std::uint64_t max_generations = std::numeric_limits<std::uint64_t>::max();
};

class Cma {
 public:
  using Genotype = Real;
  static constexpr std::uint32_t sample_purpose = heuresis::search::tag("strategies.cma.sample");
  static constexpr std::uint32_t inject_purpose = heuresis::search::tag("strategies.cma.inject");

  Cma(Real mean, double sigma, CmaOptions options = {}, heuresis::search::Draws draws = heuresis::search::Draws{}, std::uint64_t first_serial = 0)
      : options_(std::move(options)), draws_(draws), serial_(first_serial), sigma0_(sigma) {
    const std::size_t n = mean.size();
    if (n == 0 || !(sigma > 0)) throw std::invalid_argument("heuresis::strategies::Cma: an empty mean or a step size that is not positive");
    params_ = options_.parameters ? *options_.parameters : CmaParameters::tutorial(n, options_.lambda, options_.active);
    if (options_.separable && !options_.parameters) params_ = params_.separable();
    if (params_.n != n) throw std::invalid_argument("heuresis::strategies::Cma: parameters for another dimension");
    gap_ = options_.eigen_gap ? options_.eigen_gap : params_.eigen_gap();
    if (options_.bounds) {
      options_.bounds->validate();
      if (options_.bound_handling == CmaBounds::transform) {
        transform_.emplace(*options_.bounds);
        mean = transform_->inverse(mean);
      }
    }
    state_ = CmaState::initial(std::move(mean), sigma, options_.separable);
  }

  std::vector<heuresis::search::Candidate<Real>> ask(std::size_t limit) {
    if (!batch_.active() && reason_ == CmaStop::none) open_generation();
    return batch_.take(limit);
  }

  void tell(std::span<const heuresis::search::Told> told) {
    for (const auto& t : told)
      if (batch_.accept(t)) close_generation();
  }

  bool stopped() const { return reason_ != CmaStop::none && !batch_.active(); }
  CmaStop stop_reason() const { return reason_; }

  // Evaluated solutions from outside, ranked with the next generation.
  void inject(std::span<const heuresis::search::Individual<Real>> in) {
    for (const auto& x : in)
      if (x.genotype.size() == state_.size()) injected_.push_back(x);
  }

  // The point evaluated for the distribution's mean.
  Real mean_point() const { return phenotype(state_.mean); }

  // The k best of the last completed generation.
  std::vector<heuresis::search::Individual<Real>> emigrants(std::size_t k) const {
    std::vector<heuresis::search::Individual<Real>> out(last_.begin(), last_.begin() + static_cast<std::ptrdiff_t>(std::min(k, last_.size())));
    return out;
  }

  const CmaState& state() const { return state_; }
  CmaState& state() { return state_; }
  const CmaParameters& parameters() const { return params_; }
  std::uint64_t next_serial() const { return serial_; }
  // The number of points told to the update, generation by generation.
  std::uint64_t evaluations() const { return evaluations_; }

 private:
  // The point evaluated for a sample of the distribution.
  Real phenotype(const Real& x) const { return transform_ ? transform_->forward(x) : x; }

  void open_generation() {
    const std::size_t n = state_.size(), lambda = params_.lambda;
    std::vector<heuresis::search::Candidate<Real>> c(lambda);
    std::vector<Real> samples(lambda);
    std::vector<double> zrow(n);
    for (std::size_t k = 0; k < lambda; k++) {
      calx::random::Stream s = draws_(sample_purpose, state_.generation, k);
      for (auto& v : zrow) v = s.normal();
      Real x(n);
      state_.colour(zrow, x);
      for (std::size_t i = 0; i < n; i++) x[i] = state_.mean[i] + state_.sigma * x[i];
      if (options_.bounds && !transform_) heuresis::variation::repair(x, *options_.bounds, options_.repair, s);
      c[k] = {heuresis::search::make_ticket(draws_.island(), serial_++), phenotype(x), heuresis::search::Origin{sample_purpose, {}}};
      samples[k] = std::move(x);
    }
    batch_.open(std::move(c), std::move(samples));
  }

  void close_generation() {
    const std::size_t n = state_.size(), lambda = params_.lambda;
    std::vector<heuresis::search::Individual<Real>> pool = batch_.individuals();
    std::vector<Real> internal(pool.size());
    for (std::size_t k = 0; k < pool.size(); k++) internal[k] = batch_.note(k);
    batch_.close();
    evaluations_ += pool.size();
    // Injected solutions, their steps clipped in Mahalanobis length.
    const double limit = std::sqrt(static_cast<double>(n)) + 2 * static_cast<double>(n) / (static_cast<double>(n) + 2);
    for (auto& x : injected_) {
      Real g = transform_ ? transform_->inverse(x.genotype) : x.genotype;
      Real y(n);
      for (std::size_t i = 0; i < n; i++) y[i] = (g[i] - state_.mean[i]) / state_.sigma;
      const double m = norm(state_.whiten(y));
      if (m > limit)
        for (std::size_t i = 0; i < n; i++) g[i] = state_.mean[i] + state_.sigma * y[i] * limit / m;
      x.genotype = phenotype(g);
      internal.push_back(std::move(g));
      pool.push_back(std::move(x));
    }
    injected_.clear();
    std::vector<heuresis::search::Evaluation> e(pool.size());
    for (std::size_t i = 0; i < pool.size(); i++) e[i] = pool[i].evaluation;
    const auto order = heuresis::search::ranking(e);
    std::vector<double> ranked(lambda * n);
    last_.clear();
    for (std::size_t r = 0; r < lambda; r++) {
      const auto& g = internal[order[r]];
      std::copy(g.begin(), g.end(), ranked.begin() + static_cast<std::ptrdiff_t>(r * n));
      last_.push_back(pool[order[r]]);
    }
    record(e, order);
    cma_update(params_, state_, ranked);
    if (state_.generation - state_.decomposed >= gap_) state_.decompose();
    reason_ = termination();
  }

  void record(const std::vector<heuresis::search::Evaluation>& e, const std::vector<std::size_t>& order) {
    const double best = e[order.front()].value(), worst = e[order.back()].value();
    const double median = e[order[order.size() / 2]].value();
    best_history_.push_back(best);
    median_history_.push_back(median);
    last_range_ = std::fabs(worst - best);
    const std::size_t keep = 10 + static_cast<std::size_t>(std::ceil(30.0 * static_cast<double>(state_.size()) / static_cast<double>(params_.lambda)));
    recent_best_.push_back(best);
    if (recent_best_.size() > keep) recent_best_.pop_front();
    if (best_history_.size() > 20000) { best_history_.pop_front(); median_history_.pop_front(); }
  }

  static double median_of(std::deque<double>::const_iterator b, std::deque<double>::const_iterator e) {
    std::vector<double> v(b, e);
    std::sort(v.begin(), v.end());
    return v.empty() ? 0 : (v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]));
  }

  CmaStop termination() const {
    const std::size_t n = state_.size();
    const auto& s = state_;
    if (s.generation >= options_.max_generations) return CmaStop::generations;
    // TolX
    bool small = true;
    for (std::size_t i = 0; i < n && small; i++) small = s.sigma * std::fabs(s.pc[i]) < options_.tol_x * sigma0_ && s.sigma * std::sqrt(s.C(i, i)) < options_.tol_x * sigma0_;
    if (small) return CmaStop::tol_x;
    // TolFun
    const std::size_t keep = 10 + static_cast<std::size_t>(std::ceil(30.0 * static_cast<double>(n) / static_cast<double>(params_.lambda)));
    if (recent_best_.size() >= keep) {
      const auto [lo, hi] = std::minmax_element(recent_best_.begin(), recent_best_.end());
      if (*hi - *lo < options_.tol_fun && last_range_ < options_.tol_fun) return CmaStop::tol_fun;
    }
    // ConditionCov
    if (s.condition() > options_.max_condition) return CmaStop::condition;
    // NoEffectAxis: 0.1σ along the principal axis of this generation leaves m unchanged.
    {
      const std::size_t a = s.generation % n;
      bool none = true;
      for (std::size_t i = 0; i < n && none; i++) none = s.mean[i] == s.mean[i] + 0.1 * s.sigma * s.D[a] * s.B(i, a);
      if (none) return CmaStop::no_effect_axis;
    }
    // NoEffectCoord: 0.2σ√C_ii leaves a coordinate of m unchanged.
    for (std::size_t i = 0; i < n; i++)
      if (s.mean[i] == s.mean[i] + 0.2 * s.sigma * std::sqrt(s.C(i, i))) return CmaStop::no_effect_coordinate;
    // Stagnation
    {
      const double gen = static_cast<double>(s.generation);
      const auto lo = static_cast<std::size_t>(std::ceil(120 + 30.0 * static_cast<double>(n) / static_cast<double>(params_.lambda)));
      const std::size_t window = std::min<std::size_t>(20000, std::max<std::size_t>(lo, static_cast<std::size_t>(std::ceil(0.2 * gen))));
      if (best_history_.size() >= window) {
        const std::size_t part = static_cast<std::size_t>(std::ceil(0.3 * static_cast<double>(window)));
        const auto b0 = best_history_.end() - static_cast<std::ptrdiff_t>(window), m0 = median_history_.end() - static_cast<std::ptrdiff_t>(window);
        const bool best_stale = median_of(best_history_.end() - static_cast<std::ptrdiff_t>(part), best_history_.end()) >= median_of(b0, b0 + static_cast<std::ptrdiff_t>(part));
        const bool median_stale = median_of(median_history_.end() - static_cast<std::ptrdiff_t>(part), median_history_.end()) >= median_of(m0, m0 + static_cast<std::ptrdiff_t>(part));
        if (best_stale && median_stale) return CmaStop::stagnation;
      }
    }
    // TolUpSigma
    if (s.sigma / sigma0_ > options_.tol_up_sigma * *std::max_element(s.D.begin(), s.D.end())) return CmaStop::tol_up_sigma;
    return CmaStop::none;
  }

  CmaOptions options_;
  CmaParameters params_;
  CmaState state_;
  heuresis::search::Draws draws_;
  heuresis::search::Batch<Real, Real> batch_;  // the note of each candidate: its sample in ℝⁿ
  std::optional<BoxTransform> transform_;
  std::uint64_t serial_ = 0, evaluations_ = 0;
  std::size_t gap_ = 1;
  double sigma0_ = 1, last_range_ = 0;
  CmaStop reason_ = CmaStop::none;
  std::vector<heuresis::search::Individual<Real>> injected_, last_;
  std::deque<double> recent_best_, best_history_, median_history_;
};

}  // namespace heuresis::strategies
