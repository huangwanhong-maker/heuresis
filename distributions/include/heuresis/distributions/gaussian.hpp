// Gaussian laws on ℝⁿ, and the algorithms that move them.
//
// N(m, C) is an exponential family with sufficient statistic T(x) = (x, xxᵀ) and expectation parameters
// (m, C + mmᵀ). In the coordinates (m, C) — C by its entries on and above the diagonal — its score and Fisher
// information are
//
//   ∂ ln p/∂m = C⁻¹(x − m),   ∂ ln p/∂C = tr(G E),  G = ½(C⁻¹(x − m)(x − m)ᵀC⁻¹ − C⁻¹),
//   I_mm = C⁻¹,               I_CC(E, E′) = ½ tr(C⁻¹ E C⁻¹ E′),   I_mC = 0,
//
// E, E′ the symmetric basis matrices of the entries (Magnus and Neudecker 2019, Matrix Differential Calculus, 3rd
// ed., ch. 15; Ollivier et al. 2017, §4.1). The natural gradient is then (x − m, (x − m)(x − m)ᵀ − C), so that the
// IGO step in (m, C) is the rank-μ update of CMA-ES with equal learning rates (Hansen 2016, "The CMA evolution
// strategy: a tutorial", arXiv:1604.00772, its rank-μ update without evolution paths), and in expectation parameters
// with δt = 1 and truncation weights it is the maximum-likelihood refit of the cross-entropy method:
//   m ← Σ ŵᵢ xᵢ,  C ← Σ ŵᵢ (xᵢ − m)(xᵢ − m)ᵀ about the new mean (Rubinstein 1999, "The cross-entropy method for
//   combinatorial and continuous optimization", Methodol. Comput. Appl. Probab. 1, 127–190; de Boer, Kroese, Mannor
//   and Rubinstein 2005, "A tutorial on the cross-entropy method", Ann. Oper. Res. 134, 19–67, with their smoothing).
// The two parametrisations give different steps at δt < 1 (they differ by δt² times the outer product of the mean's
// step); the natural gradient is invariant only to first order (Ollivier et al. 2017, §2.4).
//
// Sampling: x = m + L z, C = L Lᵀ by calx's Cholesky factorisation, z standard normal from the stream; O(n³) per
// point, adequate for moderate dimensions; CMA-ES (strategies/) keeps its own eigendecomposition.

#pragma once

#include <heuresis/distributions/family.hpp>
#include <heuresis/variation/real.hpp>

#include <cmath>
#include <numbers>
#include <stdexcept>
#include <vector>

namespace heuresis::distributions {

using heuresis::variation::Real;

struct GaussianParameters {
  Real m;
  calx::linalg::Matrix C;
};

struct Gaussian {
  using Point = Real;
  using Parameters = GaussianParameters;

  static calx::linalg::Matrix factor(const calx::linalg::Matrix& C) {
    const auto f = calx::linalg::cholesky(C);
    if (!f.positive_definite) throw std::domain_error("heuresis::distributions::Gaussian: C is not positive definite");
    return f.l;
  }
  static Point draw(const Real& m, const calx::linalg::Matrix& L, calx::random::Stream& s) {
    const std::size_t n = m.size();
    std::vector<double> z(n);
    for (auto& v : z) v = s.normal();
    Point x = m;
    for (std::size_t i = 0; i < n; i++)
      for (std::size_t j = 0; j <= i; j++) x[i] += L(i, j) * z[j];
    return x;
  }
  Point sample(const Parameters& t, calx::random::Stream& s) const { return draw(t.m, factor(t.C), s); }

  double log_density(const Parameters& t, const Point& x) const {
    const auto L = factor(t.C);
    const std::size_t n = t.m.size();
    std::vector<double> y(n);
    double logdet = 0, q = 0;
    for (std::size_t i = 0; i < n; i++) {
      double s = x[i] - t.m[i];
      for (std::size_t j = 0; j < i; j++) s -= L(i, j) * y[j];
      y[i] = s / L(i, i);
      q += y[i] * y[i];
      logdet += std::log(L(i, i));
    }
    return -0.5 * q - logdet - 0.5 * static_cast<double>(n) * std::log(2 * std::numbers::pi);
  }

  // T(x) = (x, the entries of xxᵀ on and above the diagonal), η = (m, the entries of C + mmᵀ).
  std::vector<double> statistic(const Point& x) const {
    std::vector<double> t(x.begin(), x.end());
    for (std::size_t i = 0; i < x.size(); i++)
      for (std::size_t j = i; j < x.size(); j++) t.push_back(x[i] * x[j]);
    return t;
  }
  std::vector<double> expectation(const Parameters& p) const {
    std::vector<double> e(p.m.begin(), p.m.end());
    for (std::size_t i = 0; i < p.m.size(); i++)
      for (std::size_t j = i; j < p.m.size(); j++) e.push_back(p.C(i, j) + p.m[i] * p.m[j]);
    return e;
  }
  Parameters from_expectation(std::vector<double> eta) const {
    Parameters p{Real(eta.begin(), eta.begin() + static_cast<std::ptrdiff_t>(n)), calx::linalg::Matrix(n)};
    std::size_t k = n;
    for (std::size_t i = 0; i < n; i++)
      for (std::size_t j = i; j < n; j++, k++) p.C(i, j) = p.C(j, i) = eta[k] - p.m[i] * p.m[j];
    return p;
  }

  // Coordinates (m, the entries of C on and above the diagonal), their score and Fisher information.
  std::vector<double> coordinates(const Parameters& p) const {
    std::vector<double> c(p.m.begin(), p.m.end());
    for (std::size_t i = 0; i < n; i++)
      for (std::size_t j = i; j < n; j++) c.push_back(p.C(i, j));
    return c;
  }
  Parameters from_coordinates(std::vector<double> c) const {
    Parameters p{Real(c.begin(), c.begin() + static_cast<std::ptrdiff_t>(n)), calx::linalg::Matrix(n)};
    std::size_t k = n;
    for (std::size_t i = 0; i < n; i++)
      for (std::size_t j = i; j < n; j++, k++) p.C(i, j) = p.C(j, i) = c[k];
    return p;
  }
  std::vector<double> score(const Parameters& p, const Point& x) const {
    const auto A = calx::linalg::solve(p.C, calx::linalg::Matrix::identity(n));  // C⁻¹
    std::vector<double> d(n), Ad(n, 0.0);
    for (std::size_t i = 0; i < n; i++) d[i] = x[i] - p.m[i];
    for (std::size_t i = 0; i < n; i++)
      for (std::size_t j = 0; j < n; j++) Ad[i] += A(i, j) * d[j];
    std::vector<double> g(Ad);
    for (std::size_t i = 0; i < n; i++)
      for (std::size_t j = i; j < n; j++) {
        const double Gij = 0.5 * (Ad[i] * Ad[j] - A(i, j));
        g.push_back(i == j ? Gij : 2 * Gij);  // G is symmetric: tr(G E_ij) = G_ji + G_ij
      }
    return g;
  }
  calx::linalg::Matrix fisher(const Parameters& p) const {
    const auto A = calx::linalg::solve(p.C, calx::linalg::Matrix::identity(n));
    const std::size_t d = n + n * (n + 1) / 2;
    calx::linalg::Matrix I(d);
    for (std::size_t i = 0; i < n; i++)
      for (std::size_t j = 0; j < n; j++) I(i, j) = A(i, j);
    // The entries (i, j), i ≤ j, as lists of the unit matrices e_a e_bᵀ that make up E_ij.
    std::vector<std::vector<std::pair<std::size_t, std::size_t>>> E;
    for (std::size_t i = 0; i < n; i++)
      for (std::size_t j = i; j < n; j++) E.push_back(i == j ? std::vector<std::pair<std::size_t, std::size_t>>{{i, i}} : std::vector<std::pair<std::size_t, std::size_t>>{{i, j}, {j, i}});
    for (std::size_t u = 0; u < E.size(); u++)
      for (std::size_t v = 0; v < E.size(); v++) {
        double t = 0;  // tr(A e_a e_bᵀ A e_c e_dᵀ) = A_bc A_da
        for (auto [a, b] : E[u])
          for (auto [c, dd] : E[v]) t += A(b, c) * A(dd, a);
        I(n + u, n + v) = 0.5 * t;
      }
    return I;
  }

  static Parameters isotropic(Real m, double sigma) {
    Parameters p{std::move(m), calx::linalg::Matrix(0)};
    p.C = calx::linalg::Matrix::identity(p.m.size());
    for (auto& v : p.C.a) v *= sigma * sigma;
    return p;
  }

  std::size_t n = 0;  // the dimension
};

// The rank-μ update of CMA-ES with equal learning rates: the IGO step in (m, C), in closed form.
struct RankMu {
  std::vector<double> weights;  // by rank, best first; Σ = 1 for the classical update
  double rate = 1;              // δt
  GaussianParameters operator()(const Gaussian& f, const GaussianParameters& p, std::span<const Real> ranked, std::span<const Evaluation> evaluations) const {
    const std::vector<double> w = tied_weights(evaluations, weights);
    const std::size_t n = f.n;
    GaussianParameters q = p;
    for (std::size_t k = 0; k < ranked.size(); k++) {
      if (w[k] == 0) continue;
      for (std::size_t i = 0; i < n; i++) {
        const double di = ranked[k][i] - p.m[i];
        q.m[i] += rate * w[k] * di;
        for (std::size_t j = 0; j < n; j++) q.C(i, j) += rate * w[k] * (di * (ranked[k][j] - p.m[j]) - p.C(i, j));
      }
    }
    return q;
  }
};

// The cross-entropy method's smoothed refit: m ← α·m̂ + (1 − α)·m and C ← β_t·Ĉ + (1 − β_t)·C, (m̂, Ĉ) the weighted
// maximum-likelihood fit to the elite. With a constant β_t = α the covariance shrinks geometrically, faster than the
// mean can travel on a slope, and the search stalls short of the optimum (premature convergence: Grahl, Bosman and
// Rothlauf 2006, "The correlation-triggered adaptive variance scaling IDEA", GECCO, 397–404; the checks show it).
// Kroese, Porotsky and Rubinstein's dynamic smoothing (2006, "The cross-entropy method for continuous multi-extremal
// optimization", Methodol. Comput. Appl. Probab. 8, 383–407) takes β_t = β − β(1 − 1/t)^q at generation t ≥ 1, so that
// β_t ≈ βq/t and the covariance decays only polynomially. The rule counts its generations, so it is stateful.
struct CrossEntropy {
  std::size_t elite;          // the number of points refitted
  double smoothing = 1;       // α, for the mean (and the covariance when beta = 0)
  double beta = 0;            // dynamic smoothing of the covariance; 0: off
  double q = 7;
  double variance_floor = 0;  // added to the diagonal, so that the law never collapses to a singular one
  std::size_t generation = 0;
  GaussianParameters operator()(const Gaussian& f, const GaussianParameters& p, std::span<const Real> ranked, std::span<const Evaluation>) {
    if (elite < 2 || elite > ranked.size()) throw std::invalid_argument("heuresis::distributions::CrossEntropy: 2 ≤ elite ≤ λ");
    const std::size_t n = f.n;
    generation++;
    const double t = static_cast<double>(generation);
    const double bc = beta > 0 ? beta - beta * std::pow(1 - 1 / t, q) : smoothing;
    Real m(n, 0.0);
    for (std::size_t k = 0; k < elite; k++)
      for (std::size_t i = 0; i < n; i++) m[i] += ranked[k][i] / static_cast<double>(elite);
    calx::linalg::Matrix C(n);
    for (std::size_t k = 0; k < elite; k++)
      for (std::size_t i = 0; i < n; i++)
        for (std::size_t j = 0; j < n; j++) C(i, j) += (ranked[k][i] - m[i]) * (ranked[k][j] - m[j]) / static_cast<double>(elite);
    GaussianParameters r{Real(n), calx::linalg::Matrix(n)};
    for (std::size_t i = 0; i < n; i++) {
      r.m[i] = smoothing * m[i] + (1 - smoothing) * p.m[i];
      for (std::size_t j = 0; j < n; j++) r.C(i, j) = bc * C(i, j) + (1 - bc) * p.C(i, j);
      r.C(i, i) += variance_floor;
    }
    return r;
  }
};

}  // namespace heuresis::distributions
