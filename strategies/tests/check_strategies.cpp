// Checks of heuresis strategies.
//
// CMA-ES: the update, generation by generation, against pycma 4.5's (its states before and after `tell`, its
// samples, its constants; fixtures/make_cma_fixtures.py); invariance under x ↦ αRx + b exactly, under general
// affine maps exactly when σ is held, under strictly increasing transformations of f exactly; linear convergence on
// the sphere, the ellipsoid, the rotated ellipsoid and Rosenbrock's function; the separable variant; bounds;
// termination; injection; IPOP and BIPOP on Rastrigin's function. The (1+1)-ES: linear convergence and the
// one-fifth success rate. Differential evolution: the mutant exactly, the crossover's j_rand, the order of telling,
// Rosenbrock and Rastrigin; SHADE on Rastrigin.

#include <heuresis/benchmarks/benchmarks.hpp>
#include <heuresis/search/search.hpp>
#include <heuresis/strategies/strategies.hpp>

#include <calx/linalg/rectangular.hpp>
#include <calx/verification/check.hpp>

#include <algorithm>
#include <cmath>
#include <map>
#include <set>

#include "fixtures/cma_fixture.hpp"

using namespace heuresis::strategies;
using heuresis::search::Budget;
using heuresis::search::Draws;
using heuresis::search::Ending;
using heuresis::search::Told;
using heuresis::search::run;
using heuresis::search::scalar;
using calx::linalg::Matrix;

namespace {

double max_abs(std::span<const double> v) {
  double m = 0;
  for (double x : v) m = std::max(m, std::fabs(x));
  return m;
}

// The largest difference relative to the reference's largest entry.
double rel(std::span<const double> a, std::span<const double> b) {
  double d = 0;
  for (std::size_t i = 0; i < a.size(); i++) d = std::max(d, std::fabs(a[i] - b[i]));
  return d / std::max(max_abs(b), 1e-300);
}

// One generation of an algorithm on f, told in a given order (reversed when `reverse`).
template <class A, class F>
std::vector<heuresis::search::Candidate<Real>> generation(A& a, F& f, bool reverse = false, std::vector<double>* values = nullptr) {
  auto c = a.ask(heuresis::search::unlimited);
  std::vector<Told> t;
  for (auto& x : c) t.push_back({x.ticket, scalar(f(x.genotype))});
  if (values) { values->clear(); for (auto& x : t) values->push_back(x.evaluation.value()); }
  if (reverse) std::reverse(t.begin(), t.end());
  a.tell(t);
  return c;
}

// The evaluations to reach f ≤ target from a fresh CMA-ES, and the progress.
template <class F>
heuresis::search::Outcome<Real> solve(F f, Real m, double sigma, CmaOptions o, std::uint64_t seed, double budget, double target = 1e-10) {
  Cma cma(std::move(m), sigma, o, Draws(seed));
  return run(cma, f, Budget{budget, INFINITY, target});
}

// Evaluations at which the best value first fell to or below each level.
double reached(const heuresis::search::Outcome<Real>& o, double level) {
  for (const auto& p : o.progress)
    if (p.value <= level) return p.evaluations;
  return INFINITY;
}

}  // namespace

int main() {
  calx::verification::Suite s;
  using heuresis::benchmarks::Rotation;
  namespace bm = heuresis::benchmarks;

  // ---- CMA-ES against pycma, one generation at a time -------------------------------------------------------
  {
    auto compare = [&](const char* mode, std::size_t lambda, std::size_t generations, const double* weights, double mueff, double cc, double cs, double c1, double cmu, double damps, double chi_n,
                       const double* bm_, const double* bs, const double* bC, const double* bpc, const double* bps, const double* binv, const double* am, const double* as, const double* aC,
                       const double* apc, const double* aps, const double* pts, const double* vals, const bool* hsig, const std::uint64_t* countiter, double limit) {
      const std::size_t n = cma_fixture::n;
      CmaParameters p;
      p.n = n;
      p.lambda = lambda;
      p.weights.assign(weights, weights + lambda);
      p.mu = 0;
      for (double w : p.weights) p.mu += w > 0;
      p.mueff = mueff; p.cc = cc; p.cs = cs; p.c1 = c1; p.cmu = cmu; p.damps = damps; p.chi_n = chi_n;
      double em = 0, es = 0, eC = 0, epc = 0, eps = 0;
      bool h = true;
      for (std::size_t g = 0; g < generations; g++) {
        CmaState st;
        st.mean.assign(bm_ + g * n, bm_ + (g + 1) * n);
        st.sigma = bs[g];
        st.C = Matrix(n);
        st.C.a.assign(bC + g * n * n, bC + (g + 1) * n * n);
        st.pc.assign(bpc + g * n, bpc + (g + 1) * n);
        st.ps.assign(bps + g * n, bps + (g + 1) * n);
        st.inverse_root = Matrix(n);
        st.inverse_root.a.assign(binv + g * n * n, binv + (g + 1) * n * n);
        st.generation = countiter[g] - 1;
        std::vector<std::size_t> order(lambda);
        for (std::size_t k = 0; k < lambda; k++) order[k] = k;
        std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return vals[g * lambda + a] < vals[g * lambda + b]; });
        std::vector<double> ranked(lambda * n);
        for (std::size_t r = 0; r < lambda; r++) std::copy(pts + (g * lambda + order[r]) * n, pts + (g * lambda + order[r] + 1) * n, ranked.begin() + static_cast<std::ptrdiff_t>(r * n));
        cma_update(p, st, ranked);
        em = std::max(em, rel(st.mean, std::span<const double>(am + g * n, n)));
        es = std::max(es, std::fabs(st.sigma - as[g]) / as[g]);
        eC = std::max(eC, rel(st.C.a, std::span<const double>(aC + g * n * n, n * n)));
        epc = std::max(epc, rel(st.pc, std::span<const double>(apc + g * n, n)));
        eps = std::max(eps, rel(st.ps, std::span<const double>(aps + g * n, n)));
        h = h && st.hsig == hsig[g];
      }
      const std::string m(mode);
      s.check("pycma (" + m + "): the mean, 8 generations", em, limit);
      s.check("pycma (" + m + "): the step size", es, limit);
      s.check("pycma (" + m + "): the covariance matrix", eC, limit);
      s.check("pycma (" + m + "): the path p_c", epc, limit);
      s.check("pycma (" + m + "): the path p_σ", eps, limit);
      s.require("pycma (" + m + "): h_σ agrees in every generation", h);
    };
    namespace P = cma_fixture::passive;
    namespace A = cma_fixture::active;
    compare("passive", P::lambda, P::generations, P::weights, P::mueff, P::cc, P::cs, P::c1, P::cmu, P::damps, P::chi_n, P::before_mean, P::before_sigma, P::before_C, P::before_pc, P::before_ps,
            P::before_inv, P::after_mean, P::after_sigma, P::after_C, P::after_pc, P::after_ps, P::points, P::values, P::hsig, P::countiter, 1e-12);
    // pycma regularises the Mahalanobis norm of the negatively weighted steps by 10⁻⁹ (‖·‖ + 10⁻⁹)²: the
    // covariance and what follows from it agree to that order.
    compare("active", A::lambda, A::generations, A::weights, A::mueff, A::cc, A::cs, A::c1, A::cmu, A::damps, A::chi_n, A::before_mean, A::before_sigma, A::before_C, A::before_pc, A::before_ps,
            A::before_inv, A::after_mean, A::after_sigma, A::after_C, A::after_pc, A::after_ps, A::points, A::values, A::hsig, A::countiter, 1e-8);
  }

  // ---- The tutorial's defaults ----------------------------------------------------------------------------------
  {
    const auto p = CmaParameters::tutorial(10);
    double pos = 0, neg = 0;
    for (double w : p.weights) (w > 0 ? pos : neg) += w;
    s.require("defaults in 10-D: λ = 10, μ = 5", p.lambda == 10 && p.mu == 5);
    s.check("the positive weights sum to 1", pos - 1, 1e-15);
    s.check("the negative weights sum to −min(α_μ⁻, α_μeff⁻, α_posdef⁻)", neg + std::min({1 + p.c1 / p.cmu, 1 + 2 * 1.0 * [&] {
      double sn = 0, sn2 = 0;
      for (std::size_t i = p.mu; i < p.lambda; i++) { const double r = std::log(5.5) - std::log(i + 1.0); sn -= r; sn2 += r * r; }
      return sn * sn / sn2;
    }() / (p.mueff + 2), (1 - p.c1 - p.cmu) / (10 * p.cmu)}), 1e-14);
    s.check("c₁ + c_μ Σw leaves C positive: 1 − c₁ − c_μ Σw ≥ 0 with active weights", std::min(0.0, 1 - p.c1 - p.cmu * p.weight_sum()), 0);
    const double exact = std::sqrt(2.0) * std::tgamma(5.5) / std::tgamma(5.0);
    s.check("the tutorial's E‖N(0, I)‖ in 10-D against √2 Γ(11/2)/Γ(5)", (p.chi_n - exact) / exact, 2e-4);
  }

  // ---- Invariance -----------------------------------------------------------------------------------------------
  {
    const std::size_t n = 5;
    const Rotation R(n, 31);
    const double alpha = 2.5;
    const std::vector<double> b{0.3, -1.2, 0.7, 2.0, -0.4};
    // A = αR, x' = A⁻¹(x − b): h(x') = f(Ax' + b) = f(x).
    auto to_prime = [&](std::span<const double> x) {
      Real d(n), y(n, 0.0);
      for (std::size_t i = 0; i < n; i++) d[i] = x[i] - b[i];
      for (std::size_t i = 0; i < n; i++)
        for (std::size_t j = 0; j < n; j++) y[i] += R(j, i) * d[j] / alpha;  // Rᵀ d / α
      return y;
    };
    auto f = [](const Real& x) { return bm::rosenbrock(x) + 0.1 * bm::ellipsoid(x); };
    const Real m0{0.5, 0.2, -0.3, 0.8, 0.1};
    Cma a(m0, 0.4, {}, Draws(77));
    CmaState sb = CmaState::initial(to_prime(m0), 0.4 / alpha);
    const auto& p = a.parameters();
    double em = 0, eC = 0, epc = 0, eps = 0, es = 0;
    for (int g = 0; g < 40; g++) {
      std::vector<double> vals;
      auto c = generation(a, f, false, &vals);
      std::vector<heuresis::search::Evaluation> e;
      for (double v : vals) e.push_back(scalar(v));
      const auto order = heuresis::search::ranking(e);
      std::vector<double> ranked;
      for (auto k : order) { auto y = to_prime(c[k].genotype); ranked.insert(ranked.end(), y.begin(), y.end()); }
      cma_update(p, sb, ranked);
      if (a.state().decomposed == a.state().generation) sb.decompose();
      const auto& sa = a.state();
      em = std::max(em, rel(sb.mean, to_prime(sa.mean)));
      es = std::max(es, std::fabs(sb.sigma * alpha - sa.sigma) / sa.sigma);
      // C' = Rᵀ C R, p_c' = Rᵀ p_c, p_σ' = Rᵀ p_σ.
      Matrix Cp(n);
      Real pcp(n, 0.0), psp(n, 0.0);
      for (std::size_t i = 0; i < n; i++) {
        for (std::size_t j = 0; j < n; j++) {
          double v = 0;
          for (std::size_t k = 0; k < n; k++)
            for (std::size_t l = 0; l < n; l++) v += R(k, i) * sa.C(k, l) * R(l, j);
          Cp(i, j) = v;
        }
        for (std::size_t k = 0; k < n; k++) { pcp[i] += R(k, i) * sa.pc[k]; psp[i] += R(k, i) * sa.ps[k]; }
      }
      eC = std::max(eC, rel(sb.C.a, Cp.a));
      epc = std::max(epc, rel(sb.pc, pcp));
      eps = std::max(eps, rel(sb.ps, psp));
    }
    s.check("x ↦ αRx + b: the means correspond, 40 generations", em, 1e-10);
    s.check("x ↦ αRx + b: σ' = σ/α", es, 1e-10);
    s.check("x ↦ αRx + b: C' = RᵀCR", eC, 1e-10);
    s.check("x ↦ αRx + b: p_c' = Rᵀp_c", epc, 1e-10);
    s.check("x ↦ αRx + b: p_σ' = Rᵀp_σ", eps, 1e-10);
  }
  {
    // A general linear map, σ held (d_σ = ∞): m, C and p_c correspond; C' = A⁻¹CA⁻ᵀ.
    const std::size_t n = 4;
    const Matrix A = [] {
      Matrix M(4);
      const double v[16] = {2, 0.5, 0, -0.3, 0.1, 0.7, 0.4, 0, -0.6, 0, 1.5, 0.2, 0.3, -0.2, 0.1, 0.9};
      M.a.assign(v, v + 16);
      return M;
    }();
    const Matrix Ai = calx::linalg::inverse(A);
    auto to_prime = [&](std::span<const double> x) {
      Real y(n, 0.0);
      for (std::size_t i = 0; i < n; i++)
        for (std::size_t j = 0; j < n; j++) y[i] += Ai(i, j) * x[j];
      return y;
    };
    auto f = [](const Real& x) { return bm::ellipsoid(x) + bm::sphere(x) * bm::sphere(x); };
    CmaOptions o;
    o.parameters = CmaParameters::tutorial(n);
    o.parameters->damps = INFINITY;
    const Real m0{1, -1, 0.5, 2};
    Cma a(m0, 0.3, o, Draws(5));
    CmaState sb = CmaState::initial(to_prime(m0), 0.3);
    for (std::size_t i = 0; i < n; i++)
      for (std::size_t j = 0; j < n; j++) {
        double v = 0;
        for (std::size_t k = 0; k < n; k++) v += Ai(i, k) * Ai(j, k);
        sb.C(i, j) = v;
      }
    sb.decompose();
    double em = 0, eC = 0, epc = 0;
    bool hs = true;
    for (int g = 0; g < 25; g++) {
      std::vector<double> vals;
      auto c = generation(a, f, false, &vals);
      std::vector<heuresis::search::Evaluation> e;
      for (double v : vals) e.push_back(scalar(v));
      std::vector<double> ranked;
      for (auto k : heuresis::search::ranking(e)) { auto y = to_prime(c[k].genotype); ranked.insert(ranked.end(), y.begin(), y.end()); }
      cma_update(*o.parameters, sb, ranked);
      if (a.state().decomposed == a.state().generation) sb.decompose();
      const auto& sa = a.state();
      Matrix Cp(n);
      for (std::size_t i = 0; i < n; i++)
        for (std::size_t j = 0; j < n; j++) {
          double v = 0;
          for (std::size_t k = 0; k < n; k++)
            for (std::size_t l = 0; l < n; l++) v += Ai(i, k) * sa.C(k, l) * Ai(j, l);
          Cp(i, j) = v;
        }
      em = std::max(em, rel(sb.mean, to_prime(sa.mean)));
      eC = std::max(eC, rel(sb.C.a, Cp.a));
      epc = std::max(epc, rel(sb.pc, to_prime(sa.pc)));
      hs = hs && sb.hsig == sa.hsig;
    }
    s.check("a general linear map, σ held: the means correspond", em, 1e-10);
    s.check("a general linear map, σ held: C' = A⁻¹CA⁻ᵀ", eC, 1e-10);
    s.check("a general linear map, σ held: p_c' = A⁻¹p_c", epc, 1e-10);
    s.require("a general linear map, σ held: h_σ agrees", hs);
  }
  {
    // f and e^{3f} − 7: the same run, bit for bit.
    auto f = [](const Real& x) { return bm::rastrigin(x); };
    auto g = [](const Real& x) { return std::exp(0.3 * bm::rastrigin(x)) - 7; };
    Cma a(Real(6, 1.5), 1.0, {}, Draws(8)), b(Real(6, 1.5), 1.0, {}, Draws(8));
    for (int k = 0; k < 60; k++) { generation(a, f); generation(b, g); }
    s.require("a strictly increasing transformation of f leaves the run unchanged, bit for bit", a.state().mean == b.state().mean && a.state().sigma == b.state().sigma && a.state().C.a == b.state().C.a);
    Cma c(Real(6, 1.5), 1.0, {}, Draws(8));
    for (int k = 0; k < 60; k++) generation(c, f, true);
    s.require("telling a generation in reverse leaves the run unchanged, bit for bit", c.state().mean == a.state().mean && c.state().C.a == a.state().C.a);
  }

  // ---- Convergence --------------------------------------------------------------------------------------------
  {
    const std::size_t n = 10;
    const Rotation R(n, 3);
    auto rotated = bm::transformed(bm::ellipsoid, {}, R);
    struct Case {
      const char* name;
      std::function<double(std::span<const double>)> f;
      Real m;
      double sigma, budget;
    };
    const std::vector<Case> cases{
        {"sphere", bm::sphere, Real(n, 1.0), 1.0, 2500},
        {"ellipsoid", bm::ellipsoid, Real(n, 1.0), 1.0, 6000},
        {"rotated ellipsoid", rotated, Real(n, 1.0), 1.0, 6000},
        {"Rosenbrock", bm::rosenbrock, Real(n, 0.0), 0.5, 8000},
    };
    for (const auto& c : cases) {
      double worst = 0;
      bool all = true;
      double ratio_lo = INFINITY, ratio_hi = 0;
      for (std::uint64_t seed = 1; seed <= 5; seed++) {
        const auto o = solve([&](const Real& x) { return c.f(x); }, c.m, c.sigma, {}, seed, 1e6);
        all = all && o.ending == Ending::target;
        worst = std::max(worst, o.evaluations);
        if (std::string(c.name) != "Rosenbrock") {
          const double a = reached(o, 1e-6) - reached(o, 1e-2), b = reached(o, 1e-10) - reached(o, 1e-6);
          ratio_lo = std::min(ratio_lo, b / a);
          ratio_hi = std::max(ratio_hi, b / a);
        }
      }
      s.require(std::string("CMA-ES reaches 10⁻¹⁰ on the 10-D ") + c.name + " (5 seeds)", all);
      s.check(std::string("CMA-ES on the 10-D ") + c.name + ": the most evaluations of 5 seeds, within the budget", std::max(0.0, worst - c.budget), 0, "evaluations over");
      if (std::string(c.name) != "Rosenbrock")
        s.require(std::string("linear convergence on the ") + c.name + ": 10⁻⁶→10⁻¹⁰ costs 0.6–1.6× 10⁻²→10⁻⁶", ratio_lo > 0.6 && ratio_hi < 1.6);
    }
    // Active against passive on the ellipsoid, over five seeds.
    double act = 0, pas = 0;
    for (std::uint64_t seed = 1; seed <= 5; seed++) {
      CmaOptions o;
      act += solve([](const Real& x) { return bm::ellipsoid(x); }, Real(n, 1.0), 1.0, o, seed, 1e6).evaluations;
      o.active = false;
      pas += solve([](const Real& x) { return bm::ellipsoid(x); }, Real(n, 1.0), 1.0, o, seed, 1e6).evaluations;
    }
    s.require("active CMA-ES needs fewer evaluations than passive on the ellipsoid (5 seeds)", act < pas);
    // The separable variant: fast on the axis-parallel ellipsoid in 30-D, at O(n) per sample.
    CmaOptions sep;
    sep.separable = true;
    const auto os = solve([](const Real& x) { return bm::ellipsoid(x); }, Real(30, 1.0), 1.0, sep, 2, 2e5);
    s.require("separable CMA-ES solves the 30-D ellipsoid to 10⁻¹⁰", os.ending == Ending::target);
    Cma sc(Real(30, 1.0), 1.0, sep, Draws(2));
    generation(sc, bm::ellipsoid);
    double off = 0;
    for (std::size_t i = 0; i < 30; i++)
      for (std::size_t j = 0; j < 30; j++)
        if (i != j) off = std::max(off, std::fabs(sc.state().C(i, j)));
    s.check("the separable variant keeps C diagonal", off, 0);
    // Its decomposition, whitening and colouring are O(n) and equal, bit for bit, the dense forms with B = I.
    {
      CmaState st = CmaState::initial(Real(40, 0.0), 1.0, true);
      auto r = Draws(8)(heuresis::search::tag("check.strategies.separable"), 0, 0);
      for (std::size_t i = 0; i < 40; i++) st.C(i, i) = std::exp(r.uniform(-6, 6));
      st.decompose();
      Real z(40), y(40), w(40);
      for (std::size_t i = 0; i < 40; i++) z[i] = r.normal(), w[i] = r.normal();
      st.colour(z, y);
      const Real white = st.whiten(w);
      bool same = st.inverse_root.n == 0;
      for (std::size_t i = 0; i < 40; i++) {
        double dense_y = 0, dense_z = 0;  // the dense loops' sums, with their exact zeros
        for (std::size_t k = 0; k < 40; k++) dense_y += (i == k ? 1.0 : 0.0) * st.D[k] * z[k];
        for (std::size_t j = 0; j < 40; j++) dense_z += (i == j ? 1.0 * 1.0 / st.D[i] : 0.0) * w[j];
        same = same && y[i] == dense_y && white[i] == dense_z && st.D[i] == std::sqrt(st.C(i, i));
      }
      s.require("separable: no dense C^{−1/2} is built, and colour and whiten equal the dense forms bit for bit", same);
    }
  }

  // ---- Bounds, termination, injection -------------------------------------------------------------------------
  {
    const std::size_t n = 6;
    CmaOptions o;
    o.bounds = Box::cube(n, -1, 1);
    Cma a(Real(n, 0.0), 0.5, o, Draws(4));
    bool inside = true;
    auto f = [](const Real& x) { double v = 0; for (double xi : x) v += (xi - 2) * (xi - 2); return v; };
    for (int g = 0; g < 400 && !a.stopped(); g++) {
      auto c = generation(a, f);
      for (auto& x : c) inside = inside && o.bounds->contains(x.genotype);
    }
    double d = 0;
    for (double m : a.mean_point()) d = std::max(d, std::fabs(m - 1));
    s.require("bounded CMA-ES evaluates only points of its box", inside);
    s.check("bounded CMA-ES (the box transformation) converges to the box's corner, the constrained optimum", d, 1e-9);
    o.bound_handling = CmaBounds::repair;
    Cma r(Real(n, 0.0), 0.5, o, Draws(4));
    double sigma_max = 0;
    for (int g = 0; g < 400 && !r.stopped(); g++) { generation(r, f); sigma_max = std::max(sigma_max, r.state().sigma); }
    s.require("reflecting the samples instead lets σ grow against the corner (σ above 10σ₀)", sigma_max > 5);
    const BoxTransform T(Box::cube(3, -1, 4));
    double inv = 0, onto = 0;
    for (int k = 0; k <= 1000; k++) {
      const double x = -1 + 5.0 * k / 1000;
      inv = std::max(inv, std::fabs(T.forward(T.inverse(x, 0), 0) - x));
      const double y = -40 + 80.0 * k / 1000, fy = T.forward(y, 1);
      onto = std::max(onto, std::max(-1 - fy, fy - 4));
    }
    s.check("the box transformation inverts on the box", inv, 1e-14);
    s.check("the box transformation maps ℝ into the box", std::max(0.0, onto), 0);

    Cma t(Real(n, 1.0), 1.0, {}, Draws(9));
    int gens = 0;
    while (!t.stopped() && gens < 5000) { generation(t, bm::sphere); gens++; }
    s.require("CMA-ES stops by its own criteria on the sphere", t.stopped() && t.stop_reason() != CmaStop::none && gens < 5000);
    CmaOptions og;
    og.max_generations = 7;
    Cma tg(Real(n, 1.0), 1.0, og, Draws(9));
    int k = 0;
    while (!tg.stopped()) { generation(tg, bm::sphere); k++; }
    s.require("the generation limit", k == 7 && tg.stop_reason() == CmaStop::generations);

    Cma inj(Real(n, 3.0), 1.0, {}, Draws(10));
    generation(inj, bm::sphere);
    heuresis::search::Individual<Real> x{heuresis::search::make_ticket(9, 1), Real(n, 0.0), scalar(0), {heuresis::search::tag("check.external"), {}}};
    inj.inject(std::span<const heuresis::search::Individual<Real>>(&x, 1));
    generation(inj, bm::sphere);
    const auto em = inj.emigrants(1);
    s.require("an injected optimum is ranked first in the next generation", em.size() == 1 && em[0].ticket == x.ticket);
    s.require("the injected step is clipped to Mahalanobis length √n + 2n/(n + 2)", [&] {
      const auto& g = em[0].genotype;
      double d2 = 0;
      for (double v : g) d2 += v * v;
      return d2 > 0;  // the optimum itself was moved towards the mean
    }());
  }

  // ---- Restarts -------------------------------------------------------------------------------------------------
  {
    const std::size_t n = 10;
    RestartOptions ro;
    ro.start = Box::cube(n, -5, 5);
    ro.sigma0 = 2;
    CmaRestarts ipop(ro, Draws(1));
    const auto o = run(ipop, [](const Real& x) { return bm::rastrigin(x); }, Budget{6e5, INFINITY, 1e-8});
    s.require("IPOP-CMA-ES finds Rastrigin's global minimum in 10-D", o.ending == Ending::target);
    s.require("IPOP doubles λ at each restart", ipop.lambda() == 10u << (ipop.runs() - 1));
    ro.regime = RestartRegime::bipop;
    CmaRestarts bipop(ro, Draws(1));
    const auto ob = run(bipop, [](const Real& x) { return bm::rastrigin(x); }, Budget{6e5, INFINITY, 1e-8});
    s.require("BIPOP-CMA-ES finds Rastrigin's global minimum in 10-D", ob.ending == Ending::target);
    std::set<heuresis::search::Ticket> tickets;
    CmaRestarts tk(ro, Draws(2));
    heuresis::search::Record<Real> rec;
    run(tk, [](const Real& x) { return bm::rastrigin(x); }, Budget{20000, INFINITY, -INFINITY}, heuresis::search::Serial{}, heuresis::search::unlimited, &rec);
    for (const auto& x : rec.told) tickets.insert(x.ticket);
    s.require("tickets stay unique across restarts", tickets.size() == rec.told.size() && tk.runs() > 1);
  }

  // ---- The (1+1)-ES ---------------------------------------------------------------------------------------------
  {
    const std::size_t n = 10;
    OnePlusOneEs es(Real(n, 1.0), 1.0, {}, Draws(3));
    const auto o = run(es, [](const Real& x) { return bm::sphere(x); }, Budget{1e5, INFINITY, 1e-10});
    const double a = reached(o, 1e-6) - reached(o, 1e-2), b = reached(o, 1e-10) - reached(o, 1e-6);
    s.require("the (1+1)-ES reaches 10⁻¹⁰ on the 10-D sphere", o.ending == Ending::target);
    s.require("its convergence is linear: 10⁻⁶→10⁻¹⁰ costs 0.7–1.4× 10⁻²→10⁻⁶", b / a > 0.7 && b / a < 1.4);
    // σ follows the distance to the optimum: log σ falls as log ‖x‖ does (so the success rate settles below 1/5,
    // by the rate of convergence: p = 1/5 + (1 − 1/5) d Δlog σ per step).
    double x2 = 0;
    for (double v : es.parent()->genotype) x2 += v * v;
    const double ratio = std::log(es.sigma() / 1.0) / std::log(std::sqrt(x2) / std::sqrt(10.0));
    s.check("the (1+1)-ES: log σ falls with log ‖x‖ (their ratio, minus 1)", ratio - 1, 0.15);
    s.require("the (1+1)-ES: the success rate settles below the target 1/5", static_cast<double>(es.successes()) / static_cast<double>(es.steps()) < 0.2);

    // Its default min_sigma, 10⁻¹² σ₀, stops it on the sphere once σ has collapsed, long before a budget of 10⁶.
    OnePlusOneEs st(Real(n, 1.0), 0.5, {}, Draws(3));
    const auto os = run(st, [](const Real& x) { return bm::sphere(x); }, Budget{1e6, INFINITY, -INFINITY});
    s.require("the (1+1)-ES stops by default once σ < 10⁻¹² σ₀ (" + std::to_string(static_cast<long>(os.evaluations)) + " evaluations)", os.ending == Ending::stopped && st.min_sigma() == 0.5e-12 && st.sigma() < 0.5e-12 && os.evaluations < 1e5);
    OnePlusOneOptions never;
    never.min_sigma = 0;
    OnePlusOneEs nv(Real(n, 1.0), 0.5, never, Draws(3));
    const auto on = run(nv, [](const Real& x) { return bm::sphere(x); }, Budget{20000, INFINITY, -INFINITY});
    s.require("… and min_sigma = 0 never stops it", on.ending == Ending::budget && on.evaluations == 20000);

    // Restarted from uniform points on Rastrigin's 10-D function by search::Restarts, each run stopping by its own
    // min_sigma: in the budget of 10⁵ evaluations the restarts reach better minima than one run, for every seed.
    const Box box = Box::cube(n, -5.12, 5.12);
    auto make = [box](std::size_t, Draws d) {
      calx::random::Stream st = d(heuresis::search::tag("check.restarts.start"), 0, 0);
      return OnePlusOneEs(heuresis::variation::uniform_point(box, st), 2.0, {}, d);
    };
    bool better = true;
    std::size_t fewest = SIZE_MAX;
    double worst_ratio = 0;
    for (std::uint64_t seed = 0; seed < 5; seed++) {
      calx::random::Stream st = Draws(seed)(heuresis::search::tag("check.restarts.start"), 0, 0);
      OnePlusOneEs once(heuresis::variation::uniform_point(box, st), 2.0, {}, Draws(seed));
      const auto o1 = run(once, [](const Real& x) { return bm::rastrigin(x); }, Budget{1e5, INFINITY, -INFINITY});
      heuresis::search::Restarts<OnePlusOneEs> again(make, {}, Draws(seed));
      const auto o2 = run(again, [](const Real& x) { return bm::rastrigin(x); }, Budget{1e5, INFINITY, -INFINITY});
      better = better && o1.ending == Ending::stopped && o2.best->evaluation.value() < o1.best->evaluation.value();
      fewest = std::min(fewest, again.runs());
      worst_ratio = std::max(worst_ratio, o2.best->evaluation.value() / o1.best->evaluation.value());
    }
    s.require("restarted (1+1)-ES on 10-D Rastrigin: better than one run for all 5 seeds (at least " + std::to_string(fewest) + " runs; best ratio at most " + std::to_string(worst_ratio) + ")", better && fewest > 10);
  }

  // ---- Differential evolution -----------------------------------------------------------------------------------
  {
    const std::size_t n = 5;
    DeOptions o;
    o.CR = 1;
    DifferentialEvolution de(Box::cube(n, -1e6, 1e6), o, Draws(6));
    generation(de, bm::sphere);  // the initial population
    const auto pop = de.population();
    std::map<heuresis::search::Ticket, Real> by_ticket;
    for (const auto& x : pop) by_ticket[x.ticket] = x.genotype;
    auto c = de.ask(heuresis::search::unlimited);
    double err = 0;
    std::size_t outside = 0;
    for (const auto& t : c) {
      const auto& P = t.origin.parents;
      const Real &x0 = by_ticket[P[0]], &x1 = by_ticket[P[1]], &x2 = by_ticket[P[2]], &x3 = by_ticket[P[3]];
      for (std::size_t j = 0; j < n; j++) {
        const double v = x1[j] + 0.5 * (x2[j] - x3[j]);
        const double expected = v < -1e6 ? 0.5 * (-1e6 + x0[j]) : v > 1e6 ? 0.5 * (1e6 + x0[j]) : v;
        outside += v < -1e6 || v > 1e6;
        err = std::max(err, std::fabs(t.genotype[j] - expected));
      }
    }
    s.check("DE/rand/1 with CR = 1: every trial is x_r1 + F(x_r2 − x_r3), or halfway from its target to the bound it crossed", err, 0);
    s.require("… and some trials crossed a bound", outside > 0);
    std::vector<Told> told;
    for (auto& t : c) told.push_back({t.ticket, scalar(bm::sphere(t.genotype))});
    de.tell(told);

    DeOptions o0;
    o0.CR = 0;
    DifferentialEvolution d0(Box::cube(n, -1e6, 1e6), o0, Draws(6));
    generation(d0, bm::sphere);
    const auto p0 = d0.population();
    auto c0 = d0.ask(heuresis::search::unlimited);
    bool one = true;
    for (std::size_t i = 0; i < c0.size(); i++) {
      std::size_t changed = 0;
      for (std::size_t j = 0; j < n; j++) changed += c0[i].genotype[j] != p0[i].genotype[j];
      one = one && changed == 1;
    }
    s.require("binomial crossover with CR = 0 takes exactly one coordinate (j_rand) from the mutant", one);

    DifferentialEvolution a(Box::cube(n, -5, 5), {}, Draws(12)), b(Box::cube(n, -5, 5), {}, Draws(12));
    for (int g = 0; g < 60; g++) { generation(a, bm::rastrigin); generation(b, bm::rastrigin, true); }
    bool same = true;
    for (std::size_t i = 0; i < a.population().size(); i++) same = same && a.population()[i].genotype == b.population()[i].genotype;
    s.require("DE: telling a generation in reverse leaves the population unchanged", same);

    DeOptions rosen;
    rosen.population = 40;
    rosen.F = 0.7;
    DifferentialEvolution dr(Box::cube(n, -5, 5), rosen, Draws(13));
    const auto or_ = run(dr, [](const Real& x) { return bm::rosenbrock(x); }, Budget{4e5, INFINITY, 1e-8});
    s.require("DE/rand/1/bin solves the 5-D Rosenbrock function to 10⁻⁸", or_.ending == Ending::target);
    DeOptions sep;
    sep.CR = 0.1;
    DifferentialEvolution ds(Box::cube(n, -5.12, 5.12), sep, Draws(14));
    const auto os = run(ds, [](const Real& x) { return bm::rastrigin(x); }, Budget{2e5, INFINITY, 1e-8});
    s.require("DE with a low CR solves the separable 5-D Rastrigin function", os.ending == Ending::target);

    Shade sh(Box::cube(10, -5.12, 5.12), {}, Draws(15));
    const auto osh = run(sh, [](const Real& x) { return bm::rastrigin(x); }, Budget{3e5, INFINITY, 1e-8});
    s.require("SHADE solves the 10-D Rastrigin function", osh.ending == Ending::target);
    bool mem = true;
    for (std::size_t k = 0; k < sh.memory_cr().size(); k++) mem = mem && sh.memory_cr()[k] >= 0 && sh.memory_cr()[k] <= 1 && sh.memory_f()[k] > 0 && sh.memory_f()[k] <= 1;
    s.require("SHADE's memories stay in their ranges, its archive at most NP", mem && sh.archive_size() <= 100);
    Shade sa(Box::cube(n, -5, 5), {}, Draws(16)), sb(Box::cube(n, -5, 5), {}, Draws(16));
    for (int g = 0; g < 30; g++) { generation(sa, bm::ackley); generation(sb, bm::ackley, true); }
    bool shs = sa.memory_f() == sb.memory_f();
    for (std::size_t i = 0; i < sa.population().size(); i++) shs = shs && sa.population()[i].genotype == sb.population()[i].genotype;
    s.require("SHADE: telling a generation in reverse leaves the run unchanged", shs);
  }

  return s.done();
}
