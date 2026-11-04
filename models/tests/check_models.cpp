// Checks of heuresis models: the surrogate's posterior and log marginal likelihood against a direct Cholesky
// computation (calx), for both kernels; the Matérn kernel at a hand-computed distance; expected improvement and the
// probability of improvement against Monte Carlo of the posterior; type-II maximum likelihood raising the likelihood
// and recovering a length scale from a draw of the prior; Bayesian optimisation reaching Branin's and Hartmann's
// minima within small budgets; batches by lies spreading out, and running on simulated workers; reproducibility; the
// surrogate unscaled to the problem's units, reproducing the values told at their points.

#include <heuresis/benchmarks/benchmarks.hpp>
#include <heuresis/models/models.hpp>
#include <heuresis/search/search.hpp>
#include <heuresis/variation/distance.hpp>

#include <calx/linalg/cholesky.hpp>
#include <calx/verification/check.hpp>

#include <cmath>
#include <numbers>

using namespace heuresis::models;
using heuresis::search::Budget;
using heuresis::search::Draws;
using heuresis::search::tag;

int main() {
  calx::verification::Suite s;
  const Draws draws(51);

  // The posterior and the likelihood against a direct computation.
  for (KernelKind kind : {KernelKind::squared_exponential, KernelKind::matern52}) {
    const std::string name = kind == KernelKind::squared_exponential ? "squared exponential" : "Matérn 5/2";
    Kernel k{kind, 1.7, {0.4, 0.9}};
    const double noise = 0.01, mean = 0.3;
    std::vector<std::vector<double>> X;
    std::vector<double> y;
    auto r = draws(tag("check.gp.data"), static_cast<int>(kind), 0);
    for (int i = 0; i < 15; i++) {
      X.push_back({r.uniform(), r.uniform()});
      y.push_back(std::sin(4 * X.back()[0]) + X.back()[1] + 0.1 * r.normal());
    }
    GaussianProcess gp(k, noise, mean);
    for (int i = 0; i < 15; i++) gp.observe(X[i], y[i]);
    calx::linalg::Matrix K(15);
    for (int i = 0; i < 15; i++)
      for (int j = 0; j < 15; j++) K(i, j) = k(X[i], X[j]) + (i == j ? noise : 0);
    const auto f = calx::linalg::cholesky(K);
    std::vector<double> d(15);
    for (int i = 0; i < 15; i++) d[i] = y[i] - mean;
    const auto alpha = calx::linalg::cholesky_solve(f, d);
    double dm = 0, dv = 0;
    for (int q = 0; q < 10; q++) {
      const std::vector<double> x{r.uniform(), r.uniform()};
      std::vector<double> kx(15);
      for (int i = 0; i < 15; i++) kx[i] = k(X[i], x);
      double m = mean;
      for (int i = 0; i < 15; i++) m += kx[i] * alpha[i];
      const auto v = calx::linalg::cholesky_solve(f, kx);
      double var = k(x, x);
      for (int i = 0; i < 15; i++) var -= kx[i] * v[i];
      const auto p = gp.predict(x);
      dm = std::max(dm, std::fabs(p.mean - m));
      dv = std::max(dv, std::fabs(p.variance - var));
    }
    double logdet = 0, quad = 0;
    for (int i = 0; i < 15; i++) { logdet += std::log(f.l(i, i)); quad += d[i] * alpha[i]; }
    const double ll = -0.5 * quad - logdet - 7.5 * std::log(2 * std::numbers::pi);
    s.check(name + ": the posterior mean against Cholesky", dm, 1e-12);
    s.check(name + ": the posterior variance against Cholesky", dv, 1e-12);
    s.check(name + ": the log marginal likelihood against Cholesky", gp.log_likelihood() - ll, 1e-10);
  }
  {
    const Kernel m{KernelKind::matern52, 2.0, {0.5}};
    const std::vector<double> a{0.0}, b{0.3};
    const double t = std::sqrt(5.0) * 0.6;
    s.check("Matérn 5/2 at r = 0.6: σ²(1 + √5r + 5r²/3)e^{−√5r}", m(a, b) - 2 * (1 + t + t * t / 3) * std::exp(-t), 1e-15);
  }

  // Acquisitions against Monte Carlo.
  {
    const double mu = 0.3, sd = 0.5, best = 0.1;
    double ei = 0, pi = 0;
    const int N = 1000000;
    auto r = draws(tag("check.acquisition"), 0, 0);
    for (int t = 0; t < N; t++) {
      const double f = mu + sd * r.normal();
      ei += std::max(best - f, 0.0) / N;
      pi += (f < best ? 1.0 : 0.0) / N;
    }
    s.check("expected improvement against Monte Carlo of the posterior", expected_improvement(mu, sd, best) - ei, 1e-3);
    s.check("probability of improvement against Monte Carlo", probability_of_improvement(mu, sd, best) - pi, 2e-3);
    s.require("with no uncertainty they reduce to their limits", expected_improvement(0, 0, 1) == 1 && probability_of_improvement(2, 0, 1) == 0);
  }

  // Type-II maximum likelihood recovers the length scale of a draw from the prior.
  {
    const Kernel truth{KernelKind::squared_exponential, 1.0, {0.15}};
    std::vector<std::vector<double>> X;
    for (int i = 0; i < 60; i++) X.push_back({i / 59.0});
    calx::linalg::Matrix K(60);
    for (int i = 0; i < 60; i++)
      for (int j = 0; j < 60; j++) K(i, j) = truth(X[i], X[j]) + (i == j ? 1e-6 : 0);
    const auto L = calx::linalg::cholesky(K);
    auto r = draws(tag("check.gp.prior"), 0, 0);
    std::vector<double> z(60), y(60, 0.0);
    for (auto& v : z) v = r.normal();
    for (int i = 0; i < 60; i++)
      for (int j = 0; j <= i; j++) y[i] += L.l(i, j) * z[j];
    const Kernel start{KernelKind::squared_exponential, 0.5, {0.5}};
    const double before = log_marginal_likelihood(X, y, start, 1e-3);
    const Fit fit = fit_hyperparameters(X, y, start, 1e-3, 0, HyperparameterBounds{}, 400);
    s.require("type-II maximum likelihood raises the likelihood", fit.log_likelihood > before);
    s.check("… and recovers the length scale 0.15 of a draw from the prior (relative error)", fit.kernel.lengthscales[0] / 0.15 - 1, 0.25);
  }

  // Bayesian optimisation on Branin and Hartmann's three-dimensional function.
  {
    const heuresis::variation::Box branin_box{{-5, 0}, {10, 15}};
    BayesianOptimisation bo(branin_box, BayesianOptions{}, Draws(1));
    const auto o = heuresis::search::run(bo, [](const Real& x) { return heuresis::benchmarks::branin(x); }, Budget{60, INFINITY, -INFINITY});
    s.check("Bayesian optimisation reaches Branin's minimum 0.397887 in 60 evaluations", o.best->evaluation.value() - 0.397887357729738, 1e-4);
    BayesianOptimisation bh(heuresis::variation::Box::cube(3, 0, 1), BayesianOptions{}, Draws(2));
    const auto oh = heuresis::search::run(bh, [](const Real& x) { return heuresis::benchmarks::hartmann3(x); }, Budget{60, INFINITY, -INFINITY});
    s.check("… and Hartmann-3's minimum −3.86278 in 60", oh.best->evaluation.value() + 3.86278214782076, 1e-3);

    BayesianOptimisation again(branin_box, BayesianOptions{}, Draws(1));
    const auto o2 = heuresis::search::run(again, [](const Real& x) { return heuresis::benchmarks::branin(x); }, Budget{60, INFINITY, -INFINITY});
    s.require("the same seed gives the same run", o2.best->ticket == o.best->ticket && o2.best->evaluation.value() == o.best->evaluation.value());

    // The surrogate in the problem's units: the standardisation is the sample mean and deviation of the values told,
    // and the unscaled posterior mean reproduces each value at its point.
    BayesianOptimisation drawn(branin_box, BayesianOptions{}, Draws(1));
    heuresis::search::Record<Real> rec;
    heuresis::search::run(drawn, [](const Real& x) { return heuresis::benchmarks::branin(x); }, Budget{60, INFINITY, -INFINITY}, heuresis::search::Serial{}, heuresis::search::unlimited, &rec);
    double m = 0, v = 0;
    for (const auto& x : rec.told) m += x.evaluation.value() / 60;
    for (const auto& x : rec.told) v += (x.evaluation.value() - m) * (x.evaluation.value() - m) / 59;
    const Standardisation st = drawn.standardisation();
    s.check("the standardisation is the sample mean and standard deviation of the values told", std::max(std::fabs(st.mean - m) / std::fabs(m), std::fabs(st.deviation - std::sqrt(v)) / std::sqrt(v)), 1e-12);
    const GaussianProcess gp = drawn.surrogate();
    double worst = 0, roundtrip = 0;
    for (const auto& x : rec.told) {
      const auto u = drawn.to_unit(x.genotype);
      roundtrip = std::max(roundtrip, heuresis::variation::euclidean(drawn.to_box(u), x.genotype));
      worst = std::max(worst, std::fabs(st.posterior(gp.predict(u)).mean - x.evaluation.value()));
    }
    s.check("to_unit inverts to_box", roundtrip, 1e-12);
    s.check("unscaled, the surrogate's mean reproduces the 60 values told at their points (relative to the deviation " + std::to_string(st.deviation) + ")", worst / st.deviation, 1e-3);
  }

  // Batches by lies: distinct points per ask, and a run on simulated workers.
  {
    BayesianOptions opt;
    opt.breadth = 4;
    BayesianOptimisation bo(heuresis::variation::Box{{-5, 0}, {10, 15}}, opt, Draws(3));
    auto design = bo.ask(heuresis::search::unlimited);
    std::vector<heuresis::search::Told> told;
    for (auto& c : design) told.push_back({c.ticket, heuresis::search::scalar(heuresis::benchmarks::branin(c.genotype))});
    bo.tell(told);
    const auto batch = bo.ask(heuresis::search::unlimited);
    double closest = INFINITY;
    for (std::size_t a = 0; a < batch.size(); a++)
      for (std::size_t b = a + 1; b < batch.size(); b++) closest = std::min(closest, heuresis::variation::euclidean(batch[a].genotype, batch[b].genotype));
    s.require("a batch of 4 by the kriging believer: 4 points, pairwise apart (closest " + std::to_string(closest) + ")", batch.size() == 4 && closest > 0.05);
    s.require("breadth caps an ask after the design", bo.ask(1).size() == 1);

    opt.breadth = 1;
    BayesianOptimisation async(heuresis::variation::Box{{-5, 0}, {10, 15}}, opt, Draws(4));
    heuresis::search::Workers workers;
    workers.count = 4;
    const auto oa = heuresis::search::run_asynchronous(async, [](const Real& x) { return heuresis::benchmarks::branin(x); }, Budget{60, INFINITY, -INFINITY}, workers);
    s.check("on four asynchronous workers, pending points told by lies: Branin within 10⁻³ in 60", oa.best->evaluation.value() - 0.397887357729738, 1e-3);
  }

  return s.done();
}
