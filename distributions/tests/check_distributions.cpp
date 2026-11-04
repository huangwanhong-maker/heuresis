// Checks of heuresis distributions: the generic IGO step (score and Fisher information, solved) equal to PBIL's
// update on bits and to CMA-ES's rank-μ update on Gaussians; the expectation-parameter step equal to UMDA's and the
// cross-entropy method's refits, and differing from the (m, C) step by exactly δt² d̄d̄ᵀ; the Gaussian score centred
// with covariance the Fisher information; the compact GA's expected step against enumeration; the Plackett–Luce
// samplers against the law, Hunter's MM monotone and consistent; UMDA, PBIL, the compact GA, categorical UMDA,
// the cross-entropy method and Plackett–Luce solving their problems; tied weights; order independence; another
// sampler through the device seam.

#include <heuresis/benchmarks/benchmarks.hpp>
#include <heuresis/distributions/distributions.hpp>
#include <heuresis/search/search.hpp>
#include <heuresis/variation/distance.hpp>

#include <calx/verification/check.hpp>

#include <cmath>
#include <map>

using namespace heuresis::distributions;
using heuresis::search::Budget;
using heuresis::search::Draws;
using heuresis::search::Ending;
using heuresis::search::scalar;
using heuresis::search::tag;
using heuresis::variation::Permutation;

namespace {

double max_abs(const std::vector<double>& a, const std::vector<double>& b) {
  double m = 0;
  for (std::size_t i = 0; i < a.size(); i++) m = std::max(m, std::fabs(a[i] - b[i]));
  return m;
}

double max_abs(const calx::linalg::Matrix& a, const calx::linalg::Matrix& b) {
  double m = 0;
  for (std::size_t i = 0; i < a.a.size(); i++) m = std::max(m, std::fabs(a.a[i] - b.a[i]));
  return m;
}

// The Plackett–Luce probability of a ranking, directly.
double pl_probability(const std::vector<double>& w, const Permutation& p) {
  double rest = 0, pr = 1;
  for (auto i : p) rest += w[i];
  for (auto i : p) { pr *= w[i] / rest; rest -= w[i]; }
  return pr;
}

struct GumbelDevice {
  Permutation operator()(const PlackettLuce& f, const std::vector<double>& w, calx::random::Stream& r) const { return f.sample_by_gumbel(w, r); }
};

}  // namespace

int main() {
  calx::verification::Suite s;
  const Draws draws(31);

  // IGO on bits: the Fisher step is PBIL's update, and the expectation step at δt = 1 is UMDA's.
  {
    const ProductBernoulli f{};
    std::vector<double> p{0.2, 0.5, 0.7, 0.9, 0.35, 0.6};
    std::vector<Bits> gen;
    for (int k = 0; k < 10; k++) { auto r = draws(tag("check.bern.gen"), 0, k); gen.push_back(f.sample(p, r)); }
    std::vector<heuresis::search::Evaluation> ev;
    for (int k = 0; k < 10; k++) ev.push_back(scalar(k));  // already ranked
    const auto w = truncation_weights(10, 3);
    const auto igo = igo_step(f, p, std::span<const Bits>(gen), w, 0.15);
    const auto pbil = Pbil{3, 0.15, false}(f, p, gen, ev);
    s.check("IGO on product Bernoulli (Fisher solved) = PBIL's update", max_abs(igo, pbil), 1e-15);
    const auto exp1 = expectation_step(f, p, std::span<const Bits>(gen), w, 1.0);
    const auto umda = Umda{3, false}(f, p, gen, ev);
    s.check("the expectation step at δt = 1 = UMDA's refit", max_abs(exp1, umda), 1e-15);
    double total = 0;
    for (std::size_t m = 0; m < 64; m++) {
      Bits x(6);
      for (int i = 0; i < 6; i++) x[i] = (m >> i) & 1;
      total += std::exp(f.log_density(p, x));
    }
    s.check("the product Bernoulli law sums to 1", total - 1, 1e-14);
  }

  // IGO on Gaussians.
  {
    const std::size_t n = 3;
    const Gaussian f{n};
    GaussianParameters p{{0.5, -1, 2}, calx::linalg::Matrix(n)};
    const double C[9] = {2.0, 0.3, -0.4, 0.3, 1.0, 0.2, -0.4, 0.2, 0.7};
    for (int i = 0; i < 9; i++) p.C.a[i] = C[i];
    std::vector<Real> gen;
    for (int k = 0; k < 12; k++) { auto r = draws(tag("check.gauss.gen"), 0, k); gen.push_back(f.sample(p, r)); }
    std::vector<heuresis::search::Evaluation> ev;
    for (int k = 0; k < 12; k++) ev.push_back(scalar(k));
    const auto w = logarithmic_weights(12);
    const double dt = 0.3;
    const auto igo = igo_step(f, p, std::span<const Real>(gen), w, dt);
    const auto rank_mu = RankMu{w, dt}(f, p, gen, ev);
    s.check("IGO on Gaussians in (m, C) (Fisher solved): the mean = rank-μ CMA's", max_abs(igo.m, rank_mu.m), 1e-13);
    s.check("IGO on Gaussians in (m, C) (Fisher solved): the covariance = rank-μ CMA's", max_abs(igo.C, rank_mu.C), 1e-13);

    const auto tw = truncation_weights(12, 4);
    const auto e1 = expectation_step(f, p, std::span<const Real>(gen), tw, 1.0);
    const auto ce = CrossEntropy{4, 1.0, 0.0}(f, p, gen, ev);
    s.check("the expectation step at δt = 1 = the cross-entropy refit (mean)", max_abs(e1.m, ce.m), 1e-13);
    s.check("the expectation step at δt = 1 = the cross-entropy refit (covariance)", max_abs(e1.C, ce.C), 1e-13);

    const auto ed = expectation_step(f, p, std::span<const Real>(gen), w, dt);
    const auto md = RankMu{w, dt}(f, p, gen, ev);
    std::vector<double> dbar(n, 0.0);
    for (std::size_t k = 0; k < gen.size(); k++)
      for (std::size_t i = 0; i < n; i++) dbar[i] += w[k] * (gen[k][i] - p.m[i]);
    double dev = 0;
    for (std::size_t i = 0; i < n; i++)
      for (std::size_t j = 0; j < n; j++) dev = std::max(dev, std::fabs(ed.C(i, j) - (md.C(i, j) - dt * dt * dbar[i] * dbar[j])));
    s.check("the two parametrisations differ by exactly δt² d̄d̄ᵀ", dev, 1e-13);
    s.check("… and agree on the mean", max_abs(ed.m, md.m), 1e-14);

    // The score is centred and its covariance is the Fisher information.
    const std::size_t d = n + n * (n + 1) / 2;
    std::vector<double> mean(d, 0.0);
    calx::linalg::Matrix cov(d);
    const int N = 200000;
    for (int t = 0; t < N; t++) {
      auto r = draws(tag("check.gauss.score"), 0, t);
      const auto g = f.score(p, f.sample(p, r));
      for (std::size_t a = 0; a < d; a++) {
        mean[a] += g[a] / N;
        for (std::size_t b = 0; b < d; b++) cov(a, b) += g[a] * g[b] / N;
      }
    }
    const auto I = f.fisher(p);
    double rel = 0, scale = 0;
    for (std::size_t a = 0; a < d * d; a++) { rel = std::max(rel, std::fabs(cov.a[a] - I.a[a])); scale = std::max(scale, std::fabs(I.a[a])); }
    s.check("the Gaussian Fisher information = E[score scoreᵀ] (relative to its largest entry)", rel / scale, 0.03);
    double mm = 0;
    for (double v : mean) mm = std::max(mm, std::fabs(v));
    s.check("the Gaussian score is centred", mm, 0.02);
    s.check("the Gaussian log-density at the mean", f.log_density(p, p.m) - (-1.5 * std::log(2 * std::numbers::pi) - 0.5 * std::log(calx::linalg::determinant(p.C))), 1e-13);
  }

  // The compact GA's expected step on OneMax, against enumeration.
  {
    const ProductBernoulli f{};
    const std::vector<double> p{0.45, 0.5, 0.55};
    const double K = 20;
    std::vector<double> exact(3, 0.0);
    for (std::size_t a = 0; a < 8; a++)
      for (std::size_t b = 0; b < 8; b++) {
        Bits x(3), y(3);
        for (int i = 0; i < 3; i++) { x[i] = (a >> i) & 1; y[i] = (b >> i) & 1; }
        const double pr = std::exp(f.log_density(p, x) + f.log_density(p, y));
        const bool x_wins = heuresis::benchmarks::onemax(x) <= heuresis::benchmarks::onemax(y);  // ties to the first
        const Bits& w = x_wins ? x : y;
        const Bits& l = x_wins ? y : x;
        for (int i = 0; i < 3; i++) exact[i] += pr * (static_cast<double>(w[i]) - static_cast<double>(l[i])) / K;
      }
    std::vector<double> mean(3, 0.0);
    const int N = 200000;
    for (int t = 0; t < N; t++) {
      DistributionSearch cga(f, p, 2, Compact{K}, Draws(1000 + t));
      heuresis::search::run(cga, [](const Bits& x) { return heuresis::benchmarks::onemax(x); }, Budget{2, INFINITY, -INFINITY});
      for (int i = 0; i < 3; i++) mean[i] += (cga.parameters()[i] - p[i]) / N;
    }
    s.check("the compact GA's mean step on OneMax = its expectation by enumeration", max_abs(mean, exact), 5e-4);
  }

  // Bits: UMDA, PBIL and the compact GA solve OneMax.
  {
    const std::size_t n = 40;
    auto onemax = [](const Bits& x) { return heuresis::benchmarks::onemax(x); };
    DistributionSearch umda(ProductBernoulli{}, ProductBernoulli::uniform(n), 60, Umda{30}, Draws(1));
    const auto ou = heuresis::search::run(umda, onemax, Budget{60000, INFINITY, 0});
    DistributionSearch pbil(ProductBernoulli{}, ProductBernoulli::uniform(n), 30, Pbil{2, 0.1}, Draws(2));
    const auto op = heuresis::search::run(pbil, onemax, Budget{60000, INFINITY, 0});
    DistributionSearch cga(ProductBernoulli{}, ProductBernoulli::uniform(n), 2, Compact{40}, Draws(3));
    const auto oc = heuresis::search::run(cga, onemax, Budget{60000, INFINITY, 0});
    s.require("UMDA solves OneMax (n = 40)", ou.ending == Ending::target);
    s.require("PBIL solves OneMax (n = 40)", op.ending == Ending::target);
    s.require("the compact GA solves OneMax (n = 40, K = 40)", oc.ending == Ending::target);

    const ProductCategorical cat{std::vector<std::size_t>(20, 4)};
    auto target = [](const ProductCategorical::Point& x) {
      double d = 0;
      for (std::size_t i = 0; i < x.size(); i++) d += x[i] != i % 4;
      return d;
    };
    DistributionSearch cu(cat, cat.uniform(), 60, CategoricalUmda{30}, Draws(4));
    const auto oq = heuresis::search::run(cu, target, Budget{60000, INFINITY, 0});
    s.require("categorical UMDA solves a four-valued OneMax (n = 20)", oq.ending == Ending::target);
    double tot = 0;
    for (std::size_t i = 0; i < 20; i++) for (double v : cu.parameters()[i]) tot += v;
    s.check("categorical laws stay normalised", tot / 20 - 1, 1e-12);
  }

  // The cross-entropy method: stalled by geometric smoothing of the covariance, carried by dynamic smoothing.
  {
    auto sphere = [](const Real& x) { return heuresis::benchmarks::sphere(x); };
    DistributionSearch stall(Gaussian{5}, Gaussian::isotropic(Real(5, 3.0), 2.0), 50, CrossEntropy{10, 0.7}, Draws(5));
    const auto o0 = heuresis::search::run(stall, sphere, Budget{20000, INFINITY, 1e-10});
    s.require("the cross-entropy method with constant smoothing stalls on the sphere (f > 0.1: premature convergence)", o0.best->evaluation.value() > 0.1);
    DistributionSearch ce(Gaussian{5}, Gaussian::isotropic(Real(5, 3.0), 2.0), 50, CrossEntropy{10, 0.7, 0.8, 5}, Draws(5));
    const auto o = heuresis::search::run(ce, sphere, Budget{20000, INFINITY, 1e-10});
    s.check("with dynamic smoothing it reaches the sphere's minimum (n = 5, 20000 evaluations)", o.best->evaluation.value(), 1e-6);
    auto rosen = [](const Real& x) { return heuresis::benchmarks::rosenbrock(x); };
    DistributionSearch cr(Gaussian{3}, Gaussian::isotropic(Real(3, -1.0), 1.0), 200, CrossEntropy{20, 0.7, 0.8, 5}, Draws(6));
    const auto orr = heuresis::search::run(cr, rosen, Budget{100000, INFINITY, 1e-8});
    s.check("… and Rosenbrock's (n = 3, 100000 evaluations)", orr.best->evaluation.value(), 1e-4);
  }

  // Plackett–Luce: both samplers follow the law; Hunter's MM is monotone and consistent.
  {
    const PlackettLuce f{};
    const std::vector<double> w{0.5, 0.3, 0.2};
    std::map<Permutation, double> seq, gum;
    const int N = 300000;
    for (int t = 0; t < N; t++) {
      auto r1 = draws(tag("check.pl.seq"), 0, t), r2 = draws(tag("check.pl.gumbel"), 0, t);
      seq[f.sample(w, r1)] += 1.0 / N;
      gum[f.sample_by_gumbel(w, r2)] += 1.0 / N;
    }
    double d1 = 0, d2 = 0, total = 0;
    Permutation p = heuresis::variation::identity_permutation(3);
    do {
      const double pr = pl_probability(w, p);
      total += pr;
      d1 = std::max(d1, std::fabs(seq[p] - pr));
      d2 = std::max(d2, std::fabs(gum[p] - pr));
      s.check("the log-density is the law's", std::exp(f.log_density(w, p)) - pr, 1e-15);
    } while (std::next_permutation(p.begin(), p.end()));
    s.check("the Plackett–Luce law sums to 1 over S₃", total - 1, 1e-15);
    s.check("sequential choice draws the law", d1, 0.003);
    s.check("the Gumbel sampler draws the same law", d2, 0.003);

    const std::vector<double> truth{0.35, 0.25, 0.15, 0.12, 0.08, 0.05};
    std::vector<Permutation> sample;
    for (int t = 0; t < 20000; t++) { auto r = draws(tag("check.pl.fit"), 0, t); sample.push_back(f.sample(truth, r)); }
    auto loglik = [&](const std::vector<double>& v) { double l = 0; for (const auto& q : sample) l += f.log_density(v, q); return l; };
    std::vector<double> est = PlackettLuce::uniform(6);
    bool monotone = true;
    double prev = loglik(est);
    for (int it = 0; it < 30; it++) {
      est = plackett_luce_fit(sample, {}, est, 1);
      const double l = loglik(est);
      monotone = monotone && l >= prev - 1e-12 * std::fabs(prev);  // to the rounding of a sum of 20000 terms
      prev = l;
    }
    s.require("Hunter's MM increases the likelihood at every iteration", monotone);
    s.check("Hunter's MM recovers the weights from 20000 rankings", max_abs(est, truth), 0.01);
  }

  // Plackett–Luce on permutations: the hidden permutation found, by both samplers.
  {
    const Permutation hidden{3, 7, 0, 5, 1, 6, 2, 4};
    auto kendall = [&](const Permutation& p) { return static_cast<double>(heuresis::variation::kendall_distance(p, hidden)); };
    DistributionSearch pl(PlackettLuce{}, PlackettLuce::uniform(8), 100, PlackettLuceRule{20}, Draws(7));
    const auto o = heuresis::search::run(pl, kendall, Budget{40000, INFINITY, 0});
    s.require("the Plackett–Luce EDA finds a hidden permutation of 8 (Kendall's distance 0)", o.ending == Ending::target);
    DistributionSearch<PlackettLuce, PlackettLuceRule, GumbelDevice> pg(PlackettLuce{}, PlackettLuce::uniform(8), 100, PlackettLuceRule{20}, Draws(7), GumbelDevice{});
    const auto og = heuresis::search::run(pg, kendall, Budget{40000, INFINITY, 0});
    s.require("… and with another sampler of the same law through the device seam", og.ending == Ending::target);
  }

  // Tied weights, and the order of telling.
  {
    const std::vector<heuresis::search::Evaluation> ev{scalar(1), scalar(2), scalar(2), scalar(2), scalar(5)};
    const auto tw = tied_weights(ev, std::vector<double>{0.4, 0.3, 0.2, 0.1, 0.0});
    s.check("tied points share the mean of their ranks' weights", std::fabs(tw[1] - 0.2) + std::fabs(tw[2] - 0.2) + std::fabs(tw[3] - 0.2) + std::fabs(tw[0] - 0.4), 1e-15);

    DistributionSearch a(ProductBernoulli{}, ProductBernoulli::uniform(12), 8, Umda{4}, Draws(8));
    auto b = a;
    for (int g = 0; g < 20; g++) {
      auto ca = a.ask(heuresis::search::unlimited), cb = b.ask(heuresis::search::unlimited);
      std::vector<heuresis::search::Told> ta, tb;
      for (auto& c : ca) ta.push_back({c.ticket, scalar(heuresis::benchmarks::onemax(c.genotype))});
      for (auto& c : cb) tb.push_back({c.ticket, scalar(heuresis::benchmarks::onemax(c.genotype))});
      std::reverse(tb.begin(), tb.end());
      a.tell(ta);
      b.tell(tb);
    }
    s.check("a distribution search does not depend on the order of telling", max_abs(a.parameters(), b.parameters()), 0);
  }

  return s.done();
}
