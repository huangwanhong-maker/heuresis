// Checks of heuresis swarms: Clerc and Kennedy's constriction coefficient; SPSO 2011's move commuting exactly
// with rotations and translations, the inertia rule's not; the second moments of SPSO 2011 about a fixed
// attractor against their linear recursion, stable in two dimensions and growing from three on; particle swarms on
// the sphere and Rastrigin's function; the order of telling; ant colonies reaching the optimal tour found by
// enumeration, MAX–MIN's trails within their bounds.

#include <calx/linalg/eigenvalues.hpp>

#include <heuresis/benchmarks/benchmarks.hpp>
#include <heuresis/search/search.hpp>
#include <heuresis/swarms/swarms.hpp>

#include <calx/verification/check.hpp>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <numeric>

using namespace heuresis::swarms;
using heuresis::search::Budget;
using heuresis::search::Draws;
using heuresis::search::Ending;
using heuresis::search::Told;
using heuresis::search::run;
using heuresis::search::scalar;
using heuresis::search::tag;
namespace bm = heuresis::benchmarks;

namespace {

template <class A, class F>
void generation(A& a, F& f, bool reverse = false) {
  auto c = a.ask(heuresis::search::unlimited);
  std::vector<Told> t;
  for (auto& x : c) t.push_back({x.ticket, scalar(f(x.genotype))});
  if (reverse) std::reverse(t.begin(), t.end());
  a.tell(t);
}

// The shortest tour by enumeration, the first city fixed.
double shortest_tour(const bm::Tour& t) {
  heuresis::variation::Permutation p(t.size());
  std::iota(p.begin(), p.end(), 0u);
  double best = INFINITY;
  do best = std::min(best, t.length(p));
  while (std::next_permutation(p.begin() + 1, p.end()));
  return best;
}

std::vector<double> inverse_distances(const bm::Tour& t) {
  const std::size_t n = t.size();
  std::vector<double> eta(n * n, 0.0);
  for (std::size_t i = 0; i < n; i++)
    for (std::size_t j = 0; j < n; j++)
      if (i != j) eta[i * n + j] = 1 / t.distance(i, j);
  return eta;
}

}  // namespace

int main() {
  calx::verification::Suite s;
  const Draws draws(21);

  // Constriction.
  s.check("χ(4.1) = 0.7298 (Clerc and Kennedy 2002)", constriction(4.1) - 0.7298, 5e-5);
  s.check("χφ/2 at φ = 4.1 is the customary c = 1.49618", constriction(4.1) * 4.1 / 2 - 1.49618, 5e-6);
  s.check("χ(φ) = 2/(φ − 2 + √(φ² − 4φ)) for φ > 4", constriction(5) - 2 / (5 - 2 + std::sqrt(5.0)), 1e-15);

  // Equivariance of the moves under x ↦ Rx + b, exactly for SPSO 2011, not for the inertia rule.
  {
    const std::size_t n = 6;
    const bm::Rotation R(n, 4);
    const std::vector<double> b{0.5, -2, 1, 0, 3, -1};
    auto map = [&](std::span<const double> x, bool translate) {
      auto y = R.apply(x);
      if (translate)
        for (std::size_t i = 0; i < n; i++) y[i] += b[i];
      return y;
    };
    double spso = 0, classic = 0;
    for (int t = 0; t < 200; t++) {
      auto r = draws(tag("check.pso.move"), 0, t);
      Real x(n), v(n), p(n), l(n), dir(n), r1(n), r2(n);
      double nd = 0;
      for (std::size_t i = 0; i < n; i++) {
        x[i] = r.normal(); v[i] = r.normal(); p[i] = r.normal(); l[i] = r.normal(); dir[i] = r.normal(); nd += dir[i] * dir[i];
        r1[i] = r.uniform(); r2[i] = r.uniform();
      }
      for (auto& d : dir) d /= std::sqrt(nd);
      const double frac = r.uniform();
      const bool same = t % 5 == 0;
      const auto m = spso2011_move(x, v, p, same ? p : l, same, 0.72, 1.19, dir, frac);
      const auto mr = spso2011_move(map(x, true), map(v, false), map(p, true), same ? map(p, true) : map(l, true), same, 0.72, 1.19, map(dir, false), frac);
      const auto e = map(m.x, true), ev = map(m.v, false);
      for (std::size_t i = 0; i < n; i++) spso = std::max({spso, std::fabs(mr.x[i] - e[i]), std::fabs(mr.v[i] - ev[i])});
      const auto c = inertia_move(x, v, p, l, 0.7298, 1.49618, 1.49618, r1, r2);
      const auto cr = inertia_move(map(x, true), map(v, false), map(p, true), map(l, true), 0.7298, 1.49618, 1.49618, r1, r2);
      const auto ec = map(c.x, true);
      for (std::size_t i = 0; i < n; i++) classic = std::max(classic, std::fabs(cr.x[i] - ec[i]));
    }
    s.check("SPSO 2011's move commutes with x ↦ Rx + b (the direction rotated with it)", spso, 1e-13);
    s.require("the inertia rule's move does not: its factors act coordinate by coordinate", classic > 1e-2);
  }

  // SPSO 2011 about a fixed attractor p = l (positions equal, not the same particle): with d = x − p,
  //   v' = w v + a d + |a| |d| u,  d' = d + v',  a = −2c/3,  u uniform in the unit ball, E|u|² = n/(n + 2),
  // the second moments (E|d|², E|v|², E d·v) follow a linear map M whose spectral radius decides whether the
  // particle settles or flies off. At the standard w and c it is below 1 in two dimensions and above from three.
  {
    const double w = 1 / (2 * std::numbers::ln2), c = 0.5 + std::numbers::ln2, a = -2 * c / 3, b2 = a * a;
    auto radius = [&](double n) {
      const double r = n / (n + 2);
      calx::linalg::Matrix M(3);
      const double m[9] = {(1 + a) * (1 + a) + b2 * r, w * w, 2 * w * (1 + a), a * a + b2 * r, w * w, 2 * a * w, (1 + a) * a + b2 * r, w * w, (1 + a) * w + w * a};
      M.a.assign(m, m + 9);
      return calx::linalg::spectral_radius(M);
    };
    auto simulate = [&](std::size_t n) {
      // E|d|² over many particles after 60 and after 100 steps: its growth per step.
      const int particles = 20000;
      double s60 = 0, s100 = 0;
      for (int k = 0; k < particles; k++) {
        auto r = draws(tag("check.pso.moments"), n, k);
        Real x(n), v(n, 0.0), p(n, 0.0), dir(n);
        for (auto& xi : x) xi = r.normal();
        for (int t = 1; t <= 100; t++) {
          double nd = 0;
          for (auto& d : dir) { d = r.normal(); nd += d * d; }
          for (auto& d : dir) d /= std::sqrt(nd);
          const auto m = spso2011_move(x, v, p, p, false, w, c, dir, std::pow(r.uniform(), 1.0 / static_cast<double>(n)));
          x = m.x;
          v = m.v;
          if (t == 60 || t == 100) {
            double d2 = 0;
            for (double xi : x) d2 += xi * xi;
            (t == 60 ? s60 : s100) += d2;
          }
        }
      }
      return std::pow(s100 / s60, 1.0 / 40);
    };
    const double r2 = radius(2), r3 = radius(3), r10 = radius(10);
    s.require("SPSO 2011's second moments contract in two dimensions (spectral radius below 1)", r2 < 1);
    s.require("… and grow from three dimensions on (spectral radius above 1)", r3 > 1 && r10 > r3);
    s.check("the growth of E|x − p|² in 10-D, simulated, against the spectral radius", simulate(10) / r10 - 1, 0.03);
    s.check("the decay of E|x − p|² in 2-D, simulated, against the spectral radius", simulate(2) / r2 - 1, 0.03);
  }

  // Swarms on the sphere and Rastrigin's function.
  {
    ParticleSwarm pso(heuresis::variation::Box::cube(2, -5, 5), {}, Draws(1));
    const auto o = run(pso, [](const Real& x) { return bm::sphere(x); }, Budget{2e4, INFINITY, 1e-10});
    s.require("SPSO 2011 reaches 10⁻¹⁰ on the 2-D sphere", o.ending == Ending::target);
    ParticleSwarm p10(heuresis::variation::Box::cube(10, -5, 5), {}, Draws(1));
    run(p10, [](const Real& x) { return bm::sphere(x); }, Budget{4e4, INFINITY, -INFINITY});
    double vmax = 0;
    for (const auto& v : p10.velocities()) for (double vi : v) vmax = std::max(vmax, std::fabs(vi));
    s.require("… while in 10-D its particles still fly at the scale of the box after 1000 generations", vmax > 1);
    PsoOptions inertia;
    inertia.rule = PsoRule::inertia;
    inertia.topology = PsoTopology::ring;
    ParticleSwarm pi(heuresis::variation::Box::cube(10, -5, 5), inertia, Draws(2));
    const auto oi = run(pi, [](const Real& x) { return bm::sphere(x); }, Budget{2e5, INFINITY, 1e-10});
    s.require("the inertia rule on a ring reaches 10⁻¹⁰ on the 10-D sphere", oi.ending == Ending::target);
    PsoOptions con;
    con.rule = PsoRule::constriction;
    con.topology = PsoTopology::global;
    ParticleSwarm pc(heuresis::variation::Box::cube(5, -5, 5), con, Draws(3));
    const auto oc = run(pc, [](const Real& x) { return bm::sphere(x); }, Budget{1e5, INFINITY, 1e-10});
    s.require("constriction with the whole swarm informing reaches 10⁻¹⁰ on the 5-D sphere", oc.ending == Ending::target);
    ParticleSwarm pr(heuresis::variation::Box::cube(2, -5.12, 5.12), {}, Draws(4));
    const auto orr = run(pr, [](const Real& x) { return bm::rastrigin(x); }, Budget{1e5, INFINITY, 1e-8});
    s.require("SPSO 2011 finds Rastrigin's global minimum in 2-D", orr.ending == Ending::target);
    bool inside = true;
    for (const auto& x : p10.positions()) inside = inside && heuresis::variation::Box::cube(10, -5, 5).contains(x);
    s.require("confinement keeps every particle in the box", inside);
  }

  // The order of telling.
  {
    ParticleSwarm a(heuresis::variation::Box::cube(4, -3, 3), {}, Draws(9)), b(heuresis::variation::Box::cube(4, -3, 3), {}, Draws(9));
    auto f = [](const Real& x) { return bm::ackley(x); };
    for (int g = 0; g < 40; g++) { generation(a, f); generation(b, f, true); }
    s.require("a swarm told in reverse is the same swarm", a.positions() == b.positions() && a.velocities() == b.velocities());
  }

  // Ant colonies on tours.
  {
    const bm::Tour tour = bm::Tour::random(9, 77);
    const double optimum = shortest_tour(tour);
    AntColony mmas(9, inverse_distances(tour), {}, Draws(5));
    bool bounded = true;
    double reached = INFINITY;
    for (int g = 0; g < 400 && reached > optimum + 1e-12; g++) {
      generation(mmas, tour);
      reached = mmas.best()->evaluation.value();
      for (double t : mmas.trails()) bounded = bounded && t >= mmas.tau_min() * (1 - 1e-12) && t <= mmas.tau_max() * (1 + 1e-12);
    }
    s.check("MAX–MIN Ant System reaches the optimal 9-city tour (enumerated)", reached - optimum, 1e-12);
    s.require("MAX–MIN keeps every trail within [τ_min, τ_max]", bounded && mmas.tau_min() > 0 && mmas.tau_min() < mmas.tau_max());
    const double N = 9, root = std::pow(0.05, 1 / N);
    s.check("τ_max = 1/(ρ L_best) and τ_min = τ_max(1 − p^{1/n})/((n/2 − 1)p^{1/n})",
            std::fabs(mmas.tau_max() - 1 / (0.02 * reached)) + std::fabs(mmas.tau_min() - mmas.tau_max() * (1 - root) / ((N / 2 - 1) * root)), 1e-12);

    const bm::Tour seven = bm::Tour::random(7, 78);
    const double opt7 = shortest_tour(seven);
    AcoOptions as;
    as.variant = AcoVariant::ant_system;
    as.beta = 3;
    AntColony ant(7, inverse_distances(seven), as, Draws(6));
    const auto oa = run(ant, seven, Budget{3000, INFINITY, opt7 + 1e-12});
    s.require("the Ant System reaches the optimal 7-city tour", oa.ending == Ending::target);

    AntColony a(9, inverse_distances(tour), {}, Draws(7)), b(9, inverse_distances(tour), {}, Draws(7));
    for (int g = 0; g < 30; g++) { generation(a, tour); generation(b, tour, true); }
    s.require("a colony told in reverse is the same colony", a.trails() == b.trails());
    bool perms = true;
    for (const auto& c : a.ask(heuresis::search::unlimited)) perms = perms && heuresis::variation::is_permutation(c.genotype);
    s.require("every ant builds a permutation", perms);
  }

  return s.done();
}
