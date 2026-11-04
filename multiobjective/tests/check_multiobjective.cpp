// Checks of heuresis multiobjective: dominance a strict partial order; fast non-dominated sorting equal to the
// heights of the dominance poset; the hypervolume by sweeps, by WFG and by inclusion–exclusion equal on random
// sets, convergent at first order to the closed forms of the benchmarks' fronts, strictly monotone; contributions
// in closed form equal to differences; the indicators' defining properties; NSGA-II, NSGA-III, SMS-EMOA and MOEA/D
// on ZDT1, DTLZ1 and DTLZ2; and generational runs independent of the order of telling, the executor and the workers.

#include <heuresis/benchmarks/benchmarks.hpp>
#include <heuresis/multiobjective/multiobjective.hpp>
#include <heuresis/search/search.hpp>

#include <calx/verification/check.hpp>

#include <chrono>
#include <cmath>
#include <functional>
#include <numbers>

using namespace heuresis::multiobjective;
using heuresis::search::Draws;
using heuresis::search::tag;
using heuresis::variation::Box;
using heuresis::variation::Real;

namespace {

std::vector<Point> random_points(std::size_t n, std::size_t m, calx::random::Stream& s, bool grid = false) {
  std::vector<Point> p(n, Point(m));
  for (auto& q : p)
    for (auto& v : q) v = grid ? static_cast<double>(s.below(5)) / 5 : s.uniform();
  return p;
}

// The height of each point in the dominance poset, by memoised recursion over its dominators.
std::vector<std::size_t> heights(const std::vector<Point>& p) {
  std::vector<long> h(p.size(), -1);
  std::function<long(std::size_t)> height = [&](std::size_t i) -> long {
    if (h[i] >= 0) return h[i];
    long best = 0;
    for (std::size_t j = 0; j < p.size(); j++)
      if (dominates(p[j], p[i])) best = std::max(best, height(j) + 1);
    return h[i] = best;
  };
  std::vector<std::size_t> out(p.size());
  for (std::size_t i = 0; i < p.size(); i++) out[i] = static_cast<std::size_t>(height(i));
  return out;
}

double zdt1_hv(const std::vector<Point>& front) { return hypervolume_2d(front, Point{1, 1}); }

// The intersections of the reference lines with DTLZ2's front, and with DTLZ1's.
std::vector<Point> sphere_targets(const std::vector<Point>& w) {
  std::vector<Point> t;
  for (auto v : w) {
    double n = 0;
    for (double x : v) n += x * x;
    for (auto& x : v) x /= std::sqrt(n);
    t.push_back(v);
  }
  return t;
}
std::vector<Point> simplex_targets(const std::vector<Point>& w) {
  std::vector<Point> t;
  for (auto v : w) {
    for (auto& x : v) x *= 0.5;
    t.push_back(v);
  }
  return t;
}

template <class A>
std::vector<heuresis::search::Ticket> tickets(const A& a) {
  std::vector<heuresis::search::Ticket> t;
  for (const auto& x : a.population()) t.push_back(x.ticket);
  return t;
}

}  // namespace

int main() {
  calx::verification::Suite s;
  const Draws draws(2002);
  const auto start = std::chrono::steady_clock::now();

  // Dominance is a strict partial order; constrained dominance follows Deb's rules.
  {
    bool irreflexive = true, asymmetric = true, transitive = true;
    for (int t = 0; t < 300; t++) {
      auto r = draws(tag("check.mo.order"), 0, t);
      const auto p = random_points(3, 3, r, true);
      irreflexive = irreflexive && !dominates(p[0], p[0]);
      asymmetric = asymmetric && !(dominates(p[0], p[1]) && dominates(p[1], p[0]));
      if (dominates(p[0], p[1]) && dominates(p[1], p[2])) transitive = transitive && dominates(p[0], p[2]);
    }
    s.require("dominance is irreflexive, asymmetric and transitive", irreflexive && asymmetric && transitive);
    using heuresis::search::evaluation;
    const auto feasible = evaluation({5, 5}), infeasible = evaluation({0, 0}, {1}), worse = evaluation({0, 0}, {2}), failed = heuresis::search::failure("x");
    s.require("constrained dominance: feasible over infeasible, less violation first, failures last",
              constrained_dominates(feasible, infeasible) && constrained_dominates(infeasible, worse) && constrained_dominates(worse, failed) && !constrained_dominates(infeasible, feasible));
  }

  // Fast non-dominated sort: fronts are the level sets of the height.
  {
    bool same = true, antichains = true, covered = true;
    for (int t = 0; t < 60; t++) {
      auto r = draws(tag("check.mo.sort"), 0, t);
      const std::size_t m = 2 + t % 3;
      const auto p = random_points(50, m, r, t % 2 == 0);
      const auto f = non_dominated_sort(p);
      const auto h = heights(p);
      for (std::size_t i = 0; i < p.size(); i++) same = same && f.rank[i] == h[i];
      for (std::size_t k = 0; k < f.fronts.size(); k++) {
        for (auto i : f.fronts[k])
          for (auto j : f.fronts[k]) antichains = antichains && !dominates(p[i], p[j]);
        if (k > 0)
          for (auto i : f.fronts[k]) {
            bool dominated = false;
            for (auto j : f.fronts[k - 1]) dominated = dominated || dominates(p[j], p[i]);
            covered = covered && dominated;
          }
      }
    }
    s.require("fast non-dominated sort gives each point its height in the dominance poset", same);
    s.require("each front is an antichain", antichains);
    s.require("each point of front k + 1 is dominated by one of front k", covered);
  }

  // Hypervolume: the sweeps, WFG and inclusion–exclusion agree.
  {
    double d2 = 0, d3 = 0, d4 = 0, d5 = 0;
    for (int t = 0; t < 120; t++) {
      auto r = draws(tag("check.mo.hv"), 0, t);
      const bool grid = t % 3 == 0;
      const auto p2 = random_points(10, 2, r, grid), p3 = random_points(10, 3, r, grid), p4 = random_points(9, 4, r, grid), p5 = random_points(8, 5, r, grid);
      const Point r2(2, 1.1), r3(3, 1.1), r4(4, 1.1), r5(5, 1.1);
      d2 = std::max({d2, std::fabs(hypervolume_2d(p2, r2) - hypervolume_inclusion_exclusion(p2, r2)), std::fabs(hypervolume_wfg(p2, r2) - hypervolume_2d(p2, r2))});
      d3 = std::max({d3, std::fabs(hypervolume_3d(p3, r3) - hypervolume_inclusion_exclusion(p3, r3)), std::fabs(hypervolume_wfg(p3, r3) - hypervolume_3d(p3, r3))});
      d4 = std::max(d4, std::fabs(hypervolume_wfg(p4, r4) - hypervolume_inclusion_exclusion(p4, r4)));
      d5 = std::max(d5, std::fabs(hypervolume_wfg(p5, r5) - hypervolume_inclusion_exclusion(p5, r5)));
    }
    s.check("hypervolume, two objectives: sweep = WFG = inclusion–exclusion", d2, 5e-14);
    s.check("hypervolume, three objectives: Beume's sweep = WFG = inclusion–exclusion", d3, 5e-14);
    s.check("hypervolume, four objectives: WFG = inclusion–exclusion", d4, 5e-14);
    s.check("hypervolume, five objectives: WFG = inclusion–exclusion", d5, 5e-14);
    const std::vector<Point> outside{{1.2, 0.1}, {0.5, 1.0}};
    s.require("points not strictly inside the reference contribute nothing", hypervolume_2d(outside, Point{1, 1}) == 0);
  }

  // Hypervolume of points on the fronts converges to the closed forms at first order.
  {
    auto zdt = [&](std::size_t n, bool second) {
      std::vector<Point> p;
      for (std::size_t i = 0; i < n; i++) {
        const double t = static_cast<double>(i) / static_cast<double>(n - 1);
        p.push_back(second ? heuresis::benchmarks::zdt2_front(t) : heuresis::benchmarks::zdt1_front(t));
      }
      return hypervolume_2d(p, Point{1, 1});
    };
    const double e1 = heuresis::benchmarks::zdt1_hypervolume() - zdt(101, false), e2 = heuresis::benchmarks::zdt1_hypervolume() - zdt(201, false), e3 = heuresis::benchmarks::zdt1_hypervolume() - zdt(401, false);
    s.check("ZDT1's front: HV of 401 points within 2·10⁻³ of 2/3", e3, 2e-3);
    s.check("ZDT1's front: the error falls at first order", calx::verification::order(e2, e3) - 1, 0.05);
    s.require("ZDT1's front: the staircase stays below the front", e1 > e2 && e2 > e3 && e3 > 0);
    const double f1 = heuresis::benchmarks::zdt2_hypervolume() - zdt(201, true), f2 = heuresis::benchmarks::zdt2_hypervolume() - zdt(401, true);
    s.check("ZDT2's front: the error falls at first order to 1/3", calx::verification::order(f1, f2) - 1, 0.05);
    auto sphere = [&](std::size_t k) {
      std::vector<Point> p;
      const double h = std::numbers::pi / 2;
      for (std::size_t i = 0; i < k; i++)
        for (std::size_t j = 0; j < k; j++) {
          const double a = h * static_cast<double>(i) / static_cast<double>(k - 1), b = h * static_cast<double>(j) / static_cast<double>(k - 1);
          p.push_back({std::cos(a) * std::cos(b), std::cos(a) * std::sin(b), std::sin(a)});
        }
      return heuresis::benchmarks::dtlz2_hypervolume(3) - hypervolume_3d(p, Point{1, 1, 1});
    };
    const double g1 = sphere(20), g2 = sphere(40);
    s.check("DTLZ2's front: HV of a 40 × 40 grid within 0.02 of 1 − π/6", g2, 0.02);
    s.check("DTLZ2's front: the error falls at first order in the grid's step", calx::verification::order(g1, g2) - 1, 0.1);
  }

  // Strict monotonicity, and contributions.
  {
    bool strict = true, dominated_free = true, closed = true, twins = true;
    for (int t = 0; t < 100; t++) {
      auto r = draws(tag("check.mo.monotone"), 0, t);
      const std::size_t m = 2 + t % 2;
      auto p = random_points(12, m, r);
      std::vector<Point> nd;
      for (auto i : non_dominated(p)) nd.push_back(p[i]);
      const Point ref(m, 1);
      const double hv = hypervolume(nd, ref);
      auto better = nd;
      better[0][0] -= 0.01;
      strict = strict && hypervolume(better, ref) > hv;
      auto more = nd;
      Point worse = nd[0];
      for (auto& v : worse) v = std::min(0.999, v + 0.05);
      more.push_back(worse);
      dominated_free = dominated_free && std::fabs(hypervolume(more, ref) - hv) < 1e-14;
      const auto c = contributions(nd, ref);
      for (std::size_t i = 0; i < nd.size(); i++) {
        auto rest = nd;
        rest.erase(rest.begin() + static_cast<std::ptrdiff_t>(i));
        closed = closed && std::fabs(c[i] - (hv - hypervolume(rest, ref))) < 1e-14;
      }
      auto twin = nd;
      twin.push_back(nd[0]);
      const auto ct = contributions(twin, ref);
      twins = twins && ct[0] == 0 && ct.back() == 0;
    }
    s.require("strict monotonicity: improving a non-dominated point increases the hypervolume", strict);
    s.require("a dominated point adds nothing", dominated_free);
    s.require("contributions equal HV(A) − HV(A ∖ {a}) (closed form in two objectives)", closed);
    s.require("a point and its twin contribute nothing", twins);
  }

  // Indicators.
  {
    std::vector<Point> z;
    for (int i = 0; i <= 50; i++) z.push_back(heuresis::benchmarks::zdt1_front(i / 50.0));
    s.check("IGD of the reference set itself is 0", inverted_generational_distance(z, z), 0);
    std::vector<Point> shifted = z;
    for (auto& p : shifted) for (auto& v : p) v += 0.1;
    s.check("GD of a set translated by δ(1, 1) from the front: at most δ√2", generational_distance(shifted, z) - 0.1 * std::sqrt(2.0), 0.05);
    s.check("the additive ε of a front against its translate by δ is −δ", additive_epsilon(z, shifted) + 0.1, 1e-15);
    s.check("and of the translate against the front, +δ", additive_epsilon(shifted, z) - 0.1, 1e-15);
    bool eps_iff = true, igdp = true;
    for (int t = 0; t < 300; t++) {
      auto r = draws(tag("check.mo.eps"), 0, t);
      const auto a = random_points(4, 3, r, true), b = random_points(3, 3, r, true);
      bool weakly = true;
      for (const auto& y : b) {
        bool any = false;
        for (const auto& x : a) any = any || weakly_dominates(x, y);
        weakly = weakly && any;
      }
      eps_iff = eps_iff && ((additive_epsilon(a, b) <= 0) == weakly);
      auto worse = a;
      for (auto& p : worse) p[r.below(3)] += 0.3;
      igdp = igdp && inverted_generational_distance_plus(a, b) <= inverted_generational_distance_plus(worse, b) + 1e-15;
    }
    s.require("I_ε+(A, B) ≤ 0 exactly when A weakly dominates B", eps_iff);
    s.require("IGD⁺ does not increase when a set improves (weak Pareto compliance)", igdp);
    std::vector<Point> line;
    for (int i = 0; i <= 10; i++) line.push_back({i / 10.0, 1 - i / 10.0});
    s.check("spread of an even set reaching the extremes is 0", spread(line, Point{0, 1}, Point{1, 0}), 1e-15);
    std::vector<std::size_t> all(line.size());
    std::iota(all.begin(), all.end(), std::size_t{0});
    const auto cd = crowding_distance(line, all);
    s.require("crowding: the extremes are infinitely far", std::isinf(cd.front()) && std::isinf(cd.back()));
    s.check("crowding: an interior point of an even line has 4h", cd[5] - 0.4, 1e-14);
  }

  // The simplex lattice.
  {
    const auto w = das_dennis(3, 12);
    double worst = 0;
    for (const auto& v : w) worst = std::max(worst, std::fabs(v[0] + v[1] + v[2] - 1));
    s.require("Das and Dennis: C(14, 2) = 91 weights for m = 3, p = 12", w.size() == 91);
    s.check("Das and Dennis: every weight on the simplex", worst, 1e-15);
  }

  const Box zdt_box = Box::cube(30, 0, 1);
  auto zdt1 = [](const Real& x) { return heuresis::search::evaluation(heuresis::benchmarks::zdt1(x)); };

  // NSGA-II on ZDT1.
  {
    Nsga2<Real> a(real_kernels(zdt_box, 20, 20), 100, Draws(1));
    heuresis::search::run(a, zdt1, heuresis::search::Budget{20000});
    const double hv = zdt1_hv(a.front());
    s.check("NSGA-II on ZDT1 (20 000 evaluations): HV within 1.5% of 2/3", 1 - hv / heuresis::benchmarks::zdt1_hypervolume(), 0.015);
    s.require("NSGA-II keeps a population of μ", a.population().size() == 100);
  }

  // Generational runs do not depend on the order of telling, the executor or the workers.
  {
    const Box b = Box::cube(10, 0, 1);
    const heuresis::search::Budget budget{20 * 15};
    Nsga2<Real> serial(real_kernels(b), 20, Draws(9)), pooled(real_kernels(b), 20, Draws(9)), async(real_kernels(b), 20, Draws(9)), reversed(real_kernels(b), 20, Draws(9));
    heuresis::search::run(serial, zdt1, budget);
    calx::scheduler::ThreadPool pool(4);
    heuresis::search::run(pooled, zdt1, budget, heuresis::search::Pool(pool));
    heuresis::search::Workers wk;
    wk.count = 3;
    wk.duration = [](const heuresis::search::Evaluation&, std::size_t w, calx::random::Stream& r) { return 1 + w + 3 * r.uniform(); };
    heuresis::search::run_asynchronous(async, zdt1, budget, wk);
    for (int g = 0; g < 15; g++) {
      auto c = reversed.ask(heuresis::search::unlimited);
      std::vector<heuresis::search::Told> t;
      for (auto& x : c) t.push_back({x.ticket, zdt1(x.genotype)});
      std::reverse(t.begin(), t.end());
      reversed.tell(t);
    }
    s.require("NSGA-II: a pool of threads gives the same population", tickets(pooled) == tickets(serial));
    s.require("NSGA-II: simulated workers give the same population", tickets(async) == tickets(serial));
    s.require("NSGA-II: telling in reverse gives the same population", tickets(reversed) == tickets(serial));

    Moead<Real> m1(real_kernels(b), das_dennis(2, 19), {}, Draws(9)), m2(real_kernels(b), das_dennis(2, 19), {}, Draws(9));
    heuresis::search::run(m1, zdt1, budget);
    heuresis::search::run(m2, zdt1, budget, heuresis::search::Pool(pool), 7);
    bool same = m1.solutions().size() == m2.solutions().size();
    for (std::size_t i = 0; same && i < m1.solutions().size(); i++) same = m1.solutions()[i].ticket == m2.solutions()[i].ticket;
    s.require("MOEA/D: generations cut into parts on a pool give the same solutions", same);
  }

  // NSGA-III on DTLZ2 and DTLZ1 in three objectives.
  {
    const auto w = das_dennis(3, 12);
    const Box b2 = Box::cube(12, 0, 1), b1 = Box::cube(7, 0, 1);
    Nsga3<Real> a(real_kernels(b2, 30, 20, 1.0), w, 92, Draws(3));
    heuresis::search::run(a, [](const Real& x) { return heuresis::search::evaluation(heuresis::benchmarks::dtlz2(x, 3)); }, heuresis::search::Budget{92 * 250});
    const double igd2 = inverted_generational_distance(a.front(), sphere_targets(w));
    s.check("NSGA-III on DTLZ2 (3 objectives, 250 generations): IGD to the reference points on the front", igd2, 0.01);
    Nsga3<Real> c(real_kernels(b1, 30, 20, 1.0), w, 92, Draws(4));
    heuresis::search::run(c, [](const Real& x) { return heuresis::search::evaluation(heuresis::benchmarks::dtlz1(x, 3)); }, heuresis::search::Budget{92 * 400});
    const double igd1 = inverted_generational_distance(c.front(), simplex_targets(w));
    s.check("NSGA-III on DTLZ1 (3 objectives, 400 generations): IGD to the reference points on the front", igd1, 0.02);
  }

  // MOEA/D on DTLZ2 with PBI, and on ZDT1 with Tchebycheff.
  {
    const auto w = das_dennis(3, 13);
    Moead<Real>::Options o;
    o.scalarisation = Scalarisation::pbi;
    Moead<Real> a(real_kernels(Box::cube(12, 0, 1), 20, 20, 1.0), w, o, Draws(5));
    heuresis::search::run(a, [](const Real& x) { return heuresis::search::evaluation(heuresis::benchmarks::dtlz2(x, 3)); }, heuresis::search::Budget{static_cast<double>(w.size() * 250)});
    s.check("MOEA/D (PBI) on DTLZ2 (3 objectives, 250 generations): IGD to the weights' points on the front", inverted_generational_distance(a.front(), sphere_targets(w)), 0.01);
    Moead<Real> z(real_kernels(zdt_box), das_dennis(2, 99), {}, Draws(6));
    heuresis::search::run(z, zdt1, heuresis::search::Budget{20000});
    s.check("MOEA/D (Tchebycheff) on ZDT1 (20 000 evaluations): HV within 2% of 2/3", 1 - zdt1_hv(z.front()) / heuresis::benchmarks::zdt1_hypervolume(), 0.02);
  }

  // SMS-EMOA on ZDT1, steady state.
  {
    SmsEmoa<Real> a(real_kernels(zdt_box), 100, Draws(7));
    heuresis::search::run(a, zdt1, heuresis::search::Budget{20000});
    const double hv = zdt1_hv(a.front());
    s.check("SMS-EMOA on ZDT1 (20 000 evaluations): HV within 1.5% of 2/3", 1 - hv / heuresis::benchmarks::zdt1_hypervolume(), 0.015);
    s.require("SMS-EMOA keeps a population of μ", a.population().size() == 100);
  }

  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  s.check("the whole check runs within 20 s", seconds, 20, "s");
  return s.done();
}
