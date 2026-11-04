// Checks of heuresis diversity: grids and Voronoi cells map points to their cells; an archive keeps the better and
// its coverage and QD-score never decrease; MAP-Elites and CVT-MAP-Elites illuminate the planar arm's reachable
// disc and nothing outside it; novelty search visits more of the disc than random search with the same budget,
// its archive only growing; fitness sharing's equilibrium and clearing's winners per niche, exactly.

#include <heuresis/benchmarks/benchmarks.hpp>
#include <heuresis/diversity/diversity.hpp>
#include <heuresis/search/search.hpp>
#include <heuresis/variation/variation.hpp>

#include <calx/verification/check.hpp>

#include <chrono>
#include <cmath>
#include <numbers>
#include <set>

using namespace heuresis::diversity;
using heuresis::search::Draws;
using heuresis::search::tag;
using heuresis::variation::Box;
using heuresis::variation::Real;

namespace {

// Whether a cell of a grid on [0, 1]² meets the open disc of radius ½ about (½, ½), the arm's reachable set.
bool reachable(const Grid& g, std::size_t c) {
  const auto x = g.centre(c);
  const double hx = 0.5 / static_cast<double>(g.resolution()[0]), hy = 0.5 / static_cast<double>(g.resolution()[1]);
  const double dx = std::max({x[0] - hx - 0.5, 0.0, 0.5 - (x[0] + hx)}), dy = std::max({x[1] - hy - 0.5, 0.0, 0.5 - (x[1] + hy)});
  return std::hypot(dx, dy) < 0.5;
}

Kernels<Real> arm_kernels(std::size_t d) {
  const Box box = Box::cube(d, -std::numbers::pi, std::numbers::pi);
  Kernels<Real> k;
  k.initialise = [box](calx::random::Stream& s) { return heuresis::variation::uniform_point(box, s); };
  k.mutate = [box](const Real& x, calx::random::Stream& s) {
    Real y = heuresis::variation::gaussian_mutation(x, 0.1, s);
    heuresis::variation::repair(y, box, heuresis::variation::Repair::reflect, s);
    return y;
  };
  k.vary = [box](const Real& a, const Real& b, calx::random::Stream& s) {
    Real y = heuresis::variation::iso_line(a, b, 0.05, 0.3, s);
    heuresis::variation::repair(y, box, heuresis::variation::Repair::reflect, s);
    return y;
  };
  return k;
}

heuresis::search::Evaluation arm(const Real& x) { return heuresis::benchmarks::planar_arm(x); }

}  // namespace

int main() {
  calx::verification::Suite s;
  const Draws draws(2015);
  const auto start = std::chrono::steady_clock::now();
  const Grid grid({0, 0}, {1, 1}, {20, 20});
  std::size_t reachable_cells = 0;
  for (std::size_t c = 0; c < grid.cells(); c++) reachable_cells += reachable(grid, c);

  // Tessellations.
  {
    bool centres = true;
    for (std::size_t c = 0; c < grid.cells(); c++) centres = centres && grid.cell(grid.centre(c)) == c;
    s.require("a grid maps each cell's centre to the cell", centres);
    s.require("a grid puts points outside its box in the boundary cells", grid.cell(std::vector<double>{-3, 0.01}) == 0 && grid.cell(std::vector<double>{7, 9}) == grid.cells() - 1);
    std::vector<std::vector<double>> sites;
    for (int t = 0; t < 50; t++) { auto r = draws(tag("check.div.sites"), 0, t); sites.push_back({r.uniform(), r.uniform(), r.uniform()}); }
    const Voronoi v(sites);
    bool own = true;
    for (std::size_t j = 0; j < sites.size(); j++) own = own && v.cell(sites[j]) == j;
    s.require("a Voronoi tessellation maps each site to its own cell", own);
  }

  // The archive keeps the better; coverage and QD-score never decrease.
  {
    Archive<int, Grid> a(Grid({0}, {1}, {10}));
    using I = heuresis::search::Individual<int>;
    auto ind = [](int id, double f, double d) { return I{static_cast<heuresis::search::Ticket>(id), id, heuresis::search::evaluation({f}, {}, {d}), {}}; };
    const bool first = a.offer(ind(1, 5, 0.31)), worse = a.offer(ind(2, 6, 0.33)), better = a.offer(ind(3, 4, 0.39)), failed = a.offer({4, 4, heuresis::search::failure("x"), {}});
    s.require("an archive fills a cell, refuses the worse and a failure, takes the better", first && !worse && better && !failed && a.elite(3)->genotype == 3 && a.size() == 1);
    bool monotone = true;
    double cov = a.coverage(), qd = a.qd_score(10);
    for (int t = 0; t < 2000; t++) {
      auto r = draws(tag("check.div.archive"), 0, t);
      a.offer(ind(10 + t, 10 * r.uniform(), r.uniform()));
      monotone = monotone && a.coverage() >= cov && a.qd_score(10) >= qd;
      cov = a.coverage();
      qd = a.qd_score(10);
    }
    s.require("coverage and QD-score never decrease", monotone && a.coverage() == 1);
  }

  // MAP-Elites on the planar arm of eight links.
  {
    MapElites<Real, Grid>::Options o;
    o.initial = 200;
    o.variation_probability = 0.5;
    MapElites<Real, Grid> me(arm_kernels(8), grid, o, Draws(1));
    bool monotone = true;
    double qd = 0;
    for (int chunk = 0; chunk < 10; chunk++) {
      heuresis::search::run(me, arm, heuresis::search::Budget{3000});
      monotone = monotone && me.archive().qd_score(1e3) >= qd;
      qd = me.archive().qd_score(1e3);
    }
    std::size_t covered = 0, outside = 0;
    for (auto c : me.archive().filled()) (reachable(grid, c) ? covered : outside)++;
    s.check("MAP-Elites (30 000 evaluations) fills the cells of the reachable disc", 1 - static_cast<double>(covered) / static_cast<double>(reachable_cells), 0.05);
    s.require("MAP-Elites places no elite outside the reachable disc", outside == 0);
    s.require("MAP-Elites: the QD-score never decreases over the run", monotone);
    // The elite nearest the edge of the disc is nearly straight.
    double best_edge = INFINITY;
    for (auto c : me.archive().filled()) {
      const auto x = grid.centre(c);
      if (std::hypot(x[0] - 0.5, x[1] - 0.5) > 0.45) best_edge = std::min(best_edge, me.archive().elite(c)->evaluation.value());
    }
    s.check("MAP-Elites: elites at the edge of the disc are nearly straight (angle variance)", best_edge, 0.05);
  }

  // CVT-MAP-Elites with Voronoi cells of sites inside the disc (any sites tessellate; k-means sites wait for tyche).
  {
    std::vector<std::vector<double>> sites;
    for (int t = 0; sites.size() < 200; t++) {
      auto r = draws(tag("check.div.cvt"), 0, t);
      const double x = r.uniform(), y = r.uniform();
      if (std::hypot(x - 0.5, y - 0.5) < 0.45) sites.push_back({x, y});
    }
    MapElites<Real, Voronoi>::Options o;
    o.initial = 200;
    o.variation_probability = 0.5;
    MapElites<Real, Voronoi> me(arm_kernels(8), Voronoi(sites), o, Draws(2));
    heuresis::search::run(me, arm, heuresis::search::Budget{20000});
    s.check("CVT-MAP-Elites (20 000 evaluations) fills its Voronoi cells", 1 - me.archive().coverage(), 0.05);
  }

  // Novelty search visits more of the disc than random search with the same budget.
  {
    NoveltySearch<Real>::Options o;
    o.population = 100;
    o.threshold = 0.02;
    NoveltySearch<Real> ns(arm_kernels(8), o, Draws(3));
    heuresis::search::Record<Real> rec;
    std::size_t archive = 0;
    bool grows = true;
    for (int chunk = 0; chunk < 10; chunk++) {
      heuresis::search::Record<Real> part;
      heuresis::search::run(ns, arm, heuresis::search::Budget{1000}, heuresis::search::Serial{}, heuresis::search::unlimited, &part);
      rec.told.insert(rec.told.end(), part.told.begin(), part.told.end());
      grows = grows && ns.archive().size() >= archive;
      archive = ns.archive().size();
    }
    const Box box = Box::cube(8, -std::numbers::pi, std::numbers::pi);
    heuresis::search::RandomSearch<Real> rs([box](calx::random::Stream& r) { return heuresis::variation::uniform_point(box, r); }, Draws(3));
    heuresis::search::Record<Real> rrec;
    heuresis::search::run(rs, arm, heuresis::search::Budget{10000}, heuresis::search::Serial{}, heuresis::search::unlimited, &rrec);
    auto visited = [&](const heuresis::search::Record<Real>& r) {
      std::set<std::size_t> cells;
      for (const auto& x : r.told) cells.insert(grid.cell(x.evaluation.descriptors));
      return static_cast<double>(cells.size()) / static_cast<double>(reachable_cells);
    };
    const double vn = visited(rec), vr = visited(rrec);
    s.require("novelty search: the archive only grows", grows && archive > 0);
    s.check("novelty search (10 000 evaluations) visits most of the reachable disc", 1 - vn, 0.1);
    s.check("random search with the same budget visits less of the disc (its share; the limit is novelty's less 0.1)", vr, vn - 0.1);
  }

  // Fitness sharing and clearing.
  {
    // 20 copies on a peak of fitness 1, 10 on a peak of fitness ½, the peaks farther apart than σ.
    std::vector<double> f(30), x(30);
    for (int i = 0; i < 30; i++) { f[i] = i < 20 ? 1.0 : 0.5; x[i] = i < 20 ? 0.0 : 5.0; }
    const Distance d = [&](std::size_t i, std::size_t j) { return std::fabs(x[i] - x[j]); };
    const auto sh = shared_fitness(f, d, 1.0);
    s.check("sharing: shares proportional to the peaks equalise the shared fitness (20 : 10 for 1 : ½)", sh[0] - sh[25], 1e-17);
    s.check("sharing: the triangular function at d = σ/4 is 3/4", sharing_function(0.25, 1) - 0.75, 1e-16);
    std::vector<double> v, y;
    for (int c = 0; c < 3; c++)
      for (int k = 0; k < 5 + c; k++) { y.push_back(10.0 * c + 0.01 * k); v.push_back(static_cast<double>(k) + 0.5 * c); }
    const Distance dy = [&](std::size_t i, std::size_t j) { return std::fabs(y[i] - y[j]); };
    const auto w = clearing(v, dy, 1.0, 2);
    bool two_best = true;
    std::size_t idx = 0;
    for (int c = 0; c < 3; c++)
      for (int k = 0; k < 5 + c; k++, idx++) two_best = two_best && (w[idx] == (k < 2));
    s.require("clearing keeps the κ = 2 best of each niche and clears the rest", two_best);
  }

  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  s.check("the whole check runs within 20 s", seconds, 20, "s");
  return s.done();
}
