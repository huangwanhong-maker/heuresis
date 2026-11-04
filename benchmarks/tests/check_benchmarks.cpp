// Checks of heuresis benchmarks: each function's stated minimum at its minimiser and nowhere undercut by random
// points; rotations orthogonal; the fronts of ZDT and DTLZ reached where the theory says, and their closed-form
// hypervolumes against Monte Carlo; the pseudo-Boolean functions' optima, NK's separability at k = 0 against
// enumeration, QUBO's flip deltas against differences, the planted assignment satisfying every clause.

#include <heuresis/benchmarks/benchmarks.hpp>

#include <calx/verification/check.hpp>

#include <cmath>
#include <functional>

using namespace heuresis::benchmarks;
using heuresis::search::Draws;
using heuresis::search::tag;
using heuresis::variation::Bits;

int main() {
  calx::verification::Suite s;
  const Draws draws(5);

  // Minima of the continuous functions.
  {
    struct Case {
      const char* name;
      std::function<double(std::span<const double>)> f;
      std::vector<double> x;
      double fstar, lo, hi;
    };
    const double pi = std::numbers::pi;
    const std::vector<Case> cases{
        {"sphere", sphere, {0, 0, 0, 0}, 0, -5, 5},
        {"ellipsoid", ellipsoid, {0, 0, 0, 0}, 0, -5, 5},
        {"rosenbrock", rosenbrock, {1, 1, 1, 1}, 0, -2, 2},
        {"rastrigin", rastrigin, {0, 0, 0, 0}, 0, -5.12, 5.12},
        {"ackley", ackley, {0, 0, 0, 0}, 0, -5, 5},
        {"griewank", griewank, {0, 0, 0, 0}, 0, -600, 600},
        {"schwefel", schwefel, std::vector<double>(4, 420.9687463), 0, -500, 500},
        {"styblinski_tang", styblinski_tang, std::vector<double>(4, -2.903534027771178), -39.16616570377142 * 4, -5, 5},
        {"levy", levy, {1, 1, 1, 1}, 0, -10, 10},
        {"branin at (−π, 12.275)", branin, {-pi, 12.275}, 0.397887357729738, -5, 15},
        {"branin at (π, 2.275)", branin, {pi, 2.275}, 0.397887357729738, -5, 15},
        {"six_hump_camel", six_hump_camel, {0.08984201368301331, -0.7126564032704135}, -1.031628453489877, -2, 2},
        {"hartmann3", hartmann3, {0.114614, 0.555649, 0.852547}, -3.86278214782076, 0, 1},
        {"hartmann6", hartmann6, {0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573}, -3.32236801141551, 0, 1},
    };
    for (const auto& c : cases) {
      s.check(std::string(c.name) + ": f(x*) = f*", c.f(c.x) - c.fstar, 1e-5 * std::max(1.0, std::fabs(c.fstar)));
      double least = INFINITY;
      for (int t = 0; t < 20000; t++) {
        auto r = draws(tag("check.bench.random"), 0, t);
        std::vector<double> x(c.x.size());
        for (auto& v : x) v = r.uniform(c.lo, c.hi);
        least = std::min(least, c.f(x));
      }
      s.require(std::string(c.name) + ": no random point below f*", least >= c.fstar - 1e-6);
    }
    s.check("Schwefel's constant leaves f* within 10⁻⁸ of 0 per coordinate", schwefel(std::vector<double>(1, 420.968746359982)), 1e-8);
    const auto g = g06(std::vector<double>{14.095, 0.84296078921547957});
    s.check("G06: f* = −6961.81387558015", g.objectives[0] + 6961.81387558015, 1e-6);
    s.require("G06: the optimum is feasible to rounding", g.violation() < 1e-9);
    s.require("disc Rosenbrock: (1, 1) on the boundary", disc_rosenbrock(std::vector<double>{1, 1}).feasible() && !disc_rosenbrock(std::vector<double>{1.1, 1}).feasible());
  }

  // Rotations: orthogonal, and moving the minimiser where the transformation says.
  {
    const Rotation R(7, 99);
    double off = 0;
    for (std::size_t i = 0; i < 7; i++)
      for (std::size_t j = 0; j < 7; j++) {
        double d = 0;
        for (std::size_t k = 0; k < 7; k++) d += R(k, i) * R(k, j);
        off = std::max(off, std::fabs(d - (i == j)));
      }
    s.check("a rotation is orthogonal", off, 1e-14);
    const std::vector<double> shift{0.3, -1, 2, 0.5, 0, 1, -0.2};
    const auto f = transformed(ellipsoid, shift, R);
    s.check("a shifted, rotated ellipsoid is least at its shift", f(shift), 1e-30);
    double mean = 0;
    for (int t = 0; t < 4000; t++) { const Rotation Q(5, 1, t); mean += Q(0, 0) * Q(0, 0); }
    s.check("Haar measure: E[R₀₀²] = 1/n", mean / 4000 - 0.2, 0.01);
  }

  // Fronts and their hypervolumes.
  {
    std::vector<double> x(30, 0.0);
    x[0] = 0.36;
    const auto f1 = zdt1(x), f2 = zdt2(x);
    s.check("ZDT1 at g = 1 lies on f₂ = 1 − √f₁", f1[1] - (1 - 0.6), 1e-15);
    s.check("ZDT2 at g = 1 lies on f₂ = 1 − f₁²", f2[1] - (1 - 0.1296), 1e-15);
    std::vector<double> y(12, 0.5);
    y[0] = 0.3;
    y[1] = 0.8;
    const auto d1 = dtlz1(y, 3), d2 = dtlz2(y, 3), d4 = dtlz4(y, 3);
    s.check("DTLZ1 at x_M = ½ lies on Σ fᵢ = ½", d1[0] + d1[1] + d1[2] - 0.5, 1e-15);
    s.check("DTLZ2 at x_M = ½ lies on the unit sphere", d2[0] * d2[0] + d2[1] * d2[1] + d2[2] * d2[2] - 1, 1e-15);
    s.check("DTLZ4 at x_M = ½ lies on the unit sphere", d4[0] * d4[0] + d4[1] * d4[1] + d4[2] * d4[2] - 1, 1e-15);
    s.check("DTLZ2's hypervolume in three objectives is 1 − π/6", dtlz2_hypervolume(3) - (1 - std::numbers::pi / 6), 1e-15);
    s.check("DTLZ1's hypervolume in three objectives is 1 − 1/48", dtlz1_hypervolume(3) - (1 - 1.0 / 48), 1e-15);
    s.check("ZDT6's least f₁ is 0.2807753191", zdt6_least_f1() - 0.2807753191, 1e-9);
    // Monte Carlo: the fraction of the unit box dominated by each front.
    const int N = 400000;
    double z1 = 0, z2 = 0, z6 = 0, s2 = 0, s1 = 0;
    const double f6 = zdt6_least_f1();
    for (int t = 0; t < N; t++) {
      auto r = draws(tag("check.hv"), 0, t);
      const double a = r.uniform(), b = r.uniform(), c = r.uniform();
      z1 += b >= 1 - std::sqrt(a);
      z2 += b >= 1 - a * a;
      z6 += a >= f6 && b >= 1 - a * a;
      s2 += a * a + b * b + c * c >= 1;
      s1 += a + b + c >= 0.5;
    }
    s.check("ZDT1's hypervolume 2/3 against Monte Carlo", z1 / N - zdt1_hypervolume(), 0.003);
    s.check("ZDT2's hypervolume 1/3 against Monte Carlo", z2 / N - zdt2_hypervolume(), 0.003);
    s.check("ZDT6's hypervolume against Monte Carlo", z6 / N - zdt6_hypervolume(), 0.003);
    s.check("DTLZ2's hypervolume against Monte Carlo", s2 / N - dtlz2_hypervolume(3), 0.003);
    s.check("DTLZ1's hypervolume against Monte Carlo", s1 / N - dtlz1_hypervolume(3), 0.003);
  }

  // Pseudo-Boolean functions.
  {
    const Bits all(20, 1);
    Bits valley(20, 1);
    valley[0] = valley[1] = 0;
    Bits prefix(20, 1);
    prefix[5] = 0;
    s.require("OneMax, LeadingOnes, Jump and Trap are 0 at the all-ones string", onemax(all) == 0 && leading_ones(all) == 0 && jump(all, 3) == 0 && trap(all, 4) == 0);
    s.require("Jump₃: the valley is worse than the string of zeros", jump(valley, 3) > jump(Bits(20, 0), 3));
    s.require("LeadingOnes counts the prefix", leading_ones(prefix) == 15);
    s.require("Trap: a block of zeros is the deceptive attractor", trap(Bits(4, 0), 4) == 1 && trap(Bits{1, 1, 1, 0}, 4) == 4);

    // NK with k = 0 is separable: its optimum is each bit's best, found by enumeration.
    const NK nk(10, 0, 3);
    double best = INFINITY;
    for (std::size_t m = 0; m < 1024; m++) {
      Bits x(10);
      for (std::size_t i = 0; i < 10; i++) x[i] = (m >> i) & 1;
      best = std::min(best, nk(x));
    }
    Bits greedy(10);
    for (std::size_t i = 0; i < 10; i++) {
      Bits a = greedy, b = greedy;
      a[i] = 0;
      b[i] = 1;
      greedy[i] = nk(b) < nk(a);
    }
    s.check("NK at k = 0: the bitwise optimum is the global one", nk(greedy) - best, 1e-15);
    const NK nk3(12, 3, 4);
    s.require("NK: every bit has k distinct neighbours, none itself", [&] {
      for (std::size_t i = 0; i < 12; i++) {
        auto nb = nk3.neighbours(i);
        std::sort(nb.begin(), nb.end());
        if (nb.size() != 3 || std::adjacent_find(nb.begin(), nb.end()) != nb.end() || std::find(nb.begin(), nb.end(), i) != nb.end()) return false;
      }
      return true;
    }());

    const Qubo q = Qubo::random(9, 2);
    double dev = 0;
    for (int t = 0; t < 200; t++) {
      auto r = draws(tag("check.qubo"), 0, t);
      Bits x = heuresis::variation::random_bits(9, r);
      const std::size_t i = r.below(9);
      Bits y = x;
      y[i] ^= 1;
      dev = std::max(dev, std::fabs(q.flip_delta(x, i) - (q(y) - q(x))));
    }
    s.check("QUBO: the flip delta is the difference of values", dev, 1e-12);

    const PlantedSat sat(30, 120, 3, 8);
    s.require("planted 3-SAT: the hidden assignment satisfies every clause", sat(sat.hidden()) == 0 && sat.clauses() == 120);
  }

  // The tour and the arm.
  {
    const Tour square({{0, 0}, {1, 0}, {1, 1}, {0, 1}});
    s.check("the tour of a unit square's corners has length 4", square.length({0, 1, 2, 3}) - 4, 1e-15);
    s.check("its crossed tour has length 2 + 2√2", square.length({0, 2, 1, 3}) - (2 + 2 * std::sqrt(2.0)), 1e-15);
    const auto arm = planar_arm(std::vector<double>(6, 0.0));
    s.require("the straight arm reaches (1, 0), its descriptor (1, ½), its variance 0", std::fabs(arm.descriptors[0] - 1) < 1e-15 && std::fabs(arm.descriptors[1] - 0.5) < 1e-15 && arm.objectives[0] == 0);
  }

  return s.done();
}
