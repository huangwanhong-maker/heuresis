// Checks of heuresis local: best improvement on OneMax spends exactly 1 + n(z + 1) evaluations; climbers stop at
// local optima found by enumeration; tabu search reaches QUBO's enumerated optimum and planted SAT's zero; VND
// stops at a point optimal for both of its neighbourhoods; iterated local search finds the optimal tour of nine
// points found by enumeration; 2-opt leaves no crossing edges; memetic refinement through calx.

#include <heuresis/benchmarks/benchmarks.hpp>
#include <heuresis/local/local.hpp>

#include <calx/verification/check.hpp>

#include <algorithm>
#include <cmath>
#include <set>

using namespace heuresis::local;
using namespace heuresis::search;
using heuresis::benchmarks::onemax;

namespace {

Bits bits_of(std::size_t m, std::size_t n) {
  Bits x(n);
  for (std::size_t i = 0; i < n; i++) x[i] = (m >> i) & 1u;
  return x;
}
std::size_t index_of(const Bits& x) {
  std::size_t m = 0;
  for (std::size_t i = 0; i < x.size(); i++) m |= static_cast<std::size_t>(x[i]) << i;
  return m;
}

// Strict proper crossing of segments ab and cd.
bool crosses(std::pair<double, double> a, std::pair<double, double> b, std::pair<double, double> c, std::pair<double, double> d) {
  auto orient = [](auto p, auto q, auto r) { return (q.first - p.first) * (r.second - p.second) - (q.second - p.second) * (r.first - p.first); };
  const double o1 = orient(a, b, c), o2 = orient(a, b, d), o3 = orient(c, d, a), o4 = orient(c, d, b);
  return o1 * o2 < 0 && o3 * o4 < 0;
}

}  // namespace

int main() {
  calx::verification::Suite s;

  // Best improvement on OneMax: z moves, n neighbours each, and a last scan that finds none.
  {
    const std::size_t n = 20;
    bool exact = true;
    for (int t = 0; t < 20; t++) {
      calx::random::Stream r = Draws(1)(tag("check.local.start"), 0, t);
      const Bits x = heuresis::variation::random_bits(n, r);
      const double z = onemax(x);
      HillClimber<Bits> h(x, one_flip(), Pivot::best);
      const auto o = run(h, onemax, Budget{});
      exact = exact && o.ending == Ending::stopped && o.best->evaluation.value() == 0 && o.evaluations == 1 + n * (z + 1) && h.steps() == z;
    }
    s.require("best improvement on OneMax: z moves and exactly 1 + n(z + 1) evaluations", exact);
  }

  // First improvement: OneMax solved by z moves; telling a chunk in any order changes nothing.
  {
    const std::size_t n = 30;
    calx::random::Stream r = Draws(2)(tag("check.local.start"), 0, 0);
    const Bits x = heuresis::variation::random_bits(n, r);
    HillClimber<Bits> a(x, one_flip(), Pivot::first, 5, Draws(3)), b(x, one_flip(), Pivot::first, 5, Draws(3));
    const auto oa = run(a, onemax, Budget{});
    while (!b.stopped()) {
      auto c = b.ask(unlimited);
      std::vector<Told> t;
      for (auto& y : c) t.push_back({y.ticket, scalar(onemax(y.genotype))});
      std::reverse(t.begin(), t.end());
      b.tell(t);
    }
    s.require("first improvement on OneMax: the optimum in z moves", oa.best->evaluation.value() == 0 && a.steps() == onemax(x));
    s.require("first improvement: the same climb whatever the order of telling", b.current()->ticket == a.current()->ticket);
  }

  // Climbers stop at local optima of a QUBO, as enumeration finds them.
  const std::size_t qn = 14;
  const auto q = heuresis::benchmarks::Qubo::random(qn, 21);
  std::vector<double> values(std::size_t{1} << qn);
  for (std::size_t m = 0; m < values.size(); m++) values[m] = q(bits_of(m, qn));
  std::set<std::size_t> optima;
  for (std::size_t m = 0; m < values.size(); m++) {
    bool local = true;
    for (std::size_t i = 0; i < qn && local; i++) local = values[m] <= values[m ^ (std::size_t{1} << i)];
    if (local) optima.insert(m);
  }
  {
    bool at_optima = true;
    for (int t = 0; t < 30; t++) {
      calx::random::Stream r = Draws(4)(tag("check.local.start"), 0, t);
      HillClimber<Bits> h(heuresis::variation::random_bits(qn, r), one_flip(), t % 2 ? Pivot::first : Pivot::best, 3, Draws(t));
      run(h, q, Budget{});
      at_optima = at_optima && h.stopped() && optima.count(index_of(h.current()->genotype));
    }
    s.require("hill climbers stop at enumerated one-flip local optima of a QUBO", at_optima);
  }

  // Tabu search: the QUBO's optimum by enumeration; planted 3-SAT satisfied.
  {
    TabuSearch<Bits> tabu(Bits(16, 0), one_flip(), 4, Draws(5));
    const auto q16 = heuresis::benchmarks::Qubo::random(16, 22);
    double best16 = INFINITY;
    for (std::size_t m = 0; m < (std::size_t{1} << 16); m++) best16 = std::min(best16, q16(bits_of(m, 16)));
    const auto o = run(tabu, q16, Budget{5000, INFINITY, -INFINITY});
    s.check("tabu search reaches a QUBO's enumerated optimum (n = 16, 5000 evaluations)", o.best->evaluation.value() - best16, 1e-12);
    const heuresis::benchmarks::PlantedSat sat(40, 160, 3, 9);
    TabuSearch<Bits> t2(Bits(40, 0), one_flip(), 7, Draws(6));
    const auto o2 = run(t2, sat, Budget{100000, INFINITY, 0});
    s.require("tabu search satisfies a planted 3-SAT (n = 40, m = 160)", o2.ending == Ending::target);
  }

  // VND with one and two flips: a point optimal in both neighbourhoods.
  {
    bool both = true;
    for (int t = 0; t < 10; t++) {
      calx::random::Stream r = Draws(7)(tag("check.local.start"), 0, t);
      VariableNeighbourhoodDescent<Bits> v(heuresis::variation::random_bits(qn, r), {one_flip(), two_flip()});
      run(v, q, Budget{});
      const std::size_t m = index_of(v.current()->genotype);
      both = both && v.stopped() && optima.count(m);
      for (std::size_t i = 0; i < qn; i++)
        for (std::size_t j = i + 1; j < qn; j++) both = both && values[m] <= values[m ^ (std::size_t{1} << i) ^ (std::size_t{1} << j)];
    }
    s.require("VND stops at points optimal for one and two flips", both);
  }

  // Iterated local search with 2-opt and the double bridge finds the optimal tour of nine points.
  {
    const auto tour = heuresis::benchmarks::Tour::random(9, 31);
    heuresis::variation::Permutation p = heuresis::variation::identity_permutation(9);
    double optimum = INFINITY;
    do optimum = std::min(optimum, tour.length(p));
    while (std::next_permutation(p.begin() + 1, p.end()));
    IteratedLocalSearch<heuresis::variation::Permutation> ils(heuresis::variation::identity_permutation(9), two_opt(), double_bridge, Pivot::best, Acceptance::not_worse, Draws(8));
    run(ils, tour, Budget{20000, INFINITY, -INFINITY});
    s.check("iterated local search finds the enumerated optimal tour (n = 9)", ils.best()->evaluation.value() - optimum, 1e-12);
    s.require("iterated local search kicked more than once", ils.kicks() > 1);
  }

  // 2-opt descent: no two edges of the result cross; the tour is shorter.
  {
    const auto tour = heuresis::benchmarks::Tour::random(80, 41);
    auto p = heuresis::variation::identity_permutation(80);
    const double before = tour.length(p);
    two_opt_descent(p, [&](std::size_t a, std::size_t b) { return tour.distance(a, b); });
    const auto& pt = tour.points();
    bool none = true;
    for (std::size_t i = 0; i < 80; i++)
      for (std::size_t j = i + 2; j < 80; j++) {
        if (i == 0 && j == 79) continue;
        none = none && !crosses(pt[p[i]], pt[p[i + 1]], pt[p[j]], pt[p[(j + 1) % 80]]);
      }
    s.require("a 2-opt local optimum has no crossing edges (80 points)", none && heuresis::variation::is_permutation(p));
    s.require("2-opt shortened the tour", tour.length(p) < 0.5 * before);
  }

  // Memetic refinement: Baldwinian values and costs, the Lamarckian point.
  {
    std::size_t calls = 0;
    std::function<double(std::span<const double>)> f = [&](std::span<const double> x) {
      calls++;
      return heuresis::benchmarks::rosenbrock(x);
    };
    const auto b = baldwinian(f);
    const auto e = b(std::vector<double>{-1.2, 1});
    s.check("Baldwinian Nelder–Mead on Rosenbrock from (−1.2, 1)", e.value(), 1e-8);
    s.require("the Baldwinian cost is the number of evaluations spent", e.cost == static_cast<double>(calls));
    const Individual<std::vector<double>> x{1, {-1.2, 1}, scalar(24.2), {}};
    const auto y = lamarckian(f, x, 2, RefineOptions{Refiner::bfgs});
    s.check("Lamarckian BFGS reaches (1, 1)", std::hypot(y.genotype[0] - 1, y.genotype[1] - 1), 1e-4);
    s.require("the Lamarckian individual's parent is the original", y.origin.parents == std::vector<Ticket>{1} && y.origin.operation == refine_tag);
  }

  return s.done();
}
