// Checks of heuresis populations: the topologies' degrees; takeover times of a cellular algorithm under
// selection alone equal to the start's eccentricity, found by breadth-first search and in closed form; the
// synchronous island model the same on one thread and on a pool; migrants carrying their island in their
// tickets; the asynchronous island model with equal durations and no latency exactly the synchronous one, and
// reproducible with random durations and latencies.

#include <heuresis/benchmarks/benchmarks.hpp>
#include <heuresis/genetic/genetic.hpp>
#include <heuresis/populations/populations.hpp>

#include <calx/verification/check.hpp>

#include <deque>

using namespace heuresis::populations;
using namespace heuresis::search;
using heuresis::benchmarks::onemax;
using heuresis::variation::Bits;

namespace {

std::size_t eccentricity(const Topology& t, std::size_t s) {
  std::vector<std::size_t> d(t.size(), SIZE_MAX);
  std::deque<std::size_t> q{s};
  d[s] = 0;
  std::size_t e = 0;
  while (!q.empty()) {
    const std::size_t v = q.front();
    q.pop_front();
    e = std::max(e, d[v]);
    for (auto w : t.out[v])
      if (d[w] == SIZE_MAX) { d[w] = d[v] + 1; q.push_back(w); }
  }
  return e;
}

// Generations until the best label, 0, placed at s, fills every cell under selection alone.
std::size_t takeover(const Topology& t, std::size_t s) {
  std::vector<long> labels(t.size());
  for (std::size_t i = 0; i < t.size(); i++) labels[i] = static_cast<long>(i < s ? i + 1 : i);
  labels[s] = 0;
  CellularEA<long> ca(labels, t, CellSelection::best);
  auto f = [](long x) { return static_cast<double>(x); };
  for (std::size_t g = 0;; g++) {
    auto c = ca.ask(unlimited);
    std::vector<Told> told;
    for (auto& x : c) told.push_back({x.ticket, scalar(f(x.genotype))});
    ca.tell(told);
    bool all = true;
    for (const auto& x : ca.cells()) all = all && x.genotype == 0;
    if (all) return g;  // generation 0 is the initial evaluation
    if (g > 10 * t.size()) return SIZE_MAX;
  }
}

using GA = heuresis::genetic::GeneticAlgorithm<Bits>;

std::vector<GA> archipelago(std::size_t k, std::size_t n, std::uint64_t seed) {
  heuresis::genetic::Operators<Bits> op{heuresis::genetic::random_bits(n), heuresis::genetic::bit_flip(), heuresis::genetic::uniform_crossover<std::uint8_t>()};
  heuresis::genetic::GaOptions o;
  o.mu = 10;
  std::vector<GA> v;
  for (std::size_t i = 0; i < k; i++) v.emplace_back(op, o, Draws(seed, static_cast<std::uint32_t>(i)));
  return v;
}

std::vector<std::vector<Ticket>> populations_of(const std::vector<GA>& v) {
  std::vector<std::vector<Ticket>> out;
  for (const auto& g : v) {
    out.emplace_back();
    for (const auto& x : g.population()) out.back().push_back(x.ticket);
  }
  return out;
}

}  // namespace

int main() {
  calx::verification::Suite s;

  // Topologies.
  {
    auto degrees = [](const Topology& t, std::size_t d) {
      for (const auto& o : t.out)
        if (o.size() != d) return false;
      return true;
    };
    s.require("ring of 20: two neighbours each", degrees(Topology::ring(20), 2) && degrees(Topology::ring(20, false), 1));
    s.require("torus 8 × 8: four (von Neumann) and eight (Moore) neighbours", degrees(Topology::torus(8, 8), 4) && degrees(Topology::torus(8, 8, true), 8));
    s.require("hypercube of dimension 4, complete graph of 6, star of 5", degrees(Topology::hypercube(4), 4) && degrees(Topology::complete(6), 5) && Topology::star(5).out[0].size() == 4);
  }

  // Takeover times equal eccentricities.
  {
    struct Case {
      const char* name;
      Topology t;
      std::size_t start, closed;
    };
    const std::vector<Case> cases{
        {"ring of 20: ⌈19/2⌉ = 10", Topology::ring(20), 3, 10},
        {"torus 8 × 8, von Neumann: 4 + 4", Topology::torus(8, 8), 27, 8},
        {"torus 8 × 8, Moore: 4", Topology::torus(8, 8, true), 27, 4},
        {"torus 6 × 9, von Neumann: 3 + 4", Topology::torus(6, 9), 10, 7},
        {"hypercube of dimension 5: 5", Topology::hypercube(5), 13, 5},
        {"complete graph of 16: 1", Topology::complete(16), 7, 1},
    };
    for (const auto& c : cases) {
      const std::size_t e = eccentricity(c.t, c.start);
      s.require(std::string("takeover time equals the eccentricity: ") + c.name, takeover(c.t, c.start) == e && e == c.closed);
    }
  }

  // The synchronous island model: one thread or a pool, the same run; migrants keep their island.
  {
    const Topology ring = Topology::ring(4);
    const Migration m{2, 2};
    auto a = archipelago(4, 60, 11), b = archipelago(4, 60, 11);
    const auto oa = run_islands(a, ring, m, onemax, Budget{4000, INFINITY, -INFINITY});
    calx::scheduler::ThreadPool pool(4);
    const auto ob = run_islands(b, ring, m, onemax, Budget{4000, INFINITY, -INFINITY}, &pool);
    s.require("islands: a pool of threads gives the same run", populations_of(a) == populations_of(b) && oa.best->ticket == ob.best->ticket && oa.evaluations == ob.evaluations);
    s.require("islands: the budget is shared exactly", oa.evaluations == 4000 && oa.island_evaluations == std::vector<double>(4, 1000));
    bool foreign = false;
    for (std::size_t i = 0; i < a.size(); i++)
      for (const auto& x : a[i].population()) foreign = foreign || island_of(x.ticket) != i;
    s.require("islands: migrants carry their island in their tickets", foreign && oa.migrations > 0);

    // The asynchronous model with equal durations and no latency is the synchronous one.
    auto c = archipelago(4, 60, 11);
    const auto oc = run_islands_asynchronous(c, ring, m, onemax, Budget{4000, INFINITY, -INFINITY}, IslandClock{});
    s.require("asynchronous islands with equal durations and no latency: the synchronous run, exactly", populations_of(c) == populations_of(a) && oc.best->ticket == oa.best->ticket && oc.evaluations == 4000);
    s.check("its simulated time: 100 cycles of 10 evaluations", oc.time - 1000, 1e-9);

    // Random durations and latencies: reproducible; a different run.
    IslandClock clock;
    clock.seed = 5;
    clock.duration = [](const Evaluation&, std::size_t island, calx::random::Stream& r) { return (1 + island) * r.exponential(); };
    clock.latency = [](std::size_t, std::size_t, calx::random::Stream& r) { return 20 * r.uniform(); };
    auto d = archipelago(4, 60, 11), e = archipelago(4, 60, 11);
    const auto od = run_islands_asynchronous(d, ring, m, onemax, Budget{4000, INFINITY, -INFINITY}, clock);
    const auto oe = run_islands_asynchronous(e, ring, m, onemax, Budget{4000, INFINITY, -INFINITY}, clock);
    s.require("asynchronous islands with random durations and latencies: reproducible", populations_of(d) == populations_of(e) && od.time == oe.time && od.best->ticket == oe.best->ticket);
    s.require("asynchronous islands: the slowest island sets the time", od.time > oc.time && od.evaluations == 4000);
  }

  // Islands of (1+1) EAs on a complete graph solve OneMax.
  {
    std::vector<heuresis::genetic::OnePlusLambda<Bits>> v;
    for (std::uint32_t i = 0; i < 6; i++) v.push_back(heuresis::genetic::one_plus_one_ea(50, Draws(13, i)));
    const auto o = run_islands(v, Topology::complete(6), Migration{20, 1}, onemax, Budget{60000, INFINITY, 0});
    s.require("islands of (1+1) EAs solve OneMax (n = 50)", o.ending == Ending::target);
  }

  return s.done();
}
