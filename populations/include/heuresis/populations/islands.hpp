// Island models: several searches joined by migration.
//
// k islands, each any Migratory algorithm (search/protocol.hpp), run side by side; every `interval` cycles (a
// cycle is one ask, its evaluation and its tell, a generation for a generational algorithm) each island sends
// its `count` best along every out-edge of a topology, and each receiver injects what it received, which
// replaces its worst when not worse (the "coarse-grained" or "distributed" evolutionary algorithm; Tanese 1989,
// "Distributed genetic algorithms", ICGA 3, 434–439; Whitley, Rana and Heckendorn 1999, "The island model genetic
// algorithm: on separability, population size and convergence", J. Comput. Inf. Technol. 7, 33–47; Skolicki and
// De Jong 2005, "The influence of migration sizes and intervals on island models", GECCO, 1295–1302). Islands
// keep diversity apart and share discoveries; the topology and interval set how fast a good individual spreads.
//
// Migrants are plain values (genotype, evaluation, provenance): the island they came from is in their tickets'
// top bits (make each island's algorithm with Draws(seed, island)), and whoever carries them between machines
// can serialise them. Two drivers:
// - run_islands: synchronous. All islands run their cycles of an epoch, then migrate together, deliveries in
//   order of the sending island. The islands of an epoch are independent, so they may run in parallel on a calx
//   thread pool (the objective is then called concurrently and must allow it); the run is the same either way.
// - run_islands_asynchronous: a discrete-event simulation of islands on machines of their own. A cycle takes
//   the sum of its evaluations' durations; migrants arrive after a latency and are injected when the receiver
//   next begins a cycle. Events at one instant are taken completions first, then arrivals, then the beginnings
//   of cycles, so that with equal durations and no latency the asynchronous run is the synchronous one, exactly.
//   The outcome's time is the simulated wall time.
// Budgets: each island may spend ⌊budget/k⌋ evaluations, so the total never exceeds the budget; the cost budget
// and the target are checked at the end of each epoch (synchronous) or of each cycle (asynchronous).

#pragma once

#include <heuresis/populations/topology.hpp>
#include <heuresis/search/search.hpp>

#include <calx/scheduler/parallel_for.hpp>
#include <calx/scheduler/thread_pool.hpp>

#include <cmath>
#include <functional>
#include <optional>
#include <queue>
#include <vector>

namespace heuresis::populations {

using heuresis::search::Budget;
using heuresis::search::Candidate;
using heuresis::search::Ending;
using heuresis::search::Evaluation;
using heuresis::search::Individual;
using heuresis::search::Told;

struct Migration {
  std::size_t interval = 1;  // cycles between migrations
  std::size_t count = 1;     // emigrants sent along each edge
};

template <class G>
struct IslandOutcome {
  std::optional<Individual<G>> best;
  double evaluations = 0, cost = 0;
  double time = 0;  // simulated wall time, asynchronous runs
  std::vector<double> island_evaluations;
  std::size_t migrations = 0;  // migration events (synchronous: epochs that migrated; asynchronous: sends)
  std::vector<heuresis::search::Progress> progress;
  Ending ending = Ending::budget;
};

namespace detail {
inline std::size_t cap_of(const Budget& b, std::size_t k) {
  return std::isfinite(b.evaluations) ? static_cast<std::size_t>(std::floor(b.evaluations / static_cast<double>(k))) : heuresis::search::unlimited;
}

// One cycle of an island: ask (within its cap), evaluate on this thread, tell. Returns the individuals told.
template <class A, class F>
std::vector<Individual<typename A::Genotype>> cycle(A& a, F& f, std::size_t& spent, std::size_t cap, bool& idle) {
  using G = typename A::Genotype;
  if (a.stopped() || spent >= cap) { idle = true; return {}; }
  std::vector<Candidate<G>> c = a.ask(cap == heuresis::search::unlimited ? heuresis::search::unlimited : cap - spent);
  if (c.empty()) { idle = true; return {}; }
  std::vector<Told> told(c.size());
  std::vector<Individual<G>> out(c.size());
  for (std::size_t i = 0; i < c.size(); i++) {
    Evaluation e = heuresis::search::evaluate(f, c[i]);
    told[i] = {c[i].ticket, e};
    out[i] = {c[i].ticket, std::move(c[i].genotype), std::move(e), std::move(c[i].origin)};
  }
  a.tell(std::span<const Told>(told));
  spent += c.size();
  return out;
}
}  // namespace detail

template <heuresis::search::Migratory A, class F>
IslandOutcome<typename A::Genotype> run_islands(std::vector<A>& islands, const Topology& topology, const Migration& migration, F&& f, const Budget& budget = {}, calx::scheduler::ThreadPool* pool = nullptr) {
  using G = typename A::Genotype;
  const std::size_t k = islands.size();
  if (k == 0 || topology.size() != k) throw std::invalid_argument("heuresis::populations::run_islands: one topology vertex per island");
  if (migration.interval == 0) throw std::invalid_argument("heuresis::populations::run_islands: an interval of zero cycles");
  const std::size_t cap = detail::cap_of(budget, k);
  std::vector<std::size_t> spent(k, 0);
  std::vector<char> idle(k, 0);
  std::vector<std::vector<Individual<G>>> told(k);
  IslandOutcome<G> out;
  heuresis::search::Elite<G> elite;
  for (;;) {
    auto work = [&](std::size_t i) {
      bool stop = false;
      for (std::size_t c = 0; c < migration.interval && !stop; c++) {
        auto xs = detail::cycle(islands[i], f, spent[i], cap, stop);
        for (auto& x : xs) told[i].push_back(std::move(x));
      }
      idle[i] = stop;
    };
    if (pool && pool->size() > 1) calx::scheduler::parallel_for(*pool, k, 1, [&](std::size_t b, std::size_t e, std::size_t) { for (std::size_t i = b; i < e; i++) work(i); });
    else for (std::size_t i = 0; i < k; i++) work(i);
    bool any = false;
    for (std::size_t i = 0; i < k; i++) {
      any = any || !told[i].empty();
      for (auto& x : told[i]) {
        out.evaluations += 1;
        out.cost += x.evaluation.cost;
        elite.offer(x, out.evaluations, out.cost);
      }
      told[i].clear();
    }
    if (elite.value() <= budget.target) { out.ending = Ending::target; break; }
    if (out.cost >= budget.cost) { out.ending = Ending::budget; break; }
    if (std::all_of(idle.begin(), idle.end(), [](char c) { return c != 0; })) {
      bool capped = false, stopped = true;
      for (std::size_t i = 0; i < k; i++) { capped = capped || spent[i] >= cap; stopped = stopped && islands[i].stopped(); }
      out.ending = capped ? Ending::budget : stopped ? Ending::stopped : Ending::stalled;
      break;
    }
    if (!any) { out.ending = Ending::stalled; break; }
    std::vector<std::vector<Individual<G>>> sent(k), inbox(k);
    for (std::size_t i = 0; i < k; i++) sent[i] = islands[i].emigrants(migration.count);
    for (std::size_t i = 0; i < k; i++)
      for (auto j : topology.out[i]) inbox[j].insert(inbox[j].end(), sent[i].begin(), sent[i].end());
    for (std::size_t j = 0; j < k; j++)
      if (!inbox[j].empty()) islands[j].inject(std::span<const Individual<G>>(inbox[j]));
    out.migrations++;
  }
  out.island_evaluations.assign(spent.begin(), spent.end());
  out.best = elite.best();
  out.progress = elite.progress();
  return out;
}

struct IslandClock {
  // The duration of one evaluation on an island; a cycle takes the sum over its candidates. By default the cost.
  std::function<double(const Evaluation&, std::size_t island, calx::random::Stream&)> duration = [](const Evaluation& e, std::size_t, calx::random::Stream&) { return e.cost; };
  // The latency of a migration from one island to another; by default none.
  std::function<double(std::size_t from, std::size_t to, calx::random::Stream&)> latency = [](std::size_t, std::size_t, calx::random::Stream&) { return 0.0; };
  std::uint64_t seed = 0;
};

template <heuresis::search::Migratory A, class F>
IslandOutcome<typename A::Genotype> run_islands_asynchronous(std::vector<A>& islands, const Topology& topology, const Migration& migration, F&& f, const Budget& budget, const IslandClock& clock) {
  using G = typename A::Genotype;
  const std::size_t k = islands.size();
  if (k == 0 || topology.size() != k) throw std::invalid_argument("heuresis::populations::run_islands_asynchronous: one topology vertex per island");
  if (migration.interval == 0) throw std::invalid_argument("heuresis::populations::run_islands_asynchronous: an interval of zero cycles");
  enum Kind : int { completion = 0, arrival = 1, start = 2 };
  struct Event {
    double time;
    int kind;
    std::uint64_t seq;
    std::size_t island;
    std::vector<Candidate<G>> candidates;
    std::vector<Evaluation> evaluations;
    std::vector<Individual<G>> migrants;
  };
  struct Later {
    bool operator()(const Event& a, const Event& b) const {
      if (a.time != b.time) return a.time > b.time;
      if (a.kind != b.kind) return a.kind > b.kind;
      return a.seq > b.seq;
    }
  };
  constexpr std::uint32_t duration_tag = heuresis::search::tag("populations.islands.duration");
  constexpr std::uint32_t latency_tag = heuresis::search::tag("populations.islands.latency");
  const std::size_t cap = detail::cap_of(budget, k);
  std::priority_queue<Event, std::vector<Event>, Later> queue;
  std::uint64_t seq = 0;
  std::vector<std::size_t> spent(k, 0), cycles(k, 0), sends(k, 0);
  std::vector<std::vector<Individual<G>>> inbox(k);
  IslandOutcome<G> out;
  heuresis::search::Elite<G> elite;
  for (std::size_t i = 0; i < k; i++) queue.push({0, start, seq++, i, {}, {}, {}});
  bool reached = false;
  double t = 0;
  while (!queue.empty() && !reached) {
    Event ev = queue.top();
    queue.pop();
    t = ev.time;
    const std::size_t i = ev.island;
    if (ev.kind == arrival) {
      inbox[i].insert(inbox[i].end(), std::make_move_iterator(ev.migrants.begin()), std::make_move_iterator(ev.migrants.end()));
    } else if (ev.kind == start) {
      if (!inbox[i].empty()) {
        islands[i].inject(std::span<const Individual<G>>(inbox[i]));
        inbox[i].clear();
      }
      if (islands[i].stopped() || spent[i] >= cap || out.cost >= budget.cost) continue;
      std::vector<Candidate<G>> c = islands[i].ask(cap == heuresis::search::unlimited ? heuresis::search::unlimited : cap - spent[i]);
      if (c.empty()) continue;
      std::vector<Evaluation> e(c.size());
      double d = 0;
      const heuresis::search::Draws draws(clock.seed, static_cast<std::uint32_t>(i));
      for (std::size_t j = 0; j < c.size(); j++) {
        e[j] = heuresis::search::evaluate(f, c[j]);
        calx::random::Stream s = draws(duration_tag, cycles[i], j);
        d += clock.duration(e[j], i, s);
      }
      spent[i] += c.size();
      queue.push({t + d, completion, seq++, i, std::move(c), std::move(e), {}});
    } else {
      std::vector<Told> told(ev.candidates.size());
      for (std::size_t j = 0; j < told.size(); j++) {
        told[j] = {ev.candidates[j].ticket, ev.evaluations[j]};
        Individual<G> x{ev.candidates[j].ticket, std::move(ev.candidates[j].genotype), ev.evaluations[j], std::move(ev.candidates[j].origin)};
        out.evaluations += 1;
        out.cost += x.evaluation.cost;
        elite.offer(x, out.evaluations, out.cost);
      }
      islands[i].tell(std::span<const Told>(told));
      cycles[i]++;
      reached = elite.value() <= budget.target;
      if (cycles[i] % migration.interval == 0) {
        const auto sent = islands[i].emigrants(migration.count);
        const heuresis::search::Draws draws(clock.seed, static_cast<std::uint32_t>(i));
        for (auto j : topology.out[i]) {
          calx::random::Stream s = draws(latency_tag, sends[i], j);
          queue.push({t + clock.latency(i, j, s), arrival, seq++, j, {}, {}, sent});
        }
        sends[i]++;
        out.migrations++;
      }
      queue.push({t, start, seq++, i, {}, {}, {}});
    }
  }
  out.time = t;
  out.island_evaluations.assign(spent.begin(), spent.end());
  out.best = elite.best();
  out.progress = elite.progress();
  bool capped = false, stopped = true;
  for (std::size_t i = 0; i < k; i++) { capped = capped || spent[i] >= cap; stopped = stopped && islands[i].stopped(); }
  out.ending = reached ? Ending::target : (capped || out.cost >= budget.cost) ? Ending::budget : stopped ? Ending::stopped : Ending::stalled;
  return out;
}

}  // namespace heuresis::populations
