// Asynchronous evaluation on simulated workers: the master–worker model of distributed search, as a
// discrete-event simulation.
//
// W workers each evaluate one candidate at a time; an evaluation takes a duration that depends on the
// evaluation, the worker and a stream keyed by the ticket (heterogeneous machines, simulations whose length
// varies, a straggler). Whenever a worker is idle the master asks the algorithm for one candidate; whenever an
// evaluation finishes (the earliest first, ties by the order of issue) it is told. The objective itself is
// called when the candidate is issued, since it is pure: only its result's arrival is delayed, in simulated
// time.
//
// What it shows, exactly: a generational algorithm idles workers until its generation is complete, and its
// states are those of a serial run whatever the durations (protocol.hpp); an asynchronous algorithm keeps
// every worker busy, and its run depends on the order of arrival, which the durations decide and the record
// keeps. The outcome's `time` is the simulated wall time (the makespan), so that the cost of synchronisation can
// be measured before a search is spread over real machines (Alba and Tomassini 2002, "Parallelism and
// evolutionary algorithms", IEEE Trans. Evol. Comput. 6, 443–462; Scott and De Jong 2015, "Understanding
// simulated asynchronous evolutionary algorithms", FOGA).
//
// Budgets: evaluations are issued while fewer than the budget have been issued and the completed cost is below
// the budget's cost; evaluations in flight when the target is reached are abandoned, never told.

#pragma once

#include <heuresis/search/draws.hpp>
#include <heuresis/search/run.hpp>

#include <functional>
#include <queue>
#include <set>
#include <vector>

namespace heuresis::search {

struct Workers {
  std::size_t count = 1;
  // The duration of an evaluation on a worker; by default its cost.
  std::function<double(const Evaluation&, std::size_t worker, calx::random::Stream&)> duration = [](const Evaluation& e, std::size_t, calx::random::Stream&) { return e.cost; };
  std::uint64_t seed = 0;
};

template <Optimiser A, class F>
Outcome<typename A::Genotype> run_asynchronous(A& algorithm, F&& f, const Budget& budget, const Workers& workers, Record<typename A::Genotype>* record = nullptr) {
  using G = typename A::Genotype;
  if (workers.count == 0) throw std::invalid_argument("heuresis::search::run_asynchronous: no workers");
  struct Job {
    double finish;
    std::uint64_t order;
    std::size_t worker;
    Candidate<G> candidate;
    Evaluation evaluation;
  };
  struct Later {
    bool operator()(const Job& a, const Job& b) const { return a.finish != b.finish ? a.finish > b.finish : a.order > b.order; }
  };
  const Draws draws(workers.seed);
  constexpr std::uint32_t purpose = tag("search.workers.duration");
  std::priority_queue<Job, std::vector<Job>, Later> busy;
  std::set<std::size_t> idle;
  for (std::size_t w = 0; w < workers.count; w++) idle.insert(w);
  Outcome<G> out;
  Elite<G> elite;
  double t = 0, issued = 0;
  std::uint64_t order = 0;
  bool reached = false;
  for (;;) {
    while (!idle.empty() && !reached && issued < budget.evaluations && out.cost < budget.cost && !algorithm.stopped()) {
      std::vector<Candidate<G>> c = algorithm.ask(1);
      if (c.empty()) break;
      const std::size_t w = *idle.begin();
      idle.erase(idle.begin());
      Evaluation e = evaluate(f, c[0]);
      calx::random::Stream s = draws(purpose, 0, c[0].ticket);
      const double d = workers.duration(e, w, s);
      busy.push({t + d, order++, w, std::move(c[0]), std::move(e)});
      issued += 1;
    }
    if (reached || busy.empty()) break;
    Job job = busy.top();
    busy.pop();
    t = job.finish;
    idle.insert(job.worker);
    out.evaluations += 1;
    out.cost += job.evaluation.cost;
    const Told told{job.candidate.ticket, job.evaluation};
    Individual<G> x{job.candidate.ticket, std::move(job.candidate.genotype), std::move(job.evaluation), std::move(job.candidate.origin)};
    elite.offer(x, out.evaluations, out.cost);
    if (record) record->told.push_back(std::move(x));
    algorithm.tell(std::span<const Told>(&told, 1));
    reached = elite.value() <= budget.target;
  }
  out.time = t;
  out.best = elite.best();
  out.progress = elite.progress();
  if (reached) out.ending = Ending::target;
  else if (algorithm.stopped()) out.ending = Ending::stopped;
  else if (issued >= budget.evaluations || out.cost >= budget.cost) out.ending = Ending::budget;
  else out.ending = Ending::stalled;
  return out;
}

}  // namespace heuresis::search
