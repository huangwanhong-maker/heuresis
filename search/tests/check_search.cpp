// Checks of heuresis search: the tags and the mixing function against their published vectors; Deb's order;
// the protocol's refusals; a generational search whose states do not depend on the order of telling, the
// executor or the workers' durations; budgets kept exactly, in evaluations and in simulated time; the genealogy
// acyclic; exceptions of the objective told as failures, an algorithm's exception ending its run with its
// outcome; random search's breadth; independent restarts with tickets unique across runs.

#include <heuresis/search/search.hpp>

#include <calx/verification/check.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <set>
#include <stdexcept>

using namespace heuresis::search;

namespace {

// A generational search on the integers for the checks: λ steps about the centre, the centre moved to the best;
// stopped after `generations` generations, if given.
class Walk {
 public:
  using Genotype = long;
  Walk(std::size_t lambda, Draws draws, std::uint64_t generations = UINT64_MAX) : lambda_(lambda), draws_(draws), generations_(generations) {}

  std::vector<Candidate<long>> ask(std::size_t limit) {
    if (!batch_.active() && !stopped()) {
      std::vector<Candidate<long>> g;
      for (std::size_t i = 0; i < lambda_; i++) {
        calx::random::Stream s = draws_(tag("check.walk"), generation_, i);
        const long step = static_cast<long>(s.below(21)) - 10;
        g.push_back({make_ticket(draws_.island(), serial_++), centre_ + step, Origin{tag("check.walk"), parent_ ? std::vector<Ticket>{*parent_} : std::vector<Ticket>{}}});
      }
      batch_.open(std::move(g));
    }
    return batch_.take(limit);
  }
  void tell(std::span<const Told> told) {
    for (const auto& t : told)
      if (batch_.accept(t)) {
        const auto order = ranking(batch_.evaluations());
        centre_ = batch_.candidate(order[0]).genotype;
        parent_ = batch_.candidate(order[0]).ticket;
        batch_.close();
        generation_++;
      }
  }
  bool stopped() const { return !batch_.active() && generation_ >= generations_; }
  long centre() const { return centre_; }
  std::uint64_t generation() const { return generation_; }

 private:
  std::size_t lambda_;
  Draws draws_;
  std::uint64_t generations_;
  Batch<long> batch_;
  long centre_ = 500;
  std::optional<Ticket> parent_;
  std::uint64_t generation_ = 0, serial_ = 0;
};

double distance_to_17(long x) { return std::fabs(static_cast<double>(x - 17)); }

// Random search on [0, 1000) that keeps what it is told, by genotype, and throws at its `fault`-th tell (0: never).
class Tally {
 public:
  using Genotype = long;
  explicit Tally(Draws draws, std::size_t fault = 0) : inner_([](calx::random::Stream& r) { return static_cast<long>(r.below(1000)); }, draws), fault_(fault) {}

  std::vector<Candidate<long>> ask(std::size_t limit) {
    auto c = inner_.ask(limit);
    for (const auto& x : c) asked_[x.ticket] = x.genotype;
    return c;
  }
  void tell(std::span<const Told> told) {
    if (++tells_ == fault_) throw std::runtime_error("the fifth tell");
    inner_.tell(told);
    for (const auto& t : told) told_.push_back({asked_.at(t.ticket), t.evaluation});
  }
  bool stopped() const { return false; }
  const std::vector<std::pair<long, Evaluation>>& told() const { return told_; }

 private:
  RandomSearch<long> inner_;
  std::size_t fault_, tells_ = 0;
  std::map<Ticket, long> asked_;
  std::vector<std::pair<long, Evaluation>> told_;
};

// An objective that cannot evaluate odd numbers.
double even_only(long x) {
  if (x % 2) throw std::domain_error("odd");
  return distance_to_17(x);
}

// The serial executor, recording the widest batch it was given.
struct Widest {
  std::size_t* widest;
  template <class G, class F>
  std::vector<Evaluation> operator()(std::span<const Candidate<G>> candidates, F& f) const {
    *widest = std::max(*widest, candidates.size());
    return Serial{}(candidates, f);
  }
};

}  // namespace

int main() {
  calx::verification::Suite s;

  // FNV-1a (32 bits) test vectors, and SplitMix64's first output from the seed 0.
  s.require("FNV-1a of the empty word", tag("") == 0x811c9dc5u);
  s.require("FNV-1a of \"a\"", tag("a") == 0xe40c292cu);
  s.require("FNV-1a of \"foobar\"", tag("foobar") == 0xbf9cf968u);
  s.require("SplitMix64 from 0: 0xe220a8397b1dcdaf", mix(0) == 0xe220a8397b1dcdafull);

  // Keyed draws: the same identity gives the same stream; islands differ.
  {
    Draws a(7), b(7), c(7, 1);
    auto x = a(tag("p"), 3, 4), y = b(tag("p"), 3, 4), z = c(tag("p"), 3, 4);
    const double u = x.uniform(), v = y.uniform(), w = z.uniform();
    s.require("a stream is a function of its identity", u == v);
    s.require("another island draws another stream", u != w);
  }

  // Deb's order.
  {
    Evaluation feas = scalar(5), feas_better = scalar(1), infeas = evaluation({0}, {0.5}), infeas_worse = evaluation({-9}, {2}), failed = failure("diverged"), nan = scalar(NAN);
    s.require("Deb: the smaller objective among feasible points", better(feas_better, feas) && !better(feas, feas_better));
    s.require("Deb: feasible before infeasible, whatever the objective", better(feas, infeas));
    s.require("Deb: less violation first among infeasible points", better(infeas, infeas_worse));
    s.require("Deb: a failure after everything", better(infeas_worse, failed) && !better(failed, infeas_worse));
    s.require("an objective that is not a number is a failure", nan.failed() && !better(nan, failed) && !better(failed, nan));
    s.require("a failure keeps its reason", failed.reason == "diverged");
  }

  // The protocol's refusals.
  {
    Walk w(4, Draws(1));
    auto c = w.ask(unlimited);
    bool unknown = false, twice = false;
    try { const Told t{9999, scalar(0)}; w.tell(std::span<const Told>(&t, 1)); } catch (const std::invalid_argument&) { unknown = true; }
    const Told t0{c[0].ticket, scalar(0)};
    w.tell(std::span<const Told>(&t0, 1));
    try { w.tell(std::span<const Told>(&t0, 1)); } catch (const std::invalid_argument&) { twice = true; }
    s.require("an unknown ticket is refused", unknown);
    s.require("a ticket told twice is refused", twice);
    s.require("a generation offers nothing more until it is told", w.ask(unlimited).empty());
  }

  // Order of telling, executors and workers do not change a generational search.
  {
    const std::size_t lambda = 6;
    Walk serial(lambda, Draws(42));
    Record<long> rec_serial;
    const auto o1 = run(serial, distance_to_17, Budget{600, INFINITY, -INFINITY}, Serial{}, unlimited, &rec_serial);

    Walk shuffled(lambda, Draws(42));
    for (int g = 0; g < 100; g++) {
      auto c = shuffled.ask(unlimited);
      std::vector<Told> t;
      for (auto& x : c) t.push_back({x.ticket, scalar(distance_to_17(x.genotype))});
      std::reverse(t.begin(), t.end());
      std::rotate(t.begin(), t.begin() + g % lambda, t.end());
      shuffled.tell(t);
    }

    calx::scheduler::ThreadPool pool(4);
    Walk pooled(lambda, Draws(42));
    const auto o2 = run(pooled, distance_to_17, Budget{600, INFINITY, -INFINITY}, Pool(pool));

    Walk parts(lambda, Draws(42));
    const auto o3 = run(parts, distance_to_17, Budget{600, INFINITY, -INFINITY}, Serial{}, 4);  // generations cut into parts

    Walk async(lambda, Draws(42));
    Workers wk;
    wk.count = 3;
    wk.duration = [](const Evaluation&, std::size_t worker, calx::random::Stream& r) { return 1 + worker + 5 * r.uniform(); };
    const auto o4 = run_asynchronous(async, distance_to_17, Budget{600, INFINITY, -INFINITY}, wk);

    s.require("the serial run reaches 17", serial.centre() == 17 && o1.best && o1.best->genotype == 17);
    s.require("telling in any order gives the same states", shuffled.centre() == serial.centre() && shuffled.generation() == serial.generation());
    s.require("a pool of threads gives the same run", pooled.centre() == serial.centre() && o2.best->ticket == o1.best->ticket && o2.progress.size() == o1.progress.size());
    s.require("generations cut into parts give the same run", parts.centre() == serial.centre() && o3.best->ticket == o1.best->ticket);
    s.require("simulated workers with random durations give the same run", async.centre() == serial.centre() && o4.best->ticket == o1.best->ticket);
    s.require("the budget of evaluations is kept exactly", o1.evaluations == 600 && o3.evaluations == 600 && o4.evaluations == 600 && rec_serial.told.size() == 600);

    // The genealogy is acyclic: every parent was told before its child.
    std::set<Ticket> seen;
    bool acyclic = true;
    for (const auto& x : rec_serial.told) {
      for (Ticket p : x.origin.parents) acyclic = acyclic && seen.count(p);
      seen.insert(x.ticket);
    }
    s.require("every parent precedes its child in the record", acyclic);

    // A target ends the run at the first evaluation that reaches it.
    Walk tgt(lambda, Draws(42));
    const auto o5 = run(tgt, distance_to_17, Budget{INFINITY, INFINITY, 0});
    s.require("a target stops the run", o5.ending == Ending::target && o5.best->evaluation.value() == 0 && o5.evaluations < 600);
    s.require("a run without simulated time gives its progress time 0", std::all_of(o1.progress.begin(), o1.progress.end(), [](const Progress& p) { return p.time == 0; }));
  }

  // Exceptions: the objective's are failures with their message; the algorithm's end the run, keeping its outcome.
  {
    Tally serial(Draws(21)), pooled(Draws(21));
    const auto o = run(serial, even_only, Budget{200, INFINITY, -INFINITY});
    calx::scheduler::ThreadPool pool(4);
    const auto op = run(pooled, even_only, Budget{200, INFINITY, -INFINITY}, Pool(pool));
    std::size_t odd = 0;
    bool told_right = serial.told().size() == 200;
    for (const auto& [x, e] : serial.told()) {
      odd += x % 2 != 0;
      told_right = told_right && (x % 2 ? e.failed() && e.reason == "odd" : !e.failed() && e.value() == distance_to_17(x));
    }
    s.require("an objective's exception is told as a failure with its message (" + std::to_string(odd) + " of 200)", told_right && odd > 50);
    s.require("… the run goes on to its budget, its best a point that evaluated", o.ending == Ending::budget && o.evaluations == 200 && o.best->genotype % 2 == 0);
    s.require("… and so on a pool of threads", op.ending == Ending::budget && op.best->ticket == o.best->ticket && pooled.told().size() == 200);

    Tally faulty(Draws(22), 5);
    Record<long> rec;
    const auto oe = run(faulty, distance_to_17, Budget{200, INFINITY, -INFINITY}, Serial{}, unlimited, &rec);
    double least = INFINITY;
    for (const auto& x : rec.told) least = std::min(least, x.evaluation.value());
    s.require("an algorithm that throws at its 5th tell ends the run with Ending::error and its message", oe.ending == Ending::error && oe.error == "the fifth tell");
    s.require("… keeping what was spent and the best found", oe.evaluations == 5 && rec.told.size() == 5 && oe.best && oe.best->evaluation.value() == least && !oe.progress.empty());
    Tally faulty_async(Draws(22), 5);
    Workers wk;
    wk.count = 3;
    const auto oa = run_asynchronous(faulty_async, distance_to_17, Budget{200, INFINITY, -INFINITY}, wk);
    s.require("… and so on simulated workers", oa.ending == Ending::error && oa.error == "the fifth tell" && oa.evaluations == 5 && oa.best);
  }

  // Workers: a generational search idles them, an asynchronous one does not.
  {
    Workers wk;
    wk.count = 4;
    wk.duration = [](const Evaluation&, std::size_t worker, calx::random::Stream&) { return worker == 0 ? 4.0 : 1.0; };  // one slow machine
    Walk gen(4, Draws(3));
    const auto og = run_asynchronous(gen, distance_to_17, Budget{400, INFINITY, -INFINITY}, wk);
    RandomSearch<long> rs([](calx::random::Stream& r) { return static_cast<long>(r.below(1000)); }, Draws(3));
    const auto oa = run_asynchronous(rs, distance_to_17, Budget{400, INFINITY, -INFINITY}, wk);
    // Generational: each generation waits for the slow machine, 100 generations of 4 units.
    s.check("a generational search waits for its slowest worker (makespan 400)", og.time - 400, 1e-9);
    // Asynchronous: the slow machine does a quarter of the work of each fast one: 400 = 3·t + t/4.
    s.check("an asynchronous search keeps every worker busy (makespan 400/3.25)", oa.time - 400 / 3.25, 4.0);
  }

  // A budget of simulated time: nothing starts at or after it, and what would finish after it is abandoned.
  {
    Workers wk;
    wk.count = 2;
    wk.duration = [](const Evaluation&, std::size_t, calx::random::Stream&) { return 1.0; };
    std::size_t calls = 0;
    auto counted = [&calls](long x) { calls++; return distance_to_17(x); };
    RandomSearch<long> a([](calx::random::Stream& r) { return static_cast<long>(r.below(1000)); }, Draws(5));
    const auto o = run_asynchronous(a, counted, Budget{INFINITY, INFINITY, -INFINITY, 10}, wk);
    s.require("a time budget of 10 on two workers of unit durations: 20 evaluations, none started at 10", o.ending == Ending::budget && o.evaluations == 20 && calls == 20 && o.time == 10);
    calls = 0;
    RandomSearch<long> b([](calx::random::Stream& r) { return static_cast<long>(r.below(1000)); }, Draws(5));
    const auto ob = run_asynchronous(b, counted, Budget{INFINITY, INFINITY, -INFINITY, 10.5}, wk);
    s.require("a time budget of 10.5: the two started at 10 abandoned, the run's time the budget's", ob.ending == Ending::budget && ob.evaluations == 20 && calls == 22 && ob.time == 10.5);
    bool on_time = !ob.progress.empty();
    for (const auto& p : ob.progress) on_time = on_time && p.time == std::ceil(p.evaluations / 2);
    s.require("each improvement carries the simulated time of its arrival", on_time);
  }

  // Random search: at most `breadth` candidates an ask, and the k-th sample does not depend on how the run is cut.
  {
    std::size_t widest = 0;
    RandomSearch<long> narrow([](calx::random::Stream& r) { return static_cast<long>(r.below(1000)); }, Draws(6));
    run(narrow, distance_to_17, Budget{100, INFINITY, -INFINITY}, Widest{&widest});
    s.require("random search under a finite budget: one candidate an ask by default", widest == 1 && narrow.breadth() == 1);
    widest = 0;
    RandomSearch<long> wide([](calx::random::Stream& r) { return static_cast<long>(r.below(1000)); }, Draws(6), 8);
    run(wide, distance_to_17, Budget{100, INFINITY, -INFINITY}, Widest{&widest});
    s.require("… and at most its breadth, 8", widest == 8);
  }
  {
    auto sample = [](calx::random::Stream& r) { return static_cast<long>(r.below(1 << 20)); };
    RandomSearch<long> a(sample, Draws(9), 10), b(sample, Draws(9));
    auto x = a.ask(10);
    std::vector<Candidate<long>> y;
    for (int i = 0; i < 10; i++) { auto c = b.ask(1); y.push_back(c[0]); }
    bool same = true;
    for (int i = 0; i < 10; i++) same = same && x[i].genotype == y[i].genotype && x[i].ticket == y[i].ticket;
    s.require("random search: batches of any size give the same samples", same);
  }

  // Restarts: a new run whenever the last has stopped or spent its own budget, tickets unique across runs.
  {
    Restarts<Walk> r([](std::size_t, Draws d) { return Walk(4, d, 5); }, RestartLimits{}, Draws(31));
    Record<long> rec;
    const auto o = run(r, distance_to_17, Budget{200, INFINITY, -INFINITY}, Serial{}, unlimited, &rec);
    std::set<Ticket> tickets, seen;
    std::size_t sources = 0;
    bool acyclic = true;
    for (const auto& x : rec.told) {
      tickets.insert(x.ticket);
      sources += x.origin.parents.empty();
      for (Ticket p : x.origin.parents) acyclic = acyclic && seen.count(p);
      seen.insert(x.ticket);
    }
    s.require("restarts: runs of five generations of four, ten runs in 200 evaluations", r.runs() == 10 && o.evaluations == 200);
    s.require("… every ticket unique across runs", tickets.size() == 200);
    s.require("… one genealogy, acyclic, whose sources are the ten runs' first generations", acyclic && sources == 40);
    Restarts<Walk> cut([](std::size_t, Draws d) { return Walk(4, d); }, RestartLimits{unlimited, 12}, Draws(31));
    run(cut, distance_to_17, Budget{120, INFINITY, -INFINITY});
    s.require("restarts: a per-run budget of 12 ends each run, ten in 120", cut.runs() == 10);
    Restarts<Walk> few([](std::size_t, Draws d) { return Walk(4, d, 5); }, RestartLimits{3}, Draws(31));
    const auto of = run(few, distance_to_17, Budget{});
    s.require("restarts: three runs at most, then stopped", of.ending == Ending::stopped && of.evaluations == 60 && few.runs() == 3);
    Restarts<RandomSearch<long>> async([](std::size_t, Draws d) { return RandomSearch<long>([](calx::random::Stream& t) { return static_cast<long>(t.below(1000)); }, d); }, RestartLimits{unlimited, 25}, Draws(32));
    Workers wk;
    wk.count = 3;
    wk.duration = [](const Evaluation&, std::size_t worker, calx::random::Stream& t) { return 1 + static_cast<double>(worker) + t.uniform(); };
    Record<long> ra;
    const auto oa = run_asynchronous(async, distance_to_17, Budget{100, INFINITY, -INFINITY}, wk, &ra);
    std::set<Ticket> ta;
    for (const auto& x : ra.told) ta.insert(x.ticket);
    s.require("restarts of an asynchronous search on workers: four runs of 25, tickets unique", async.runs() == 4 && oa.evaluations == 100 && ta.size() == 100);
  }

  return s.done();
}
