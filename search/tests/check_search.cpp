// Checks of heuresis search: the tags and the mixing function against their published vectors; Deb's order;
// the protocol's refusals; a generational search whose states do not depend on the order of telling, the
// executor or the workers' durations; budgets kept exactly; the genealogy acyclic.

#include <heuresis/search/search.hpp>

#include <calx/verification/check.hpp>

#include <algorithm>
#include <cmath>
#include <set>

using namespace heuresis::search;

namespace {

// A generational search on the integers for the checks: λ steps about the centre, the centre moved to the best.
class Walk {
 public:
  using Genotype = long;
  Walk(std::size_t lambda, Draws draws) : lambda_(lambda), draws_(draws) {}

  std::vector<Candidate<long>> ask(std::size_t limit) {
    if (!batch_.active()) {
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
  bool stopped() const { return false; }
  long centre() const { return centre_; }
  std::uint64_t generation() const { return generation_; }

 private:
  std::size_t lambda_;
  Draws draws_;
  Batch<long> batch_;
  long centre_ = 500;
  std::optional<Ticket> parent_;
  std::uint64_t generation_ = 0, serial_ = 0;
};

double distance_to_17(long x) { return std::fabs(static_cast<double>(x - 17)); }

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

  // Random search: the k-th sample does not depend on how the run is cut.
  {
    auto sample = [](calx::random::Stream& r) { return static_cast<long>(r.below(1 << 20)); };
    RandomSearch<long> a(sample, Draws(9)), b(sample, Draws(9));
    auto x = a.ask(10);
    std::vector<Candidate<long>> y;
    for (int i = 0; i < 10; i++) { auto c = b.ask(1); y.push_back(c[0]); }
    bool same = true;
    for (int i = 0; i < 10; i++) same = same && x[i].genotype == y[i].genotype && x[i].ticket == y[i].ticket;
    s.require("random search: batches of any size give the same samples", same);
  }

  return s.done();
}
