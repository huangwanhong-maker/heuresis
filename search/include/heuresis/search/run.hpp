// Driving a search: asking, evaluating where the executor says, telling, until a budget or a target is reached.
//
// The driver is the only place where the objective is called. An objective is any callable taking the genotype
// (and, if it wants, the ticket, so that a noisy objective can draw its noise from a stream keyed by the ticket
// and stay reproducible) and returning an Evaluation or a number. The executor says where the calls run:
//
//   Serial  — on this thread, in order;
//   Pool    — on a calx thread pool, the objective called concurrently (it must be safe to call so);
//   others  — any type with the same call (a cluster, a device, a queue of jobs) at the edges of the
//             infrastructure; the core supplies these two and the simulated workers of workers.hpp.
//
// The budget counts evaluations, their cost, or both, and for the simulated workers of workers.hpp their simulated
// time; a target stops the run when the best feasible value reaches it. The outcome holds the best individual by
// Deb's rule (ties to the earlier ticket), what was spent, each improvement of the best (the anytime curve, from which
// analysis/ reads runtimes) and why the run ended. A Record keeps every individual in the order told, with its origin:
// the run's genealogy and enough to replay it.
//
// Errors. An exception the objective throws (a std::exception) is its evaluation's failure, with what() as the
// reason (decision 0003 §2.3): the algorithm is told it like any other failure, and the run goes on. An exception
// the algorithm's ask or tell throws ends the run with Ending::error and its message; the outcome keeps what was
// spent and found until then, and the record what was told. Exceptions of other types are not caught.

#pragma once

#include <heuresis/search/protocol.hpp>

#include <calx/scheduler/parallel_for.hpp>
#include <calx/scheduler/thread_pool.hpp>

#include <cmath>
#include <exception>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

namespace heuresis::search {

struct Budget {
  double evaluations = std::numeric_limits<double>::infinity();
  double cost = std::numeric_limits<double>::infinity();
  double target = -std::numeric_limits<double>::infinity();  // stop when the best feasible value ≤ target
  double time = std::numeric_limits<double>::infinity();     // simulated time, honoured by run_asynchronous
};

enum class Ending : std::uint8_t { budget, target, stopped, stalled, error };

// An improvement of the best: after `evaluations` evaluations costing `cost`, the best value was `value`; `time` is
// the simulated time at which it arrived (run_asynchronous), 0 in a run without simulated time.
struct Progress {
  double evaluations = 0, cost = 0, value = 0;
  Ticket ticket = 0;
  double time = 0;
};

template <class G>
struct Outcome {
  std::optional<Individual<G>> best;
  double evaluations = 0, cost = 0;
  double time = 0;  // simulated wall time, for asynchronous runs (workers.hpp)
  std::vector<Progress> progress;
  Ending ending = Ending::budget;
  std::string error;  // what the algorithm threw, when the ending is Ending::error
};

template <class G>
struct Record {
  std::vector<Individual<G>> told;  // in the order told
};

// Calls the objective on a candidate: f(g, ticket) if it takes the ticket, else f(g); a number becomes a scalar
// evaluation, and a std::exception thrown becomes a failure whose reason is its what().
template <class G, class F>
Evaluation evaluate(F& f, const Candidate<G>& c) {
  try {
    if constexpr (std::is_invocable_v<F&, const G&, Ticket>) {
      using R = std::invoke_result_t<F&, const G&, Ticket>;
      if constexpr (std::is_convertible_v<R, Evaluation>) return f(c.genotype, c.ticket);
      else return scalar(static_cast<double>(f(c.genotype, c.ticket)));
    } else {
      using R = std::invoke_result_t<F&, const G&>;
      if constexpr (std::is_convertible_v<R, Evaluation>) return f(c.genotype);
      else return scalar(static_cast<double>(f(c.genotype)));
    }
  } catch (const std::exception& e) {
    return failure(e.what());
  }
}

struct Serial {
  template <class G, class F>
  std::vector<Evaluation> operator()(std::span<const Candidate<G>> candidates, F& f) const {
    std::vector<Evaluation> out;
    out.reserve(candidates.size());
    for (const auto& c : candidates) out.push_back(evaluate(f, c));
    return out;
  }
};

class Pool {
 public:
  explicit Pool(calx::scheduler::ThreadPool& pool, std::size_t grain = 1) : pool_(&pool), grain_(grain) {}
  template <class G, class F>
  std::vector<Evaluation> operator()(std::span<const Candidate<G>> candidates, F& f) const {
    std::vector<Evaluation> out(candidates.size());
    calx::scheduler::parallel_for(*pool_, candidates.size(), grain_, [&](std::size_t b, std::size_t e, std::size_t) {
      for (std::size_t i = b; i < e; i++) out[i] = evaluate(f, candidates[i]);
    });
    return out;
  }

 private:
  calx::scheduler::ThreadPool* pool_;
  std::size_t grain_;
};

// The best so far, by Deb's rule, ties to the earlier ticket; its improvements, as progress.
template <class G>
class Elite {
 public:
  // Returns true when the individual is a new best; `time` is the simulated time of its arrival, if any.
  bool offer(const Individual<G>& x, double evaluations, double cost, double time = 0) {
    if (best_) {
      if (better(best_->evaluation, x.evaluation)) return false;
      if (!better(x.evaluation, best_->evaluation) && !(x.ticket < best_->ticket)) return false;  // a tie: the earlier ticket stays
    }
    const double before = value();
    best_ = x;
    if (value() < before) progress_.push_back({evaluations, cost, value(), x.ticket, time});
    return true;
  }
  const std::optional<Individual<G>>& best() const { return best_; }
  const std::vector<Progress>& progress() const { return progress_; }
  // The best feasible value, or +∞.
  double value() const { return best_ && best_->evaluation.feasible() ? best_->evaluation.value() : std::numeric_limits<double>::infinity(); }

 private:
  std::optional<Individual<G>> best_;
  std::vector<Progress> progress_;
};

// Runs the search until the budget is spent, the target is reached, the algorithm stops, offers nothing or throws.
// `batch` caps the candidates asked at once (the executor's width); the budget of evaluations is never exceeded. The
// budget's simulated time has no meaning here and is ignored; progress carries time 0.
template <Optimiser A, class F, class Executor = Serial>
Outcome<typename A::Genotype> run(A& algorithm, F&& f, const Budget& budget = {}, Executor&& executor = Executor{}, std::size_t batch = unlimited, Record<typename A::Genotype>* record = nullptr) {
  using G = typename A::Genotype;
  Outcome<G> out;
  Elite<G> elite;
  for (;;) {
    if (elite.value() <= budget.target) { out.ending = Ending::target; break; }
    if (out.evaluations >= budget.evaluations || out.cost >= budget.cost) { out.ending = Ending::budget; break; }
    if (algorithm.stopped()) { out.ending = Ending::stopped; break; }
    std::size_t limit = batch;
    if (std::isfinite(budget.evaluations)) limit = std::min(limit, static_cast<std::size_t>(std::ceil(budget.evaluations - out.evaluations)));
    std::vector<Candidate<G>> candidates;
    try {
      candidates = algorithm.ask(limit);
    } catch (const std::exception& e) {
      out.ending = Ending::error;
      out.error = e.what();
      break;
    }
    if (candidates.empty()) { out.ending = algorithm.stopped() ? Ending::stopped : Ending::stalled; break; }
    std::vector<Evaluation> evaluations = executor(std::span<const Candidate<G>>(candidates), f);
    std::vector<Told> told(candidates.size());
    for (std::size_t i = 0; i < candidates.size(); i++) {
      out.evaluations += 1;
      out.cost += evaluations[i].cost;
      told[i] = {candidates[i].ticket, evaluations[i]};
      Individual<G> x{candidates[i].ticket, std::move(candidates[i].genotype), std::move(evaluations[i]), std::move(candidates[i].origin)};
      elite.offer(x, out.evaluations, out.cost);
      if (record) record->told.push_back(std::move(x));
    }
    try {
      algorithm.tell(std::span<const Told>(told));
    } catch (const std::exception& e) {
      out.ending = Ending::error;
      out.error = e.what();
      break;
    }
  }
  out.best = elite.best();
  out.progress = elite.progress();
  return out;
}

}  // namespace heuresis::search
