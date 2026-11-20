// Evaluators and the assessor that runs them for a search (decision 0010).
//
// An evaluator declares its name, version, kind, standing, fidelity and cost, and whether it draws at random; it answers an
// assessment of a value. The assessor runs every evaluator on a candidate, keeps what each answered in the cache by (input,
// evaluator, version, fidelity, stream), lowers each metric's standing to its evaluator's where it claimed more, merges the
// answers, and gives the evolutionary loop an Evaluation by the tiers (tiers.hpp), so that only what may select does. Each
// stochastic evaluator draws from a stream keyed by the input, the evaluator and the replicate, so that a candidate assessed again
// on the same replicate gets the same answer (and two candidates the same draws: common random numbers), and a fresh replicate gets
// fresh draws. A winner is re-assessed on fresh replicates before it is reported, its metrics then the means with their errors.

#pragma once

#include <heuresis/assessment/cascade.hpp>
#include <heuresis/assessment/record.hpp>
#include <heuresis/assessment/tiers.hpp>

#include <heuresis/search/draws.hpp>
#include <heuresis/search/evaluation.hpp>
#include <heuresis/spaces/value.hpp>

#include <calx/random/stream.hpp>

#include <algorithm>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace heuresis::assessment {

struct Evaluator {
  std::string name, version = "1";
  Kind kind = Kind::closed_form;
  Standing standing = Standing::computed;
  double fidelity = 1;
  double cost = 1;
  bool stochastic = false;
  std::function<Assessment(const spaces::Value&, calx::random::Stream&)> assess;
};

class Assessor {
 public:
  explicit Assessor(std::vector<Evaluator> evaluators, Tiering tiering = {}, std::uint64_t seed = 0)
      : evaluators_(std::move(evaluators)), tiering_(tiering), seed_(seed) {
    if (evaluators_.empty()) throw Refusal("an assessor needs at least one evaluator");
    for (const auto& e : evaluators_)
      if (!e.assess || e.name.empty()) throw Refusal("an evaluator needs a name and an assessment");
  }
  const std::vector<Evaluator>& evaluators() const { return evaluators_; }
  const Tiering& tiering() const { return tiering_; }
  AssessmentCache& cache() { return cache_; }
  std::size_t runs() const { return runs_; }

  // Every evaluator on x, at a replicate; recorded and returned exactly when asked again.
  Assessment assess(const spaces::Value& x, std::size_t replicate = 0) {
    const std::string input = spaces::key(x);
    std::vector<Assessment> parts;
    for (std::size_t i = 0; i < evaluators_.size(); i++) {
      const Evaluator& e = evaluators_[i];
      const std::uint64_t stream = e.stochastic ? stream_id(input, i, replicate) : 0;
      CacheKey k{input, e.name, e.version, e.fidelity, stream};
      if (const Assessment* hit = cache_.find(k)) {
        parts.push_back(*hit);
        continue;
      }
      calx::random::Stream s(seed_, search::tag("assessment.assessor.stream"), static_cast<std::uint32_t>(stream), static_cast<std::uint32_t>(stream >> 32));
      Assessment a;
      try {
        a = e.assess(x, s);
      } catch (const std::exception& ex) {
        a = Assessment::outcome(Status::failed, e.name + " threw: " + ex.what());
      }
      runs_++;
      for (auto& m : a.metrics) {
        m.standing = weaker(m.standing, e.standing);
        if (m.evaluator.empty()) m.evaluator = e.name;
      }
      if (a.cost == 0) a.cost = e.cost;
      a.provenance.push_back({e.name, e.version, e.kind, e.standing, e.fidelity, stream, input, 1});
      parts.push_back(cache_.record(k, std::move(a)));
    }
    return merge(parts);
  }
  search::Evaluation evaluation(const spaces::Value& x) { return to_evaluation(assess(x), tiering_); }
  // For heuresis's evolve: assessments by the tiers, every one recorded.
  std::function<search::Evaluation(const spaces::Value&)> objective() {
    return [this](const spaces::Value& x) { return evaluation(x); };
  }
  // x on `replicates` fresh replicates (from `first`): each metric the mean of its replicates with its standard error.
  Assessment reassess(const spaces::Value& x, std::size_t replicates, std::size_t first = 1u << 20) {
    if (replicates == 0) throw Refusal("a re-assessment needs at least one replicate");
    std::vector<Assessment> runs;
    for (std::size_t r = 0; r < replicates; r++) runs.push_back(assess(x, first + r));
    Assessment out = runs[0];
    for (const auto& a : runs)
      if (!a.ok()) return a;
    for (auto& m : out.metrics) {
      std::vector<double> v;
      for (const auto& a : runs) v.push_back(a.value(m.name));
      const Metric merged = from_samples(m.name, m.role, v, m.standing, m.evaluator);
      const bool exact = std::all_of(v.begin(), v.end(), [&](double y) { return y == v[0]; });
      m.value = merged.value, m.replicates = merged.replicates, m.samples = merged.samples;
      m.uncertainty = exact ? std::optional<double>(0.0) : merged.uncertainty;
    }
    out.cost = 0;
    for (const auto& a : runs) out.cost += a.cost;
    for (auto& p : out.provenance) p.replicates = replicates;
    return out;
  }

 private:
  std::uint64_t stream_id(const std::string& input, std::size_t evaluator, std::size_t replicate) const {
    (void)input;  // common random numbers: candidates share the draws of a replicate
    return (static_cast<std::uint64_t>(evaluator) << 40) ^ static_cast<std::uint64_t>(replicate);
  }
  std::vector<Evaluator> evaluators_;
  Tiering tiering_;
  std::uint64_t seed_;
  AssessmentCache cache_;
  std::size_t runs_ = 0;
};

}  // namespace heuresis::assessment
