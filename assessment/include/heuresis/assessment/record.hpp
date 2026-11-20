// The assessment of a candidate by evaluators of many kinds (decision 0010; the infrastructure's survey Evolutionary
// heterogeneous resource allocation, Part V). A closed form, a simulation, an agent model, a surrogate, a language model and a
// person establish different things, and a score from one cannot stand for a score from another. An assessment therefore keeps,
// for every metric, its role (objective, constraint, descriptor, report), its value with its uncertainty and replicates, the
// per-unit outcomes it was summarised from (for equity), and its standing:
// - verified: established by a check apart from the computation that produced it;
// - computed: given exactly by a declared model;
// - estimated: a statistical estimate, with its uncertainty;
// - proposed: put forward by a language model or a person, never evidence of quality.
// A metric's standing is never stronger than its evaluator's. Provenance names the evaluator, its version, kind and fidelity, the
// random stream it drew from, and the input it assessed. A failure, refusal or abstention is an outcome with its reason, never a
// made-up value.

#pragma once

#include <heuresis/spaces/space.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <vector>

namespace heuresis::assessment {

using spaces::Refusal;

enum class Role { objective, constraint, descriptor, report };
enum class Standing { verified = 0, computed = 1, estimated = 2, proposed = 3 };  // a smaller value is stronger
enum class Kind { closed_form, simulation, agent_model, surrogate, language_model, person, other };
enum class Status { ok = 0, abstained = 1, refused = 2, failed = 3 };  // a larger value is worse

inline const char* name(Role r) {
  switch (r) {
    case Role::objective: return "objective";
    case Role::constraint: return "constraint";
    case Role::descriptor: return "descriptor";
    case Role::report: return "report";
  }
  return "";
}
inline const char* name(Standing s) {
  switch (s) {
    case Standing::verified: return "verified";
    case Standing::computed: return "computed";
    case Standing::estimated: return "estimated";
    case Standing::proposed: return "proposed";
  }
  return "";
}
inline const char* name(Kind k) {
  switch (k) {
    case Kind::closed_form: return "closed form";
    case Kind::simulation: return "simulation";
    case Kind::agent_model: return "agent model";
    case Kind::surrogate: return "surrogate";
    case Kind::language_model: return "language model";
    case Kind::person: return "person";
    case Kind::other: return "other";
  }
  return "";
}
inline const char* name(Status s) {
  switch (s) {
    case Status::ok: return "ok";
    case Status::abstained: return "abstained";
    case Status::refused: return "refused";
    case Status::failed: return "failed";
  }
  return "";
}
inline Standing weaker(Standing a, Standing b) { return static_cast<int>(a) >= static_cast<int>(b) ? a : b; }

struct Metric {
  std::string name;
  Role role = Role::objective;
  double value = 0;                   // minimised when an objective; satisfied when ≤ 0 when a constraint
  std::optional<double> uncertainty;  // the standard error of the value: 0 when exact, nothing when unknown
  std::size_t replicates = 1;
  std::vector<double> samples;        // the replicates, when kept
  std::vector<double> units;          // per-unit outcomes the value summarises, when kept
  Standing standing = Standing::computed;
  std::string evaluator;
};

struct Provenance {
  std::string evaluator, version;
  Kind kind = Kind::other;
  Standing standing = Standing::computed;
  double fidelity = 1;
  std::uint64_t stream = 0;
  std::string input;
  std::size_t replicates = 1;
};

struct Assessment {
  std::vector<Metric> metrics;
  double cost = 0;
  Status status = Status::ok;
  std::string reason;
  std::vector<Provenance> provenance;

  bool ok() const { return status == Status::ok; }
  const Metric* find(const std::string& n) const {
    for (const auto& m : metrics)
      if (m.name == n) return &m;
    return nullptr;
  }
  double value(const std::string& n) const {
    const Metric* m = find(n);
    if (!m) throw Refusal("an assessment has no metric named " + n);
    return m->value;
  }
  static Assessment outcome(Status s, std::string why) {
    Assessment a;
    a.status = s, a.reason = std::move(why);
    return a;
  }
};

// The metrics of several assessments of one candidate side by side; the worst status stands, with every reason.
inline Assessment merge(const std::vector<Assessment>& parts) {
  Assessment out;
  std::set<std::string> names;
  for (const auto& a : parts) {
    for (const auto& m : a.metrics) {
      if (!names.insert(m.name).second) throw Refusal("two evaluators report a metric named " + m.name);
      out.metrics.push_back(m);
    }
    out.provenance.insert(out.provenance.end(), a.provenance.begin(), a.provenance.end());
    out.cost += a.cost;
    if (static_cast<int>(a.status) > static_cast<int>(out.status)) out.status = a.status;
    if (!a.reason.empty()) out.reason += (out.reason.empty() ? "" : "; ") + a.reason;
  }
  return out;
}

// A metric from replicates: their mean and its standard error.
inline Metric from_samples(std::string name, Role role, std::vector<double> samples, Standing standing, std::string evaluator = {}, bool keep = true) {
  if (samples.empty()) throw Refusal("a metric from no samples");
  const double n = static_cast<double>(samples.size());
  double mean = 0;
  for (double v : samples) mean += v;
  mean /= n;
  double ss = 0;
  for (double v : samples) ss += (v - mean) * (v - mean);
  Metric m;
  m.name = std::move(name), m.role = role, m.value = mean, m.standing = standing, m.evaluator = std::move(evaluator);
  m.replicates = samples.size();
  if (samples.size() > 1) m.uncertainty = std::sqrt(ss / (n - 1) / n);
  if (keep) m.samples = std::move(samples);
  return m;
}

}  // namespace heuresis::assessment
