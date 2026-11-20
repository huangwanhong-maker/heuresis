// Cascades from cheap to costly evaluators, a stored record of every assessment, and the agreement of a surrogate with what it
// stands in for (decision 0010; the survey, Part V, design §3–5).
//
// CASCADE. Fidelities are applied in turn, each with its cost and a declared bound ε on its error against the last (exact)
// fidelity. At each level the candidates within 2ε of the least value go on. If every bound holds, the last fidelity's minimiser
// is never dropped: with j* that minimiser and i the level's, value_j* ≤ exact_j* + ε ≤ exact_i + ε ≤ value_i + 2ε. The
// selection is then the one an exhaustive evaluation at the last fidelity makes, at the cost of the candidates each level keeps.
// A bound that is only an estimate makes the guarantee conditional on it, which the result says.
//
// CACHE. Assessments are recorded by the input, the evaluator, its version, the fidelity and the stream, and returned exactly when
// asked again; a record is never replaced. The entries are kept in the order recorded, so that a caller can seal them in a
// journal (the infrastructure seals them in archeion's).
//
// AGREEMENT. Each point the true evaluator re-evaluates is set beside what the surrogate predicted: Kendall's τ between them, the
// root-mean-square error, and the share of true values within the surrogate's intervals. A surrogate is trusted for screening
// only when enough points have been compared and τ and the coverage reach declared levels (model management after Jin 2005, "A
// comprehensive survey of fitness approximation in evolutionary computation", Soft Computing 9, 3–12).

#pragma once

#include <heuresis/assessment/record.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace heuresis::assessment {

// ---- cascade ----

struct Fidelity {
  std::string name;
  double cost = 1;
  double error = 0;          // declared bound on |value − the last fidelity's value|; the last must be 0
  bool error_estimated = false;
  std::function<double(std::size_t)> value;
};

struct Cascade {
  std::size_t best = 0;
  double cost = 0;
  std::vector<std::vector<std::size_t>> kept;  // the candidates evaluated at each level
  std::string guarantee;
};

inline Cascade cascade(std::size_t k, const std::vector<Fidelity>& levels) {
  if (k == 0 || levels.empty()) throw Refusal("a cascade needs candidates and fidelities");
  if (levels.back().error != 0) throw Refusal("a cascade's last fidelity is its reference: its error must be zero");
  Cascade c;
  std::vector<std::size_t> alive(k);
  for (std::size_t i = 0; i < k; i++) alive[i] = i;
  bool estimated = false;
  for (const auto& f : levels) {
    if (!(f.error >= 0) || !f.value) throw Refusal("a cascade's fidelity needs a value and an error at least zero");
    estimated = estimated || f.error_estimated;
    c.kept.push_back(alive);
    std::vector<double> v;
    for (auto i : alive) v.push_back(f.value(i)), c.cost += f.cost;
    const double least = *std::min_element(v.begin(), v.end());
    std::vector<std::size_t> next;
    for (std::size_t j = 0; j < alive.size(); j++)
      if (v[j] <= least + 2 * f.error) next.push_back(alive[j]);
    if (&f == &levels.back()) {
      c.best = alive[static_cast<std::size_t>(std::min_element(v.begin(), v.end()) - v.begin())];
      break;
    }
    alive = next;
  }
  c.guarantee = estimated ? "the last fidelity's minimiser is kept if the estimated error bounds hold" : "the last fidelity's minimiser is kept: the selection equals an exhaustive evaluation at the last fidelity";
  return c;
}

// ---- cache ----

struct CacheKey {
  std::string input, evaluator, version;
  double fidelity = 1;
  std::uint64_t stream = 0;
  std::string text() const {
    char f[32];
    std::snprintf(f, sizeof f, "%a", fidelity);
    return evaluator + "@" + version + "|" + f + "|" + std::to_string(stream) + "|" + input;
  }
};

class AssessmentCache {
 public:
  const Assessment* find(const CacheKey& k) {
    auto it = index_.find(k.text());
    if (it == index_.end()) {
      misses_++;
      return nullptr;
    }
    hits_++;
    return &entries_[it->second].second;
  }
  // Records an assessment under a key; a key already recorded keeps its first assessment, which is returned.
  const Assessment& record(const CacheKey& k, Assessment a) {
    const std::string t = k.text();
    auto it = index_.find(t);
    if (it != index_.end()) return entries_[it->second].second;
    index_.emplace(t, entries_.size());
    entries_.push_back({k, std::move(a)});
    return entries_.back().second;
  }
  const std::vector<std::pair<CacheKey, Assessment>>& entries() const { return entries_; }
  std::size_t hits() const { return hits_; }
  std::size_t misses() const { return misses_; }

 private:
  std::map<std::string, std::size_t> index_;
  std::vector<std::pair<CacheKey, Assessment>> entries_;
  std::size_t hits_ = 0, misses_ = 0;
};

// ---- agreement of a surrogate ----

struct Trust {
  std::size_t least = 10;   // points compared before any trust
  double tau = 0.7;         // least Kendall τ
  double coverage = 0.8;    // least share within the intervals (when the surrogate gives them)
  double z = 1.96;          // the intervals' half-width in standard deviations
};

class Agreement {
 public:
  void add(double predicted, double actual, std::optional<double> sd = std::nullopt) {
    predicted_.push_back(predicted), actual_.push_back(actual), sd_.push_back(sd);
  }
  std::size_t size() const { return actual_.size(); }
  double rmse() const {
    double s = 0;
    for (std::size_t i = 0; i < actual_.size(); i++) s += (predicted_[i] - actual_[i]) * (predicted_[i] - actual_[i]);
    return actual_.empty() ? 0 : std::sqrt(s / static_cast<double>(actual_.size()));
  }
  // Kendall's τ-b.
  double tau() const {
    double concordant = 0, discordant = 0, tie_p = 0, tie_a = 0;
    for (std::size_t i = 0; i < actual_.size(); i++)
      for (std::size_t j = i + 1; j < actual_.size(); j++) {
        const double dp = predicted_[i] - predicted_[j], da = actual_[i] - actual_[j];
        if (dp == 0 && da == 0) continue;
        if (dp == 0) tie_p++;
        else if (da == 0) tie_a++;
        else if ((dp > 0) == (da > 0)) concordant++;
        else discordant++;
      }
    const double n1 = concordant + discordant + tie_p, n2 = concordant + discordant + tie_a;
    return n1 > 0 && n2 > 0 ? (concordant - discordant) / std::sqrt(n1 * n2) : 0;
  }
  // The share of actual values within predicted ± z·sd, among points with an interval; nothing when none has one.
  std::optional<double> coverage(double z = 1.96) const {
    std::size_t with = 0, inside = 0;
    for (std::size_t i = 0; i < actual_.size(); i++)
      if (sd_[i]) with++, inside += std::abs(actual_[i] - predicted_[i]) <= z * *sd_[i];
    if (with == 0) return std::nullopt;
    return static_cast<double>(inside) / static_cast<double>(with);
  }
  // Nothing when trusted, else why not.
  std::optional<std::string> distrust(const Trust& t = {}) const {
    if (size() < t.least) return "only " + std::to_string(size()) + " points compared, fewer than " + std::to_string(t.least);
    const double k = tau();
    if (k < t.tau) return "Kendall's τ " + std::to_string(k) + " below " + std::to_string(t.tau);
    const auto c = coverage(t.z);
    if (c && *c < t.coverage) return "coverage " + std::to_string(*c) + " below " + std::to_string(t.coverage);
    return std::nullopt;
  }

 private:
  std::vector<double> predicted_, actual_;
  std::vector<std::optional<double>> sd_;
};

}  // namespace heuresis::assessment
