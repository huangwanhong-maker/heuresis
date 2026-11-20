// Spaces built from spaces: the product (a record of named parts), the choice (a tagged union) and the constrained subspace (a
// predicate with a repair, else rejection, counted). Sequences of a space's values are SequenceSpace (base.hpp); sets and
// multisets of a finite space's values are Subset and the multisets of base.hpp over its elements, by index.
//
// - Product. Its distance is Σ wᵢ dᵢ/uᵢ, each part's distance divided by its unit so that no part dominates for being wide.
//   Recombination part by part is geometric under that distance when every part's is geometric under its own, even when the
//   parts are coupled (Moraglio 2007, Thm 9.2.2 and Cor. 9.2.3); a part without recombination contributes one parent's value,
//   which is geometric for any metric. A mutation moves each part that can move with probability max(1/m, scale) (m the number
//   of such parts), at least one, each at the given scale: at scale 1 every part moves about a unit, at small scales about one
//   part does. The unit is Σ wᵢ, exactly the mean distance when every part's unit is its mean distance; otherwise an
//   approximation, said so.
// - Choice. A value is a branch and its payload. Within a branch the distance is the payload's divided by its unit; across
//   branches it is a constant C, the jump. With C at least half the largest scaled diameter the triangle inequality holds; where
//   a branch is unbounded, distances within it are truncated at 2C, which keeps a metric and is said to cost geometry. The
//   segment between values of different branches is the two values alone, so recombination across branches takes one parent.
//   Sampling chooses a branch with its weight (uniformly by default), or in proportion to the branches' sizes, which is uniform
//   over a finite union (Boltzmann's principle, Duchon et al. 2004). A choice keeps only the active payload; a representation
//   that keeps dormant payloads is a product with a categorical selector.
// - Constrained subspace. A predicate selects valid values of a base space. Sampling, mutation and recombination retry within a
//   bound and count what they reject; past the bound a declared repair is applied, else a mutation returns its parent and a
//   recombination one of its parents. Retried children stay on the base segment, so recombination is geometric when the base's
//   is and no repair is declared.

#pragma once

#include <heuresis/spaces/base.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace heuresis::spaces {

namespace detail {
inline double scaled(double d, double unit) { return unit > 0 ? d / unit : d; }
}  // namespace detail

// ---------------------------------------------------------------------------------------------------------- product ----

struct Part {
  std::string name;
  SpacePtr space;
  double weight = 1;
};

class Product final : public Space {
 public:
  explicit Product(std::vector<Part> parts) : parts_(std::move(parts)) {
    if (parts_.empty()) throw Refusal("a product needs at least one part");
    for (std::size_t i = 0; i < parts_.size(); i++) {
      if (!parts_[i].space) throw Refusal("a product's part " + parts_[i].name + " has no space");
      if (!(parts_[i].weight >= 0) || !std::isfinite(parts_[i].weight)) throw Refusal("a product's weights must be finite and at least zero");
      if (!index_.emplace(parts_[i].name, i).second) throw Refusal("a product's part names must differ: " + parts_[i].name + " twice");
      units_.push_back(parts_[i].space->unit());
      if (parts_[i].space->change_floor() > 0) movable_.push_back(i);
    }
  }
  std::size_t size() const { return parts_.size(); }
  const Part& part(std::size_t i) const { return parts_.at(i); }
  std::size_t index(const std::string& name) const {
    auto it = index_.find(name);
    if (it == index_.end()) throw Refusal("a product has no part named " + name);
    return it->second;
  }
  const Value& get(const Value& x, const std::string& name) const { return x.parts.at(index(name)); }
  std::string describe() const override {
    std::string s = "a product of " + std::to_string(parts_.size()) + " parts (";
    for (std::size_t i = 0; i < parts_.size(); i++) s += (i ? ", " : "") + parts_[i].name;
    return s + ")";
  }

  Value sample(calx::random::Stream& s, double size = 1) const override {
    std::vector<Value> xs;
    for (const auto& p : parts_) xs.push_back(p.space->sample(s, size));
    return record_value(std::move(xs));
  }
  Value mutate(const Value& x, double scale, calx::random::Stream& s) const override {
    detail::require(*this, x, "mutate");
    if (movable_.empty()) return x;
    const double q = std::max(1.0 / static_cast<double>(movable_.size()), detail::clamp_scale(scale));
    std::vector<char> chosen(parts_.size(), 0);
    bool any = false;
    for (auto i : movable_)
      if (s.uniform() < q) chosen[i] = 1, any = true;
    if (!any) chosen[movable_[s.below(movable_.size())]] = 1;
    Value y = x;
    for (std::size_t i = 0; i < parts_.size(); i++)
      if (chosen[i]) y.parts[i] = parts_[i].space->mutate(x.parts[i], scale, s);
    return y;
  }
  double distance(const Value& x, const Value& y) const override {
    detail::require(*this, x, "distance");
    detail::require(*this, y, "distance");
    double d = 0;
    for (std::size_t i = 0; i < parts_.size(); i++)
      if (parts_[i].weight > 0) d += parts_[i].weight * detail::scaled(parts_[i].space->distance(x.parts[i], y.parts[i]), units_[i]);
    return d;
  }
  double unit() const override {
    double u = 0;
    for (std::size_t i = 0; i < parts_.size(); i++)
      if (units_[i] > 0) u += parts_[i].weight;
    return u;
  }
  std::optional<std::string> invalid(const Value& x) const override {
    if (x.kind != Value::Kind::record) return std::string("a ") + name_of(x.kind) + " value, not a record";
    if (x.parts.size() != parts_.size()) return "a record of " + std::to_string(x.parts.size()) + " parts, not " + std::to_string(parts_.size());
    for (std::size_t i = 0; i < parts_.size(); i++)
      if (auto why = parts_[i].space->invalid(x.parts[i])) return "part " + parts_[i].name + ": " + *why;
    return std::nullopt;
  }
  bool can_recombine() const override { return true; }
  Value recombine(const Value& x, const Value& y, calx::random::Stream& s) const override {
    detail::require(*this, x, "recombine");
    detail::require(*this, y, "recombine");
    std::vector<Value> c;
    for (std::size_t i = 0; i < parts_.size(); i++) {
      const auto& sp = *parts_[i].space;
      if (x.parts[i] == y.parts[i]) c.push_back(x.parts[i]);
      else if (sp.can_recombine()) c.push_back(sp.recombine(x.parts[i], y.parts[i], s));
      else c.push_back(s.uniform() < 0.5 ? x.parts[i] : y.parts[i]);
    }
    return record_value(std::move(c));
  }
  Properties properties() const override {
    Properties p{true, true, true, true, Unit::mean, true};
    for (const auto& part : parts_) {
      const auto& sp = *part.space;
      const Properties q = sp.properties();
      if (sp.can_recombine()) p.geometric &= q.geometric, p.pure &= q.pure, p.respectful &= q.respectful;
      p.exact_metric &= q.exact_metric;
      p.unit_exact &= q.unit_exact && q.unit == Unit::mean;
    }
    return p;
  }
  bool can_repair() const override {
    return std::all_of(parts_.begin(), parts_.end(), [](const Part& p) { return p.space->can_repair(); });
  }
  Value repair(const Value& x) const override {
    if (valid(x)) return x;
    if (x.kind != Value::Kind::record || x.parts.size() != parts_.size()) throw Refusal("repair on " + describe() + ": a value of another shape");
    Value y = x;
    for (std::size_t i = 0; i < parts_.size(); i++)
      if (!parts_[i].space->valid(y.parts[i])) y.parts[i] = parts_[i].space->repair(y.parts[i]);
    return y;
  }
  std::optional<double> mutation_radius(double scale) const override {
    double r = 0;
    for (std::size_t i = 0; i < parts_.size(); i++) {
      if (parts_[i].weight == 0 || parts_[i].space->change_floor() == 0) continue;
      const auto ri = parts_[i].space->mutation_radius(scale);
      if (!ri) return std::nullopt;
      r += parts_[i].weight * detail::scaled(*ri, units_[i]);
    }
    return r;
  }
  double change_floor() const override {
    double f = movable_.empty() ? 0 : 1;
    for (auto i : movable_) f = std::min(f, parts_[i].space->change_floor());
    return f;
  }
  double diameter() const override {
    double d = 0;
    for (std::size_t i = 0; i < parts_.size(); i++)
      if (parts_[i].weight > 0) d += parts_[i].weight * detail::scaled(parts_[i].space->diameter(), units_[i]);
    return d;
  }
  bool has_chart() const override {
    return std::all_of(parts_.begin(), parts_.end(), [](const Part& p) { return p.space->has_chart(); });
  }
  std::size_t chart_dimension() const override {
    std::size_t d = 0;
    for (const auto& p : parts_) d += p.space->chart_dimension();
    return d;
  }
  std::vector<double> log(const Value& x, const Value& y) const override {
    if (!has_chart()) throw Refusal(describe() + " has no chart: a part has none");
    detail::require(*this, x, "log");
    detail::require(*this, y, "log");
    std::vector<double> v;
    for (std::size_t i = 0; i < parts_.size(); i++) {
      const auto vi = parts_[i].space->log(x.parts[i], y.parts[i]);
      v.insert(v.end(), vi.begin(), vi.end());
    }
    return v;
  }
  Value retract(const Value& x, std::span<const double> v) const override {
    if (!has_chart()) throw Refusal(describe() + " has no chart: a part has none");
    detail::require(*this, x, "retract");
    if (v.size() != chart_dimension()) throw Refusal("retract on " + describe() + ": a tangent vector of another dimension");
    Value y = x;
    std::size_t at = 0;
    for (std::size_t i = 0; i < parts_.size(); i++) {
      const std::size_t k = parts_[i].space->chart_dimension();
      y.parts[i] = parts_[i].space->retract(x.parts[i], v.subspan(at, k));
      at += k;
    }
    return y;
  }
  double trust_radius(const Value& x) const override {
    double r = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < parts_.size(); i++) r = std::min(r, parts_[i].space->trust_radius(x.parts.at(i)));
    return r;
  }
  std::optional<std::size_t> embedding_dimension() const override {
    std::size_t d = 0;
    for (const auto& p : parts_) {
      const auto di = p.space->embedding_dimension();
      if (!di) return std::nullopt;
      d += *di;
    }
    return d;
  }
  bool embedding_bijective() const override {
    return std::all_of(parts_.begin(), parts_.end(), [](const Part& p) { return p.space->embedding_bijective(); });
  }
  std::vector<double> embed(const Value& x) const override {
    if (!embedding_dimension()) throw Refusal(describe() + " has no embedding: a part has none");
    detail::require(*this, x, "embed");
    std::vector<double> z;
    for (std::size_t i = 0; i < parts_.size(); i++) {
      const auto zi = parts_[i].space->embed(x.parts[i]);
      z.insert(z.end(), zi.begin(), zi.end());
    }
    return z;
  }
  Value unembed(std::span<const double> z) const override {
    const auto d = embedding_dimension();
    if (!d) throw Refusal(describe() + " has no embedding: a part has none");
    if (z.size() != *d) throw Refusal("unembed on " + describe() + ": coordinates of another dimension");
    std::vector<Value> xs;
    std::size_t at = 0;
    for (const auto& p : parts_) {
      const std::size_t k = *p.space->embedding_dimension();
      xs.push_back(p.space->unembed(z.subspan(at, k)));
      at += k;
    }
    return record_value(std::move(xs));
  }
  std::optional<std::size_t> cardinality() const override {
    unsigned __int128 c = 1;
    for (const auto& p : parts_) {
      const auto ci = p.space->cardinality();
      if (!ci) return std::nullopt;
      c *= *ci;
      if (c > (static_cast<unsigned __int128>(1) << 62)) return std::nullopt;
    }
    return static_cast<std::size_t>(c);
  }
  std::vector<Value> elements(std::size_t limit) const override {
    const auto c = cardinality();
    if (!c || *c > limit) throw Refusal(describe() + " has more values than the limit");
    std::vector<std::vector<Value>> lists;
    for (const auto& p : parts_) lists.push_back(p.space->elements(limit));
    std::vector<Value> out;
    std::vector<std::size_t> at(parts_.size(), 0);
    for (std::size_t k = 0; k < *c; k++) {
      std::vector<Value> xs;
      for (std::size_t i = 0; i < parts_.size(); i++) xs.push_back(lists[i][at[i]]);
      out.push_back(record_value(std::move(xs)));
      for (std::size_t i = parts_.size(); i-- > 0;) {
        if (++at[i] < lists[i].size()) break;
        at[i] = 0;
      }
    }
    return out;
  }

 private:
  std::vector<Part> parts_;
  std::map<std::string, std::size_t> index_;
  std::vector<double> units_;
  std::vector<std::size_t> movable_;
};

// ----------------------------------------------------------------------------------------------------------- choice ----

struct Branch {
  std::string name;
  SpacePtr space;
  double weight = 1;  // the chance of sampling this branch, relative to the others (ignored when sampling by size)
};

class Choice final : public Space {
 public:
  enum class Sampling { by_weight, by_size };
  // jump ≤ 0 takes the default: max(1, half the largest scaled diameter) where every branch is bounded, else 1.
  explicit Choice(std::vector<Branch> branches, double jump = 0, Sampling sampling = Sampling::by_weight) : branches_(std::move(branches)), sampling_(sampling) {
    if (branches_.empty()) throw Refusal("a choice needs at least one branch");
    double half = 0;
    bounded_ = true;
    for (const auto& b : branches_) {
      if (!b.space) throw Refusal("a choice's branch " + b.name + " has no space");
      units_.push_back(b.space->unit());
      const double d = detail::scaled(b.space->diameter(), b.space->unit());
      if (!std::isfinite(d)) bounded_ = false;
      else half = std::max(half, d / 2);
    }
    jump_ = jump > 0 ? jump : (bounded_ ? std::max(1.0, half) : 1.0);
    truncates_ = !bounded_ || half > jump_;
    if (sampling_ == Sampling::by_size) {
      for (const auto& b : branches_) {
        const auto c = b.space->cardinality();
        if (!c) throw Refusal("a choice sampled by size needs every branch finite: " + b.name + " is not");
        chance_.push_back(static_cast<double>(*c));
      }
    } else {
      for (const auto& b : branches_) {
        if (!(b.weight > 0) || !std::isfinite(b.weight)) throw Refusal("a choice's weights must be positive and finite");
        chance_.push_back(b.weight);
      }
    }
    double total = 0;
    for (double c : chance_) total += c;
    for (double& c : chance_) c /= total;
  }
  std::size_t size() const { return branches_.size(); }
  const Branch& branch(std::size_t i) const { return branches_.at(i); }
  double jump() const { return jump_; }
  std::string describe() const override {
    std::string s = "a choice among " + std::to_string(branches_.size()) + " branches (";
    for (std::size_t i = 0; i < branches_.size(); i++) s += (i ? ", " : "") + branches_[i].name;
    return s + ")";
  }

  Value sample(calx::random::Stream& s, double size = 1) const override {
    const std::size_t b = detail::pick(chance_, s);
    return choice_value(static_cast<std::int64_t>(b), branches_[b].space->sample(s, size));
  }
  Value mutate(const Value& x, double scale, calx::random::Stream& s) const override {
    detail::require(*this, x, "mutate");
    const auto b = static_cast<std::size_t>(x.ints[0]);
    const bool stays = branches_[b].space->change_floor() > 0;
    if (branches_.size() > 1 && (!stays || s.uniform() < detail::clamp_scale(scale) / 2)) {
      std::size_t c = static_cast<std::size_t>(s.below(branches_.size() - 1));
      if (c >= b) c++;
      return choice_value(static_cast<std::int64_t>(c), branches_[c].space->sample(s));
    }
    if (!stays) return x;
    return choice_value(x.ints[0], branches_[b].space->mutate(x.parts[0], scale, s));
  }
  double distance(const Value& x, const Value& y) const override {
    detail::require(*this, x, "distance");
    detail::require(*this, y, "distance");
    if (x.ints[0] != y.ints[0]) return jump_;
    const auto b = static_cast<std::size_t>(x.ints[0]);
    return std::min(2 * jump_, detail::scaled(branches_[b].space->distance(x.parts[0], y.parts[0]), units_[b]));
  }
  double unit() const override {
    double same = 0, within = 0;
    for (std::size_t i = 0; i < branches_.size(); i++) {
      same += chance_[i] * chance_[i];
      within += chance_[i] * chance_[i] * (units_[i] > 0 ? 1 : 0);
    }
    return within + (1 - same) * jump_;
  }
  std::optional<std::string> invalid(const Value& x) const override {
    if (x.kind != Value::Kind::choice || x.ints.size() != 1 || x.parts.size() != 1) return std::string("a ") + name_of(x.kind) + " value, not a choice";
    if (x.ints[0] < 0 || x.ints[0] >= static_cast<std::int64_t>(branches_.size())) return "branch " + std::to_string(x.ints[0]) + " outside 0…" + std::to_string(branches_.size() - 1);
    if (auto why = branches_[static_cast<std::size_t>(x.ints[0])].space->invalid(x.parts[0])) return "branch " + branches_[static_cast<std::size_t>(x.ints[0])].name + ": " + *why;
    return std::nullopt;
  }
  bool can_recombine() const override { return true; }
  Value recombine(const Value& x, const Value& y, calx::random::Stream& s) const override {
    detail::require(*this, x, "recombine");
    detail::require(*this, y, "recombine");
    if (x.ints[0] != y.ints[0] || x == y) return s.uniform() < 0.5 ? x : y;
    const auto& sp = *branches_[static_cast<std::size_t>(x.ints[0])].space;
    if (!sp.can_recombine()) return s.uniform() < 0.5 ? x : y;
    return choice_value(x.ints[0], sp.recombine(x.parts[0], y.parts[0], s));
  }
  Properties properties() const override {
    Properties p{!truncates_, true, true, true, Unit::mean, !truncates_};
    for (const auto& b : branches_) {
      const Properties q = b.space->properties();
      if (b.space->can_recombine()) p.geometric &= q.geometric, p.pure &= q.pure, p.respectful &= q.respectful;
      p.exact_metric &= q.exact_metric;
      p.unit_exact &= q.unit_exact && q.unit == Unit::mean;
    }
    return p;
  }
  bool can_repair() const override {
    return std::all_of(branches_.begin(), branches_.end(), [](const Branch& b) { return b.space->can_repair(); });
  }
  Value repair(const Value& x) const override {
    if (valid(x)) return x;
    if (x.kind != Value::Kind::choice || x.ints.size() != 1 || x.parts.size() != 1 || x.ints[0] < 0 || x.ints[0] >= static_cast<std::int64_t>(branches_.size()))
      throw Refusal("repair on " + describe() + ": a value of another shape");
    return choice_value(x.ints[0], branches_[static_cast<std::size_t>(x.ints[0])].space->repair(x.parts[0]));
  }
  double change_floor() const override {
    if (branches_.size() > 1) return 1;
    return branches_[0].space->change_floor();
  }
  double diameter() const override {
    double d = branches_.size() > 1 ? jump_ : 0;
    for (std::size_t i = 0; i < branches_.size(); i++) d = std::max(d, std::min(2 * jump_, detail::scaled(branches_[i].space->diameter(), units_[i])));
    return d;
  }
  // A one-hot branch followed by every branch's coordinates; the inactive branches' coordinates are zero and unread.
  std::optional<std::size_t> embedding_dimension() const override {
    std::size_t d = branches_.size();
    for (const auto& b : branches_) {
      const auto di = b.space->embedding_dimension();
      if (!di) return std::nullopt;
      d += *di;
    }
    return d;
  }
  std::vector<double> embed(const Value& x) const override {
    const auto d = embedding_dimension();
    if (!d) throw Refusal(describe() + " has no embedding: a branch has none");
    detail::require(*this, x, "embed");
    std::vector<double> z(*d, 0);
    const auto b = static_cast<std::size_t>(x.ints[0]);
    z[b] = 1;
    std::size_t at = branches_.size();
    for (std::size_t i = 0; i < b; i++) at += *branches_[i].space->embedding_dimension();
    const auto zi = branches_[b].space->embed(x.parts[0]);
    std::copy(zi.begin(), zi.end(), z.begin() + static_cast<std::ptrdiff_t>(at));
    return z;
  }
  Value unembed(std::span<const double> z) const override {
    const auto d = embedding_dimension();
    if (!d) throw Refusal(describe() + " has no embedding: a branch has none");
    if (z.size() != *d) throw Refusal("unembed on " + describe() + ": coordinates of another dimension");
    std::size_t b = 0;
    for (std::size_t i = 1; i < branches_.size(); i++)
      if (z[i] > z[b]) b = i;
    std::size_t at = branches_.size();
    for (std::size_t i = 0; i < b; i++) at += *branches_[i].space->embedding_dimension();
    return choice_value(static_cast<std::int64_t>(b), branches_[b].space->unembed(z.subspan(at, *branches_[b].space->embedding_dimension())));
  }
  std::optional<std::size_t> cardinality() const override {
    std::uint64_t c = 0;
    for (const auto& b : branches_) {
      const auto ci = b.space->cardinality();
      if (!ci || c + *ci > (std::uint64_t{1} << 62)) return std::nullopt;
      c += *ci;
    }
    return static_cast<std::size_t>(c);
  }
  std::vector<Value> elements(std::size_t limit) const override {
    const auto c = cardinality();
    if (!c || *c > limit) throw Refusal(describe() + " has more values than the limit");
    std::vector<Value> out;
    for (std::size_t i = 0; i < branches_.size(); i++)
      for (auto& v : branches_[i].space->elements(limit)) out.push_back(choice_value(static_cast<std::int64_t>(i), std::move(v)));
    return out;
  }

 private:
  std::vector<Branch> branches_;
  Sampling sampling_;
  std::vector<double> units_, chance_;
  double jump_ = 1;
  bool bounded_ = true, truncates_ = false;
};

// ------------------------------------------------------------------------------------------------ constrained space ----

class Constrained final : public Space {
 public:
  using Rule = std::function<std::optional<std::string>(const Value&)>;  // a reason a value is excluded, or nothing
  using Fix = std::function<Value(const Value&)>;                         // a repair onto the subspace

  Constrained(SpacePtr base, std::string name, Rule rule, Fix fix = {}, std::size_t tries = 1000)
      : base_(std::move(base)), name_(std::move(name)), rule_(std::move(rule)), fix_(std::move(fix)), tries_(tries) {
    if (!base_) throw Refusal("a constrained space needs a base space");
    if (!rule_) throw Refusal("a constrained space needs a rule");
    if (tries_ == 0) throw Refusal("a constrained space needs at least one try");
  }
  const SpacePtr& base() const { return base_; }
  // Values drawn by the base and rejected by the rule, over the space's life.
  std::uint64_t rejections() const { return rejections_.load(); }
  std::uint64_t repairs() const { return repairs_.load(); }
  std::string describe() const override { return base_->describe() + " where " + name_; }

  Value sample(calx::random::Stream& s, double size = 1) const override {
    Value last;
    for (std::size_t t = 0; t < tries_; t++) {
      last = base_->sample(s, size);
      if (!rule_(last)) return last;
      rejections_++;
    }
    if (fix_) return fixed(last);
    throw Refusal("no value of " + describe() + " found in " + std::to_string(tries_) + " draws, and no repair declared");
  }
  Value mutate(const Value& x, double scale, calx::random::Stream& s) const override {
    detail::require(*this, x, "mutate");
    Value last = x;
    for (std::size_t t = 0; t < tries_; t++) {
      last = base_->mutate(x, scale, s);
      if (!rule_(last)) return last;
      rejections_++;
    }
    return fix_ ? fixed(last) : x;
  }
  double distance(const Value& x, const Value& y) const override { return base_->distance(x, y); }
  double unit() const override { return base_->unit(); }
  std::optional<std::string> invalid(const Value& x) const override {
    if (auto why = base_->invalid(x)) return why;
    if (auto why = rule_(x)) return name_ + ": " + *why;
    return std::nullopt;
  }
  bool can_recombine() const override { return base_->can_recombine(); }
  Value recombine(const Value& x, const Value& y, calx::random::Stream& s) const override {
    detail::require(*this, x, "recombine");
    detail::require(*this, y, "recombine");
    Value last = x;
    for (std::size_t t = 0; t < tries_; t++) {
      last = base_->recombine(x, y, s);
      if (!rule_(last)) return last;
      rejections_++;
    }
    return fix_ ? fixed(last) : (s.uniform() < 0.5 ? x : y);
  }
  Properties properties() const override {
    Properties p = base_->properties();
    p.geometric = p.geometric && !fix_;
    p.unit_exact = false;  // the base's unit, not the subspace's
    return p;
  }
  bool can_repair() const override { return static_cast<bool>(fix_); }
  Value repair(const Value& x) const override {
    if (!fix_) throw Refusal(describe() + " has no repair");
    if (valid(x)) return x;
    return fixed(base_->can_repair() && !base_->valid(x) ? base_->repair(x) : x);
  }
  std::optional<double> mutation_radius(double scale) const override { return fix_ ? std::nullopt : base_->mutation_radius(scale); }
  double diameter() const override { return base_->diameter(); }  // a bound
  std::optional<std::size_t> embedding_dimension() const override { return fix_ ? base_->embedding_dimension() : std::nullopt; }
  std::vector<double> embed(const Value& x) const override {
    if (!fix_) throw Refusal(describe() + " has no embedding without a repair");
    detail::require(*this, x, "embed");
    return base_->embed(x);
  }
  Value unembed(std::span<const double> z) const override {
    if (!fix_) throw Refusal(describe() + " has no embedding without a repair");
    return repair(base_->unembed(z));
  }
  std::optional<std::size_t> cardinality() const override {
    const auto c = base_->cardinality();
    if (!c || *c > 1'000'000) return std::nullopt;
    std::size_t k = 0;
    for (const auto& v : base_->elements(*c)) k += !rule_(v);
    return k;
  }
  std::vector<Value> elements(std::size_t limit) const override {
    const auto c = base_->cardinality();
    if (!c || *c > std::max<std::size_t>(limit, 1'000'000)) throw Refusal(describe() + " has more values than the limit");
    std::vector<Value> out;
    for (auto& v : base_->elements(*c))
      if (!rule_(v)) out.push_back(std::move(v));
    if (out.size() > limit) throw Refusal(describe() + " has more values than the limit");
    return out;
  }

 private:
  Value fixed(const Value& x) const {
    repairs_++;
    Value r = fix_(x);
    if (auto why = invalid(r)) throw Refusal("the repair of " + describe() + " gave an invalid value: " + *why);
    return r;
  }
  SpacePtr base_;
  std::string name_;
  Rule rule_;
  Fix fix_;
  std::size_t tries_;
  mutable std::atomic<std::uint64_t> rejections_{0}, repairs_{0};
};

}  // namespace heuresis::spaces
