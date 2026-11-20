// The laws every space is held to, written once and run on any space, built in or a person's (the infrastructure's survey
// Evolutionary heterogeneous resource allocation, Part III, design §2). A law that a space does not claim, or whose operation it
// does not supply, is reported as not applying; a law that fails is reported with its worst deviation and an example, not
// refused, since an operator that is not geometric can still serve a search.
//
// The laws:
// - samples, mutations, children and repairs are valid;
// - the metric: d(x, x) = 0, d(x, y) = d(y, x) ≥ 0, d(x, z) ≤ d(x, y) + d(y, z), on sampled triples; distances within the
//   stated diameter;
// - a mutation stays within the ball it states, and moves at least as often as its stated floor;
// - recombination is pure, recombine(x, x) = x; where geometric is claimed, children lie on the segment,
//   d(x, c) + d(c, y) = d(x, y), on random pairs and on near pairs (a parent and its mutant);
// - repair is the identity on valid values, and idempotent with valid results on values made invalid;
// - a chart: retract_x(0) = x, log_x(retract_x(v)) = v within the trust radius, retract_x(log_x(y)) = y;
// - an embedding: unembed(embed(x)) = x; embed(unembed(z)) = z where it is bijective;
// - the unit: the sample mean (or root-mean-square) distance between independent samples agrees with the stated unit within
//   five standard errors where the unit is exact, within a quarter where it is an estimate;
// - a finite space lists exactly as many values as it counts, each valid and each once;
// - keys identify values: two values have one key exactly when they are equal.

#pragma once

#include <heuresis/spaces/space.hpp>

#include <cmath>
#include <limits>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace heuresis::spaces {

struct Law {
  std::string name;
  bool applies = true;         // false where the space does not claim or supply what the law concerns
  std::size_t trials = 0, failures = 0;
  double worst = 0;            // the largest deviation seen
  std::string example;         // the first failure, in words
  bool holds() const { return !applies || failures == 0; }
};

struct LawReport {
  std::string space;
  std::vector<Law> laws;
  bool all_hold() const {
    for (const auto& l : laws)
      if (!l.holds()) return false;
    return true;
  }
  std::size_t applied() const {
    std::size_t k = 0;
    for (const auto& l : laws) k += l.applies && l.trials > 0;
    return k;
  }
  std::string failures() const {
    std::string s;
    for (const auto& l : laws)
      if (!l.holds()) s += (s.empty() ? "" : "; ") + l.name + " (" + std::to_string(l.failures) + "/" + std::to_string(l.trials) + ", worst " + std::to_string(l.worst) + ": " + l.example + ")";
    return s;
  }
  const Law* find(const std::string& name) const {
    for (const auto& l : laws)
      if (l.name == name) return &l;
    return nullptr;
  }
};

struct LawOptions {
  std::size_t trials = 200;          // per law
  double tolerance = 1e-9;           // relative, for equalities of distances
  double chart_tolerance = 1e-7;     // relative, for the chart's identities (trigonometry loses digits)
  std::size_t unit_pairs = 4000;     // pairs for the unit
  double unit_sigmas = 5;
  double estimate_tolerance = 0.25;  // relative, for a unit that is an estimate
  std::size_t elements_limit = 20000;
  double scale = 0.3;                // the mutation scale tried, beside 1 and 0.01
};

namespace detail {

// A value made invalid in a way a repair might undo: coordinates pushed out, items repeated or out of range, kinds kept.
inline Value perturb(const Value& v, calx::random::Stream& s) {
  Value w = v;
  switch (v.kind) {
    case Value::Kind::real:
    case Value::Kind::point:
      for (double& x : w.reals) x = 2 * x + 1 + s.normal();
      break;
    case Value::Kind::integer:
      for (auto& x : w.ints) x += 7 + static_cast<std::int64_t>(s.below(5));
      break;
    case Value::Kind::category: w.ints[0] += 1000; break;
    case Value::Kind::permutation:
      if (w.ints.size() > 1) w.ints[1] = w.ints[0];
      else w.ints.push_back(-1);
      break;
    case Value::Kind::set: w.ints.insert(w.ints.begin(), -1); break;
    case Value::Kind::sequence:
    case Value::Kind::record:
    case Value::Kind::choice:
      if (!w.parts.empty()) {
        const std::size_t i = static_cast<std::size_t>(s.below(w.parts.size()));
        w.parts[i] = perturb(w.parts[i], s);
      }
      break;
  }
  return w;
}

struct Tally {
  Law law;
  explicit Tally(std::string name, bool applies = true) { law.name = std::move(name), law.applies = applies; }
  void trial(bool ok, double deviation, const std::function<std::string()>& what) {
    law.trials++;
    if (std::isfinite(deviation)) law.worst = std::max(law.worst, deviation);
    else law.worst = deviation;
    if (!ok) {
      if (law.failures == 0) law.example = what();
      law.failures++;
    }
  }
};

inline double norm(const std::vector<double>& v) {
  double s = 0;
  for (double x : v) s += x * x;
  return std::sqrt(s);
}

}  // namespace detail

// Runs every law on a space with draws from a stream; operations that throw count as failures with the reason.
inline LawReport check_laws(const Space& sp, calx::random::Stream& s, const LawOptions& o = {}) {
  LawReport report;
  report.space = sp.describe();
  const Properties props = sp.properties();
  const double unit = sp.unit();
  const double tol_d = o.tolerance * (1 + unit);
  auto guarded = [&](detail::Tally& t, const std::function<void()>& f) {
    try {
      f();
    } catch (const std::exception& e) {
      t.trial(false, std::numeric_limits<double>::infinity(), [&] { return std::string("threw: ") + e.what(); });
    }
  };

  std::vector<Value> xs;
  detail::Tally samples("samples valid");
  for (std::size_t i = 0; i < o.trials; i++)
    guarded(samples, [&] {
      Value x = sp.sample(s);
      const auto why = sp.invalid(x);
      samples.trial(!why, why ? 1 : 0, [&] { return words(x) + ": " + *why; });
      if (!why) xs.push_back(std::move(x));
    });
  report.laws.push_back(samples.law);
  if (xs.size() < 3) return report;
  auto any = [&]() -> const Value& { return xs[s.below(xs.size())]; };

  // ---- the metric ----
  detail::Tally identity("metric: d(x, x) = 0"), symmetry("metric: symmetric and non-negative"), triangle("metric: triangle inequality"),
      within("distances within the diameter");
  const double diam = sp.diameter();
  for (std::size_t i = 0; i < o.trials; i++) {
    const Value &x = any(), &y = any(), &z = any();
    guarded(identity, [&] {
      const double d = sp.distance(x, x);
      identity.trial(std::abs(d) <= tol_d, std::abs(d), [&] { return words(x); });
    });
    guarded(symmetry, [&] {
      const double a = sp.distance(x, y), b = sp.distance(y, x);
      symmetry.trial(a >= 0 && std::abs(a - b) <= tol_d, std::abs(a - b) + (a < 0 ? -a : 0), [&] { return words(x) + ", " + words(y); });
    });
    guarded(triangle, [&] {
      const double xz = sp.distance(x, z), xy = sp.distance(x, y), yz = sp.distance(y, z);
      const double excess = xz - xy - yz;
      triangle.trial(excess <= tol_d, std::max(0.0, excess), [&] { return words(x) + ", " + words(y) + ", " + words(z); });
    });
    guarded(within, [&] {
      const double d = sp.distance(x, y);
      within.trial(d <= diam * (1 + o.tolerance) + tol_d, std::max(0.0, d - diam), [&] { return words(x) + ", " + words(y); });
    });
  }
  for (auto* t : {&identity, &symmetry, &triangle, &within}) report.laws.push_back(t->law);

  // ---- mutation ----
  detail::Tally mvalid("mutations valid"), mball("mutations within their ball", sp.mutation_radius(1).has_value()), mmove("mutations move as often as stated");
  std::size_t moved = 0, tried = 0;
  for (double scale : {0.01, o.scale, 1.0}) {
    const auto radius = sp.mutation_radius(scale);
    for (std::size_t i = 0; i < o.trials; i++) {
      const Value& x = any();
      guarded(mvalid, [&] {
        const Value y = sp.mutate(x, scale, s);
        const auto why = sp.invalid(y);
        mvalid.trial(!why, why ? 1 : 0, [&] { return words(y) + ": " + *why; });
        if (why) return;
        tried++, moved += !(y == x);
        if (radius) {
          const double d = sp.distance(x, y);
          mball.trial(d <= *radius + tol_d, std::max(0.0, d - *radius), [&] { return "d = " + std::to_string(d) + " beyond " + std::to_string(*radius) + " at scale " + std::to_string(scale); });
        }
      });
    }
  }
  {
    const double floor = sp.change_floor();
    const double share = tried ? static_cast<double>(moved) / static_cast<double>(tried) : 0;
    const double se = std::sqrt(std::max(floor * (1 - floor), 1e-12) / std::max<double>(1, static_cast<double>(tried)));
    const bool ok = floor >= 1 ? moved == tried : share + 4 * se >= floor;
    mmove.law.applies = floor > 0;
    mmove.trial(ok, std::max(0.0, floor - share), [&] { return "moved " + std::to_string(moved) + " of " + std::to_string(tried) + " against a floor of " + std::to_string(floor); });
  }
  for (auto* t : {&mvalid, &mball, &mmove}) report.laws.push_back(t->law);

  // ---- recombination ----
  const bool rec = sp.can_recombine();
  detail::Tally cvalid("children valid", rec), pure("recombination pure", rec && props.pure), segment("children on the segment", rec && props.geometric);
  if (rec) {
    for (std::size_t i = 0; i < 2 * o.trials; i++) {
      const Value& x = any();
      Value y;
      try {
        y = i % 2 ? any() : sp.mutate(x, o.scale, s);  // random pairs and near pairs
      } catch (const std::exception&) {
        continue;
      }
      guarded(cvalid, [&] {
        const Value c = sp.recombine(x, y, s);
        const auto why = sp.invalid(c);
        cvalid.trial(!why, why ? 1 : 0, [&] { return words(c) + ": " + *why; });
        if (why || !segment.law.applies) return;
        const double dxy = sp.distance(x, y), dxc = sp.distance(x, c), dcy = sp.distance(c, y);
        const double dev = std::abs(dxc + dcy - dxy);
        segment.trial(dev <= o.tolerance * (1 + dxy) * 10 + tol_d, dev, [&] { return words(x) + " × " + words(y) + " → " + words(c); });
      });
      if (pure.law.applies && i < o.trials)
        guarded(pure, [&] {
          const Value c = sp.recombine(x, x, s);
          const double d = sp.distance(x, c);
          pure.trial(d <= tol_d, d, [&] { return words(x) + " → " + words(c); });
        });
    }
  }
  for (auto* t : {&cvalid, &pure, &segment}) report.laws.push_back(t->law);

  // ---- repair ----
  const bool rep = sp.can_repair();
  detail::Tally rid("repair the identity on valid values", rep), ridem("repair valid and idempotent", rep);
  if (rep) {
    for (std::size_t i = 0; i < o.trials; i++) {
      const Value& x = any();
      guarded(rid, [&] {
        const Value r = sp.repair(x);
        rid.trial(r == x, r == x ? 0 : 1, [&] { return words(x) + " → " + words(r); });
      });
      const Value bad = detail::perturb(x, s);
      if (sp.valid(bad)) continue;
      try {
        const Value r = sp.repair(bad);
        const auto why = sp.invalid(r);
        const Value r2 = why ? r : sp.repair(r);
        ridem.trial(!why && r2 == r, why ? 1 : (r2 == r ? 0 : 1), [&] { return words(bad) + " → " + words(r) + (why ? ": " + *why : " → " + words(r2)); });
      } catch (const Refusal&) {
        // a value of another shape than the space repairs: not a failure of the law
      } catch (const std::exception& e) {
        ridem.trial(false, std::numeric_limits<double>::infinity(), [&] { return std::string("threw: ") + e.what(); });
      }
    }
  }
  for (auto* t : {&rid, &ridem}) report.laws.push_back(t->law);

  // ---- the chart ----
  const bool chart = sp.has_chart();
  detail::Tally zero("chart: retract_x(0) = x", chart), logret("chart: log_x(retract_x(v)) = v within the trust radius", chart),
      retlog("chart: retract_x(log_x(y)) = y", chart);
  if (chart) {
    const std::size_t k = sp.chart_dimension();
    for (std::size_t i = 0; i < o.trials; i++) {
      const Value& x = any();
      guarded(zero, [&] {
        const std::vector<double> v0(k, 0);
        const double d = sp.distance(sp.retract(x, v0), x);
        zero.trial(d <= o.chart_tolerance * (1 + unit), d, [&] { return words(x); });
      });
      guarded(logret, [&] {
        const double trust = sp.trust_radius(x);
        if (!(trust > 0)) return;
        std::vector<double> v(k);
        for (double& c : v) c = s.normal();
        const double r = s.uniform() * std::min(0.9 * trust, unit);
        const double n = detail::norm(v);
        for (double& c : v) c *= r / n;
        const auto back = sp.log(x, sp.retract(x, v));
        double err = 0;
        for (std::size_t j = 0; j < k; j++) err += (back[j] - v[j]) * (back[j] - v[j]);
        err = std::sqrt(err);
        logret.trial(err <= o.chart_tolerance * (1 + r), err, [&] { return words(x) + " with a step of length " + std::to_string(r); });
      });
      guarded(retlog, [&] {
        const Value y = sp.mutate(x, o.scale, s);
        const auto v = sp.log(x, y);
        if (detail::norm(v) >= sp.trust_radius(x)) return;
        const double d = sp.distance(sp.retract(x, v), y);
        retlog.trial(d <= o.chart_tolerance * (1 + unit), d, [&] { return words(x) + " to " + words(y); });
      });
    }
  }
  for (auto* t : {&zero, &logret, &retlog}) report.laws.push_back(t->law);

  // ---- the embedding ----
  const auto edim = sp.embedding_dimension();
  detail::Tally dec("embedding: unembed(embed(x)) = x", edim.has_value()), bij("embedding: embed(unembed(z)) = z", edim.has_value() && sp.embedding_bijective());
  if (edim) {
    for (std::size_t i = 0; i < o.trials; i++) {
      const Value& x = any();
      guarded(dec, [&] {
        const auto z = sp.embed(x);
        if (z.size() != *edim) {
          dec.trial(false, 1, [&] { return std::string("an embedding of another dimension"); });
          return;
        }
        const Value back = sp.unembed(z);
        const double d = sp.distance(back, x);
        dec.trial(d <= o.chart_tolerance * (1 + unit), d, [&] { return words(x) + " → " + words(back); });
      });
      if (bij.law.applies)
        guarded(bij, [&] {
          std::vector<double> z(*edim);
          for (double& c : z) c = s.normal();
          const auto again = sp.embed(sp.unembed(z));
          double err = 0;
          for (std::size_t j = 0; j < z.size(); j++) err += (again[j] - z[j]) * (again[j] - z[j]);
          err = std::sqrt(err);
          bij.trial(err <= o.chart_tolerance * (1 + detail::norm(z)), err, [&] { return std::string("a random point of the embedding"); });
        });
    }
  }
  for (auto* t : {&dec, &bij}) report.laws.push_back(t->law);

  // ---- the unit ----
  {
    detail::Tally u(props.unit_exact ? "the unit, exact" : "the unit, an estimate", unit > 0 && props.unit != Unit::diameter);
    if (u.law.applies) {
      double sum = 0, sum2 = 0, sum4 = 0;
      std::size_t n = 0;
      for (std::size_t i = 0; i < o.unit_pairs; i++) {
        try {
          const double d = sp.distance(sp.sample(s), sp.sample(s));
          sum += d, sum2 += d * d, sum4 += d * d * d * d, n++;
        } catch (const std::exception&) {
        }
      }
      const double N = static_cast<double>(n);
      double estimate, se;
      if (props.unit == Unit::mean) {
        estimate = sum / N;
        se = std::sqrt(std::max(0.0, sum2 / N - estimate * estimate) / N);
      } else {
        const double m2 = sum2 / N;
        estimate = std::sqrt(m2);
        se = std::sqrt(std::max(0.0, sum4 / N - m2 * m2) / N) / (2 * std::max(estimate, 1e-300));
      }
      const double dev = std::abs(estimate - unit);
      const double limit = props.unit_exact ? o.unit_sigmas * se + 1e-12 * unit : o.estimate_tolerance * unit + o.unit_sigmas * se;
      u.trial(dev <= limit, dev / unit, [&] { return "measured " + std::to_string(estimate) + " ± " + std::to_string(se) + " against " + std::to_string(unit); });
    }
    report.laws.push_back(u.law);
  }

  // ---- finite spaces ----
  {
    const auto c = sp.cardinality();
    detail::Tally el("elements: each valid, each once, as many as counted", c.has_value() && *c <= o.elements_limit);
    if (el.law.applies)
      guarded(el, [&] {
        const auto all = sp.elements(o.elements_limit);
        std::set<std::string> keys;
        std::size_t bad = 0;
        for (const auto& v : all) bad += !sp.valid(v), keys.insert(key(v));
        const bool ok = all.size() == *c && keys.size() == all.size() && bad == 0;
        el.trial(ok, ok ? 0 : 1, [&] { return std::to_string(all.size()) + " listed, " + std::to_string(keys.size()) + " distinct, " + std::to_string(bad) + " invalid, against " + std::to_string(*c); });
      });
    report.laws.push_back(el.law);
  }

  // ---- keys ----
  {
    detail::Tally k("keys identify values");
    for (std::size_t i = 0; i < o.trials; i++) {
      const Value &x = any(), &y = any();
      const bool ok = (key(x) == key(y)) == (x == y) && key(x) == key(Value(x));
      k.trial(ok, ok ? 0 : 1, [&] { return words(x) + ", " + words(y); });
    }
    report.laws.push_back(k.law);
  }
  return report;
}

}  // namespace heuresis::spaces
