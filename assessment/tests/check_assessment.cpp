// Checks of heuresis/assessment: evaluators of many kinds, where what is to be found is known.
//
// - Assessments merge with the worst status, refuse two metrics of one name, and keep each metric no stronger than its evaluator.
// - Tiers: a proposed or estimated objective never selects and never outranks what a stronger one decides, on 300 random sets;
//   failures last, the infeasible after the feasible; an estimated constraint does not bind.
// - Replicates for a difference equal the closed form, and reach their power in simulation.
// - Races select the least mean with error rates within δ on 400 noisy pairs, under Hoeffding's and the empirical Bernstein
//   bounds; Bernstein's spends fewer samples where the variance is small.
// - Common random numbers kept where they lower the variance of differences, and refused where they raise it.
// - Cascades select as an exhaustive evaluation at the last fidelity does, on 200 instances, for less.
// - The cache returns recorded assessments exactly and runs a new version afresh.
// - A surrogate is trusted when it agrees and distrusted with its reason when it does not; coverage counted exactly.
// - Bradley–Terry recovers known strengths and an order effect from judgements, its likelihood rising at every step; judgements
//   that leave a group never preferred are refused unless a prior is declared.
// - An evolution assessed by a computed objective, a noisy estimate and an adversarial proposal finds the computed optimum; the
//   winner re-assessed on fresh replicates carries its uncertainties.

#include <heuresis/assessment/assessment.hpp>

#include <heuresis/evolution/evolution.hpp>
#include <heuresis/spaces/adapters.hpp>
#include <heuresis/spaces/base.hpp>

#include <calx/verification/check.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>
#include <vector>

namespace ha = heuresis::assessment;
namespace hs = heuresis::spaces;
namespace he = heuresis::evolution;

namespace {
calx::random::Stream stream(std::uint32_t id) { return calx::random::Stream(20261120, 1200 + id, 0, 0); }

ha::Metric metric(std::string name, ha::Role role, double v, ha::Standing st) {
  ha::Metric m;
  m.name = std::move(name), m.role = role, m.value = v, m.standing = st;
  return m;
}
}  // namespace

int main() {
  calx::verification::Suite s;

  // ---- records ----
  {
    ha::Assessment a, b;
    a.metrics = {metric("cost", ha::Role::objective, 3, ha::Standing::computed)};
    b.metrics = {metric("liking", ha::Role::descriptor, 0.9, ha::Standing::proposed)};
    b.status = ha::Status::abstained, b.reason = "the model abstained";
    const auto m = ha::merge({a, b});
    s.require("merged assessments keep the worst status with its reason", m.status == ha::Status::abstained && m.reason == "the model abstained" && m.metrics.size() == 2);
    bool refused = false;
    try {
      ha::merge({a, a});
    } catch (const ha::Refusal&) {
      refused = true;
    }
    s.require("two metrics of one name are refused", refused);
  }

  // ---- tiers ----
  {
    ha::Assessment v1, v2;
    v1.metrics = {metric("welfare loss", ha::Role::objective, 1.0, ha::Standing::verified), metric("a model's dislike", ha::Role::objective, 0.0, ha::Standing::proposed)};
    v2.metrics = {metric("welfare loss", ha::Role::objective, 0.5, ha::Standing::verified), metric("a model's dislike", ha::Role::objective, 10.0, ha::Standing::proposed)};
    const auto order = ha::tiered_ranking({v1, v2});
    s.require("a proposed metric does not outrank a verified one", order[0] == 1);
    const auto e = ha::to_evaluation(v1);
    s.require("a proposed objective travels as a descriptor, not an objective", e.objectives.size() == 1 && e.descriptors.size() == 1 && e.descriptors[0] == 0.0);
    ha::Assessment c;
    c.metrics = {metric("loss", ha::Role::objective, 1, ha::Standing::computed), metric("over budget (estimated)", ha::Role::constraint, 5, ha::Standing::estimated)};
    s.require("an estimated constraint does not bind", ha::to_evaluation(c).constraints.empty() && ha::tiered(c).violation == 0);

    auto st = stream(1);
    bool respected = true, failures_last = true, infeasible_after = true;
    for (int t = 0; t < 300; t++) {
      std::vector<ha::Assessment> as;
      const std::size_t n = 8;
      for (std::size_t i = 0; i < n; i++) {
        ha::Assessment a;
        a.metrics = {metric("f1", ha::Role::objective, std::floor(st.uniform(0, 4)), ha::Standing::verified),
                     metric("f2", ha::Role::objective, std::floor(st.uniform(0, 4)), ha::Standing::computed),
                     metric("guess", ha::Role::objective, st.uniform(-100, 100), st.uniform() < 0.5 ? ha::Standing::estimated : ha::Standing::proposed),
                     metric("budget", ha::Role::constraint, st.uniform() < 0.2 ? st.uniform(0.1, 2) : -1, ha::Standing::computed)};
        if (st.uniform() < 0.1) a.status = ha::Status::failed, a.reason = "planted";
        as.push_back(a);
      }
      const auto order = ha::tiered_ranking(as);
      std::vector<std::size_t> place(n);
      for (std::size_t k = 0; k < n; k++) place[order[k]] = k;
      for (std::size_t i = 0; i < n; i++)
        for (std::size_t j = 0; j < n; j++) {
          const auto ti = ha::tiered(as[i]), tj = ha::tiered(as[j]);
          if (!ti.failed && tj.failed) failures_last = failures_last && place[i] < place[j];
          if (!ti.failed && !tj.failed && ti.violation == 0 && tj.violation > 0) infeasible_after = infeasible_after && place[i] < place[j];
          if (!ti.failed && !tj.failed && ti.violation == 0 && tj.violation == 0 && heuresis::multiobjective::dominates(ti.select, tj.select))
            respected = respected && place[i] < place[j];
        }
    }
    s.require("on 300 random sets, what stronger objectives decide no weaker one reverses", respected);
    s.require("failures rank last", failures_last);
    s.require("the infeasible rank after the feasible", infeasible_after);
  }

  // ---- replicates for a difference ----
  {
    const std::size_t n = ha::replicates_for(1.0, 0.5);
    const double z = 1.959963984540054 + 0.8416212335729143;
    s.require("replicates for a difference of 0.5σ at 5% with power 0.8: " + std::to_string(n), n == static_cast<std::size_t>(std::ceil(2 * z * z / 0.25)));
    auto st = stream(2);
    int rejected = 0;
    const int trials = 4000;
    for (int t = 0; t < trials; t++) {
      double a = 0, b = 0;
      for (std::size_t i = 0; i < n; i++) a += st.normal(), b += st.normal(0.5, 1);
      const double diff = (b - a) / static_cast<double>(n);
      rejected += std::abs(diff) / std::sqrt(2.0 / static_cast<double>(n)) > 1.959963984540054;
    }
    s.check("and in simulation they reach that power", static_cast<double>(rejected) / trials - 0.8, 0.025);
  }

  // ---- races ----
  {
    auto st = stream(3);
    for (auto bound : {ha::Bound::hoeffding, ha::Bound::bernstein}) {
      const char* name = bound == ha::Bound::hoeffding ? "Hoeffding" : "empirical Bernstein";
      int wrong = 0, decided = 0;
      for (int t = 0; t < 400; t++) {
        const std::uint64_t key = 7000 + static_cast<std::uint64_t>(t);
        auto draw = [&](std::size_t i, std::size_t r) {
          calx::random::Stream x(key, static_cast<std::uint32_t>(i), static_cast<std::uint32_t>(r), 0);
          return x.uniform() < (i == 0 ? 0.6 : 0.5) ? 1.0 : 0.0;  // candidate 1 has the least mean
        };
        const auto r = ha::race(2, draw, 0, 1, 0.1, 4000, bound);
        if (r.decided) decided++, wrong += r.best != 1;
      }
      s.require(std::string(name) + " races: wrong selections within δ = 0.1 (" + std::to_string(wrong) + " of " + std::to_string(decided) + " decided)",
                decided > 0 && static_cast<double>(wrong) <= 0.1 * decided);
    }
    // Low variance: values 0.50 ± 0.02 against 0.53 ± 0.02 in a declared range [0, 1].
    std::size_t hoeffding = 0, bernstein = 0;
    for (auto bound : {ha::Bound::hoeffding, ha::Bound::bernstein}) {
      auto draw = [&](std::size_t i, std::size_t r) {
        calx::random::Stream x(99, static_cast<std::uint32_t>(i), static_cast<std::uint32_t>(r), 0);
        return (i == 0 ? 0.53 : 0.50) + x.uniform(-0.02, 0.02);
      };
      const auto r = ha::race(2, draw, 0, 1, 0.05, 100000, bound);
      (bound == ha::Bound::hoeffding ? hoeffding : bernstein) = r.total;
      (void)st;
    }
    s.require("where the variance is small, the empirical Bernstein race spends fewer samples (" + std::to_string(bernstein) + " against " + std::to_string(hoeffding) + ")", bernstein < hoeffding);
  }

  // ---- common random numbers ----
  {
    auto a = [](std::uint64_t k) {
      calx::random::Stream x(5, static_cast<std::uint32_t>(k), 0, 0);
      return std::exp(x.normal());
    };
    auto b = [&](std::uint64_t k) { return a(k) + 0.1; };
    auto c = [&](std::uint64_t k) { return -a(k); };
    const auto good = ha::common_numbers(a, b, 500), bad = ha::common_numbers(a, c, 500);
    s.require("common random numbers are kept where they lower the variance of a difference", good.keep() && good.common < 1e-20);
    s.require("and refused where they raise it", !bad.keep());
  }

  // ---- cascades ----
  {
    auto st = stream(4);
    bool same = true;
    double spent = 0, full = 0;
    for (int t = 0; t < 200; t++) {
      const std::size_t k = 40;
      std::vector<double> exact(k), mid(k), cheap(k);
      const double eps = 0.5;
      for (std::size_t i = 0; i < k; i++) {
        exact[i] = st.uniform(0, 10);
        mid[i] = exact[i] + st.uniform(-eps / 4, eps / 4);
        cheap[i] = exact[i] + st.uniform(-eps, eps);
      }
      const auto c = ha::cascade(k, {{"cheap", 1, eps, false, [&](std::size_t i) { return cheap[i]; }},
                                     {"middle", 10, eps / 4, false, [&](std::size_t i) { return mid[i]; }},
                                     {"exact", 100, 0, false, [&](std::size_t i) { return exact[i]; }}});
      same = same && c.best == static_cast<std::size_t>(std::min_element(exact.begin(), exact.end()) - exact.begin());
      spent += c.cost, full += 100.0 * k;
    }
    s.require("a cascade selects as exhaustive evaluation at the last fidelity does, on 200 instances", same);
    s.require("for less (" + std::to_string(static_cast<int>(100 * spent / full)) + "% of the cost)", spent < full);
  }

  // ---- the cache, and the assessor ----
  {
    int calls = 0;
    ha::Evaluator exact{"exact", "1", ha::Kind::closed_form, ha::Standing::computed, 1, 1, false, [&](const hs::Value& x, calx::random::Stream&) {
                          calls++;
                          ha::Assessment a;
                          a.metrics = {metric("loss", ha::Role::objective, static_cast<double>(x.ints[0] * x.ints[0]), ha::Standing::verified)};
                          return a;
                        }};
    ha::Assessor assessor({exact});
    const auto x = hs::integer_value({3});
    const auto a1 = assessor.assess(x), a2 = assessor.assess(x);
    s.require("an assessment asked again is the one recorded, without running again", calls == 1 && a1.metrics[0].value == a2.metrics[0].value && a1.metrics.size() == a2.metrics.size());
    s.require("a metric is kept no stronger than its evaluator (verified claimed, computed declared)", a1.metrics[0].standing == ha::Standing::computed);
    exact.version = "2";
    ha::Assessor next({exact});
    next.assess(x);
    s.require("a new version runs afresh", calls == 2);
  }

  // ---- surrogates ----
  {
    auto st = stream(5);
    ha::Agreement good, bad;
    for (int i = 0; i < 40; i++) {
      const double y = st.uniform(0, 10);
      good.add(y + st.normal(0, 0.1), y, 0.2);
      bad.add(st.uniform(0, 10), y, 0.2);
    }
    s.require("a surrogate that agrees is trusted", !good.distrust().has_value(), good.distrust().value_or(""));
    s.require("one that does not is distrusted, with its reason", bad.distrust().has_value(), bad.distrust().value_or(""));
    ha::Agreement c;
    for (int i = 0; i < 10; i++) c.add(0, i < 8 ? 0.5 : 5, 1);
    s.check("coverage is counted exactly: 8 of 10 within their intervals", *c.coverage() - 0.8, 1e-15);
  }

  // ---- Bradley–Terry ----
  {
    auto st = stream(6);
    const std::vector<double> truth = {-1.0, -0.4, 0.0, 0.3, 0.5, 0.6};  // centred log-strengths
    const double theta = 1.5;
    std::vector<ha::Judgement> js;
    for (int t = 0; t < 8000; t++) {
      const std::size_t i = st.below(6);
      std::size_t j = st.below(5);
      if (j >= i) j++;
      const double a = theta * std::exp(truth[i]), b = std::exp(truth[j]);
      js.push_back({i, j, st.uniform() < a / (a + b)});
    }
    const auto bt = ha::bradley_terry(6, js);
    double err = 0;
    for (std::size_t i = 0; i < 6; i++) err = std::max(err, std::abs(bt.log_strength[i] - truth[i]));
    s.check("Bradley–Terry recovers known log-strengths from 8000 judgements", err, 0.1);
    s.check("and the advantage of being shown first", bt.order_effect - theta, 0.1);
    s.require("its likelihood rises at every step, and it converges", bt.likelihood_rose && bt.converged);
    std::vector<ha::Judgement> lopsided = {{0, 1, true}, {1, 2, true}, {0, 2, true}};
    bool refused = false;
    try {
      ha::bradley_terry(3, lopsided);
    } catch (const ha::Refusal&) {
      refused = true;
    }
    const auto with_prior = ha::bradley_terry(3, lopsided, false, 1.0);
    s.require("judgements that leave an item never preferred are refused, and a declared prior answers them",
              refused && with_prior.converged && with_prior.strength[0] > with_prior.strength[1] && with_prior.strength[1] > with_prior.strength[2]);
  }

  // ---- an evolution assessed by evaluators of three kinds ----
  {
    auto genotype = std::make_shared<hs::IntegerBox>(std::vector<std::int64_t>{0, 0, 0}, std::vector<std::int64_t>{20, 20, 20});
    const std::vector<std::int64_t> target = {7, 13, 2};
    auto loss = [target](const hs::Value& x) {
      double d = 0;
      for (std::size_t i = 0; i < 3; i++) d += static_cast<double>((x.ints[i] - target[i]) * (x.ints[i] - target[i]));
      return d;
    };
    ha::Evaluator closed{"loss", "1", ha::Kind::closed_form, ha::Standing::computed, 1, 1, false, [&](const hs::Value& x, calx::random::Stream&) {
                           ha::Assessment a;
                           a.metrics = {metric("loss", ha::Role::objective, loss(x), ha::Standing::computed)};
                           return a;
                         }};
    ha::Evaluator noisy{"simulated loss", "1", ha::Kind::simulation, ha::Standing::estimated, 0.5, 5, true, [&](const hs::Value& x, calx::random::Stream& r) {
                          ha::Assessment a;
                          a.metrics = {metric("simulated loss", ha::Role::objective, loss(x) + r.normal(0, 3), ha::Standing::estimated)};
                          return a;
                        }};
    ha::Evaluator model{"a model's opinion", "1", ha::Kind::language_model, ha::Standing::proposed, 1, 1, false, [&](const hs::Value& x, calx::random::Stream&) {
                          ha::Assessment a;  // adversarial: it prefers what is far from the target, and claims to be verified
                          a.metrics = {metric("a model's opinion", ha::Role::objective, -loss(x), ha::Standing::verified)};
                          return a;
                        }};
    ha::Assessor assessor({closed, noisy, model});
    he::Strategy strategy;
    strategy.mu = 16, strategy.lambda = 16, strategy.generations = 60, strategy.from_archive = 0;
    he::Evolution<hs::Value> e(hs::operators(genotype), strategy, {}, 3);
    he::evolve(e, assessor.objective());
    const auto best = e.state().best->genotype;
    s.require("an evolution assessed by a computed loss, a noisy estimate and an adversarial proposal finds the computed optimum",
              best.ints == target, hs::words(best));
    const auto again = assessor.reassess(best, 20);
    const auto* sim = again.find("simulated loss");
    const auto* exact_loss = again.find("loss");
    const auto* opinion = again.find("a model's opinion");
    s.require("re-assessed on 20 fresh replicates, the estimate carries its error and the computed loss none",
              sim && sim->replicates == 20 && sim->uncertainty && *sim->uncertainty > 0 && exact_loss && exact_loss->uncertainty && *exact_loss->uncertainty == 0);
    s.require("the model's opinion is kept as proposed though it claimed verification", opinion && opinion->standing == ha::Standing::proposed);
  }

  return s.done();
}
