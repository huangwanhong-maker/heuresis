// Checks of heuresis/evolution: an evolutionary loop assembled from strategies, on bit strings, where what is to be found is known.
//
// - Every survival strategy (generational, plus, comma, replace-worst, aging, deterministic crowding, restricted tournament, clearing,
//   NSGA-II, ALPS, niches) and every parent strategy (uniform, tournament, roulette, ranking, Boltzmann, truncation, lexicase,
//   ε-lexicase, novelty) reaches OneMax's optimum.
// - Adaptive schedules (probability matching, adaptive pursuit, UCB1, Thompson sampling) give more children to a mutation that
//   improves than to one that scrambles.
// - Aging keeps no member longer than μ/λ generations.
// - The archive admits what its judge admits, retires what has gone long without a surviving child, what its recheck no longer
//   upholds, and what overflows it, each with its reason.
// - A proposal from outside is asked with the next generation, credited to its source, and survives when it is best.
// - A run does not depend on the order in which its evaluations are told; a run resumed from its state is the run uninterrupted;
//   the same seed gives the same run.
// - Lexicase reaches the optimum of four objectives; NSGA-II spreads along the front of LOTZ; novelty keeps a population more
//   diverse; stagnation stops a run or restarts it; every parent was told before its child; what is refused is refused.

#include <heuresis/evolution/evolution.hpp>

#include <calx/verification/check.hpp>

#include <algorithm>
#include <set>
#include <string>
#include <vector>

namespace he = heuresis::evolution;
using heuresis::search::Evaluation;

namespace {

constexpr std::size_t N = 24;

std::size_t ones(const std::string& g) { return static_cast<std::size_t>(std::count(g.begin(), g.end(), '1')); }

he::Operators<std::string> bits(std::size_t n = N, bool scramble = false) {
  he::Operators<std::string> ops;
  ops.makers.push_back({"random", [n](std::size_t k, const he::Context<std::string>&, calx::random::Stream& s) {
                          std::vector<std::string> out;
                          for (std::size_t i = 0; i < k; i++) {
                            std::string g(n, '0');
                            for (auto& c : g) c = s.uniform() < 0.5 ? '1' : '0';
                            out.push_back(g);
                          }
                          return out;
                        }, 1.0});
  ops.mutations.push_back({"flip", [n](const he::Member<std::string>& p, const he::Context<std::string>&, calx::random::Stream& s) -> std::optional<std::string> {
                             std::string g = p.genotype;
                             auto& c = g[s.below(n)];
                             c = c == '1' ? '0' : '1';
                             return g;
                           }, 1.0});
  if (scramble)
    ops.mutations.push_back({"scramble", [n](const he::Member<std::string>& p, const he::Context<std::string>&, calx::random::Stream& s) -> std::optional<std::string> {
                               std::string g = p.genotype;
                               for (int k = 0; k < 12; k++) g[s.below(n)] = s.uniform() < 0.5 ? '1' : '0';
                               return g;
                             }, 1.0});
  ops.crossovers.push_back({"uniform", [n](const he::Member<std::string>& a, const he::Member<std::string>& b, const he::Context<std::string>&, calx::random::Stream& s) -> std::optional<std::string> {
                              std::string g = a.genotype;
                              for (std::size_t i = 0; i < n; i++)
                                if (s.uniform() < 0.5) g[i] = b.genotype[i];
                              return g;
                            }, 1.0});
  ops.distance = [](const std::string& a, const std::string& b) {
    std::size_t d = 0;
    for (std::size_t i = 0; i < a.size(); i++) d += a[i] != b[i];
    return static_cast<double>(d) / static_cast<double>(a.size());
  };
  ops.key = [](const std::string& g) { return g; };
  return ops;
}

Evaluation onemax(const std::string& g) {
  Evaluation e = heuresis::search::scalar(-static_cast<double>(ones(g)));
  std::size_t first = 0;
  for (std::size_t i = 0; i < g.size() / 2; i++) first += g[i] == '1';
  e.descriptors = {static_cast<double>(first), static_cast<double>(ones(g) - first)};
  return e;
}

double best_ones(const he::Evolution<std::string>& e) {
  double b = 0;
  for (const auto& m : e.population()) b = std::max(b, static_cast<double>(ones(m.genotype)));
  for (const auto& k : e.archive()) b = std::max(b, static_cast<double>(ones(k.member.genotype)));
  if (e.state().best) b = std::max(b, static_cast<double>(ones(e.state().best->genotype)));  // the best ever found
  return b;
}

he::Strategy base() {
  he::Strategy st;
  st.mu = 20, st.lambda = 20, st.generations = 80, st.from_archive = 0;
  return st;
}

}  // namespace

int main() {
  calx::verification::Suite s;

  // ---- every survival strategy ----
  for (auto sv : {he::Survival::generational, he::Survival::plus, he::Survival::comma, he::Survival::replace_worst, he::Survival::aging,
                  he::Survival::deterministic_crowding, he::Survival::restricted_tournament, he::Survival::clearing, he::Survival::nsga2,
                  he::Survival::alps, he::Survival::niches}) {
    he::Strategy st = base();
    st.survival = sv;
    st.radius = 0.1;
    if (sv == he::Survival::deterministic_crowding) st.generations = 200;
    if (sv == he::Survival::comma) st.lambda = 40;                         // (μ,λ) wants λ > μ
    if (sv == he::Survival::aging) st.lambda = 5, st.generations = 400;    // regularised evolution replaces a few at a time
    if (sv == he::Survival::niches) st.bins = {13, 13}, st.lower = {0, 0}, st.upper = {13, 13}, st.parents = he::Parents::uniform, st.generations = 600;
    he::Evolution<std::string> e(bits(), st, {}, 11);
    he::evolve(e, onemax);
    s.check(std::string("survival by ") + he::name(sv) + " reaches OneMax's optimum", static_cast<double>(N) - best_ones(e), 0.0);
  }

  // ---- every parent strategy ----
  for (auto pa : {he::Parents::uniform, he::Parents::tournament, he::Parents::roulette, he::Parents::ranking, he::Parents::boltzmann,
                  he::Parents::truncation, he::Parents::lexicase, he::Parents::epsilon_lexicase, he::Parents::novelty}) {
    he::Strategy st = base();
    st.parents = pa;
    st.generations = 120;
    he::Evolution<std::string> e(bits(), st, {}, 13);
    he::evolve(e, onemax);
    s.check(std::string("parents by ") + he::name(pa) + " reach OneMax's optimum", static_cast<double>(N) - best_ones(e), 0.0);
  }

  // ---- adaptive schedules ----
  for (auto sc : {he::Schedule::probability_matching, he::Schedule::adaptive_pursuit, he::Schedule::ucb, he::Schedule::thompson}) {
    he::Strategy st = base();
    st.schedule = sc, st.crossover_rate = 0, st.generator_rate = 0, st.generations = 30, st.survival = he::Survival::plus, st.credit = he::Credit::survival;
    he::Evolution<std::string> e(bits(N, true), st, {}, 17);
    he::evolve(e, onemax);
    const auto names = e.operator_names();
    const auto& c = e.credits();
    std::size_t flip = 0, scramble = 0;
    for (std::size_t k = 0; k < names.size(); k++) {
      if (names[k] == "flip") flip = c.count[k];
      if (names[k] == "scramble") scramble = c.count[k];
    }
    s.require(std::string("by ") + he::name(sc) + ", the mutation that improves makes more children than the one that scrambles",
              flip > scramble, std::to_string(flip) + " against " + std::to_string(scramble));
  }

  // ---- aging ----
  {
    he::Strategy st = base();
    st.survival = he::Survival::aging, st.mu = 20, st.lambda = 10, st.generations = 25;
    he::Evolution<std::string> e(bits(), st, {}, 19);
    he::evolve(e, onemax);
    std::size_t oldest = 0;
    for (const auto& m : e.population()) oldest = std::max(oldest, e.generation() - m.born);
    s.require("aging keeps no member longer than μ/λ generations", oldest <= 2, std::to_string(oldest));
  }

  // ---- the archive ----
  {
    he::Strategy st = base();
    st.retire_after = 3, st.generations = 30;
    he::Judge<std::string> judge;
    judge.admit = [](const he::Member<std::string>& m, const he::Context<std::string>&) -> std::optional<std::string> {
      if (ones(m.genotype) >= 20) return std::string("twenty ones or more");
      return std::nullopt;
    };
    he::Evolution<std::string> e(bits(), st, judge, 23);
    he::evolve(e, onemax);
    bool idle = !e.retired().empty();
    for (const auto& r : e.retired()) idle = idle && r.reason.rfind("no child of it survived", 0) == 0;
    s.require("kept members that no parent chose are retired after their idle generations, with the reason", idle, std::to_string(e.retired().size()) + " retired");
    std::size_t lineage_admitted = 0;
    for (const auto& l : e.lineage()) lineage_admitted += l.admitted;
    s.require("the genealogy marks what was admitted", lineage_admitted == e.archive().size() + e.retired().size(), std::to_string(lineage_admitted));

    he::Strategy st2 = base();
    st2.retire_after = 1000, st2.generations = 30, st2.from_archive = 0.3;
    he::Judge<std::string> judge2 = judge;
    judge2.recheck = [](const he::Kept<std::string>& k, const he::Context<std::string>& ctx) -> std::optional<std::string> {
      if (ctx.generation >= 10 && ones(k.member.genotype) < 22) return std::string("fewer than 22 ones, now that better is known");
      return std::nullopt;
    };
    he::Evolution<std::string> e2(bits(), st2, judge2, 23);
    he::evolve(e2, onemax);
    bool rechecked = false;
    for (const auto& r : e2.retired()) rechecked = rechecked || r.reason.rfind("fewer than 22", 0) == 0;
    bool kept_ok = true;
    for (const auto& k : e2.archive())  // a member admitted in the last generation has not yet been rechecked
      if (k.admitted + 1 < e2.generation()) kept_ok = kept_ok && ones(k.member.genotype) >= 22;
    s.require("a recheck that no longer upholds a kept member retires it, and what stays is upheld", rechecked && kept_ok);

    he::Strategy st3 = base();
    st3.retire_after = 1000, st3.archive_capacity = 5, st3.generations = 30, st3.from_archive = 0.2;
    he::Evolution<std::string> e3(bits(), st3, judge, 23);
    he::evolve(e3, onemax);
    bool full = false;
    for (const auto& r : e3.retired()) full = full || r.reason.rfind("the archive was full", 0) == 0;
    s.require("an archive over its capacity retires the longest idle", e3.archive().size() <= 5 && full);
  }

  // ---- proposals ----
  {
    he::Strategy st = base();
    st.generations = 3;
    he::Evolution<std::string> e(bits(), st, {}, 29);
    e.step(onemax);
    e.propose({std::string(N, '1')}, "oracle");
    e.step(onemax);
    const auto& stats = e.statistics().back();
    bool asked = false;
    for (const auto& l : e.lineage()) asked = asked || (l.operation == "oracle" && l.survived);
    s.require("a proposal is asked with the next generation, credited to its source, and survives when best",
              stats.proposals.count("oracle") && stats.proposals.at("oracle") == 1 && stats.proposals_surviving.count("oracle") && asked && best_ones(e) == N);
  }

  // ---- order, state and seed ----
  {
    he::Strategy st = base();
    st.generations = 15, st.schedule = he::Schedule::adaptive_pursuit;
    auto run_forward = [&](bool reverse) {
      he::Evolution<std::string> e(bits(N, true), st, {}, 31);
      while (!e.stopped()) {
        auto cs = e.ask(1000);
        if (cs.empty()) break;
        std::vector<heuresis::search::Told> told;
        for (auto& c : cs) told.push_back({c.ticket, onemax(c.genotype)});
        if (reverse) std::reverse(told.begin(), told.end());
        e.tell(told);
      }
      std::vector<std::string> keys;
      for (const auto& m : e.population()) keys.push_back(m.genotype);
      return std::make_pair(keys, e.lineage().size());
    };
    const auto a = run_forward(false), b = run_forward(true);
    s.require("a run does not depend on the order in which its evaluations are told", a == b);
    const auto c = run_forward(false);
    s.require("the same seed gives the same run", a == c);

    he::Strategy st20 = st;
    st20.generations = 20;
    he::Evolution<std::string> whole(bits(N, true), st20, {}, 37);
    he::evolve(whole, onemax);
    he::Strategy st10 = st;
    st10.generations = 10;
    he::Evolution<std::string> half(bits(N, true), st10, {}, 37);
    he::evolve(half, onemax);
    he::Evolution<std::string> resumed(bits(N, true), st20, {}, half.state(), 37);
    he::evolve(resumed, onemax);
    std::vector<std::string> kw, kr;
    for (const auto& m : whole.population()) kw.push_back(m.genotype);
    for (const auto& m : resumed.population()) kr.push_back(m.genotype);
    s.require("a run resumed from its state after ten generations is the run of twenty uninterrupted", kw == kr && whole.lineage().size() == resumed.lineage().size(),
              std::to_string(whole.lineage().size()) + " and " + std::to_string(resumed.lineage().size()));
  }

  // ---- lexicase on four objectives ----
  {
    he::Strategy st = base();
    st.parents = he::Parents::lexicase, st.scoring = he::Scoring::lexicographic, st.generations = 100;
    he::Evolution<std::string> e(bits(), st, {}, 41);
    he::evolve(e, [](const std::string& g) {
      std::vector<double> zeros(4, 0.0);
      for (std::size_t i = 0; i < g.size(); i++) zeros[i / 6] += g[i] == '0';
      return heuresis::search::evaluation(zeros);
    });
    s.check("lexicase reaches the optimum of four objectives, one a block", static_cast<double>(N) - best_ones(e), 0.0);
  }

  // ---- NSGA-II along the front of LOTZ ----
  {
    he::Strategy st = base();
    st.scoring = he::Scoring::pareto, st.survival = he::Survival::nsga2, st.mu = 30, st.lambda = 30, st.generations = 150;
    he::Evolution<std::string> e(bits(12), st, {}, 43);
    he::evolve(e, [](const std::string& g) {
      std::size_t lo = 0, tz = 0;
      while (lo < g.size() && g[lo] == '1') lo++;
      while (tz < g.size() && g[g.size() - 1 - tz] == '0') tz++;
      return heuresis::search::evaluation({-static_cast<double>(lo), -static_cast<double>(tz)});
    });
    std::set<std::size_t> front;
    for (const auto& m : e.population()) {
      const double lo = -m.evaluation.objectives[0], tz = -m.evaluation.objectives[1];
      if (lo + tz == 12) front.insert(static_cast<std::size_t>(lo));
    }
    s.require("NSGA-II spreads along LOTZ's front, at least ten of its thirteen points", front.size() >= 10, std::to_string(front.size()));
  }

  // ---- novelty ----
  {
    auto diversity = [&](bool novelty) {
      he::Strategy st = base();
      st.generations = 40, st.novelty = novelty, st.scoring = novelty ? he::Scoring::pareto : he::Scoring::first;
      if (novelty) st.survival = he::Survival::nsga2;
      he::Evolution<std::string> e(bits(), st, {}, 47);
      he::evolve(e, onemax);
      return e.statistics().back().diversity;
    };
    const double with = diversity(true), without = diversity(false);
    s.require("novelty as a further objective keeps the population more diverse", with > without, std::to_string(with) + " against " + std::to_string(without));
  }

  // ---- stagnation ----
  {
    auto flat = [](const std::string&) { return heuresis::search::scalar(0.0); };
    he::Strategy st = base();
    st.stagnation = 3, st.generations = 50;
    he::Evolution<std::string> e(bits(), st, {}, 53);
    he::evolve(e, flat);
    s.require("a run that does not improve for three generations stops", e.stopped() && e.generation() <= 5, std::to_string(e.generation()));
    he::Strategy st2 = st;
    st2.restart_on_stagnation = true, st2.generations = 12;
    he::Evolution<std::string> e2(bits(), st2, {}, 53);
    he::evolve(e2, flat);
    bool restarted = false;
    for (const auto& g : e2.statistics()) restarted = restarted || g.restarted;
    s.require("with a restart instead, it is reseeded and runs on", restarted && e2.generation() == 12, std::to_string(e2.generation()));
  }

  // ---- the genealogy ----
  {
    he::Evolution<std::string> e(bits(), base(), {}, 59);
    he::evolve(e, onemax);
    bool before = true;
    for (const auto& l : e.lineage())
      for (auto p : l.parents) before = before && p < l.ticket;
    s.require("every parent was told before its child", before);
  }

  // ---- refusals ----
  {
    auto ops = bits();
    ops.distance = nullptr;
    he::Strategy st = base();
    st.survival = he::Survival::deterministic_crowding;
    bool refused = false;
    try { he::Evolution<std::string> e(ops, st); } catch (const std::invalid_argument&) { refused = true; }
    s.require("crowding without a distance is refused", refused);
    refused = false;
    try { (void)he::of<he::Survival>("tournament-of-champions"); } catch (const std::invalid_argument&) { refused = true; }
    s.require("a strategy that is not there is refused, with those that are", refused);
    he::Strategy bad = base();
    bad.survival = he::Survival::niches;
    refused = false;
    try { he::Evolution<std::string> e(bits(), bad); } catch (const std::invalid_argument&) { refused = true; }
    s.require("niches without their bins are refused", refused);
  }

  return s.done();
}
