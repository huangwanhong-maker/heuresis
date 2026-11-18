// An evolutionary loop over any genotype, assembled from a strategy at every stage, with an archive that keeps and retires, and a
// seam for proposals from outside (decision 0006).
//
// THE LOOP. A population of μ members is kept. Each generation λ children are made: for each, an operator is chosen (a generator,
// a mutation or a crossover, by the schedule), its parents are selected (by the parent strategy, from the population or, now and
// then, from the archive), and the child is made: tried again when it copies a living member or a sibling, and given its remembered
// evaluation when it repeats a genotype evaluated before, so that nothing is evaluated twice and a strategy that lost a good
// genotype can make it again. Proposals from outside are
// added to the generation as they are. The children are asked and told as in every heuresis algorithm (ask–tell), and then:
// novelty is reckoned if asked for; the survivors are chosen; the members age; the operators are credited with what their children
// did; the archive admits what its judge admits and retires what has gone long without use, what is too old, or what its recheck
// no longer upholds; the generation's statistics and every child's lineage are recorded; and stagnation stops the run or restarts it.
//
// SOURCES of the strategies.
//   parents: tournaments, ranking, Boltzmann, roulette, truncation and lexicase as heuresis::selection holds them (Goldberg and Deb
//     1991; Baker 1987; de la Maza and Tidor 1993; Spector 2012; La Cava, Spector and Danai 2016 for ε-lexicase); novelty
//     tournaments after Lehman and Stanley (2011).
//   survival: generational with elitism (De Jong 1975); (μ+λ) and (μ,λ) (Schwefel 1981); steady state (Whitley 1989); aging, which
//     removes the oldest whatever its fitness (regularized evolution: Real, Aggarwal, Huang and Le 2019, AAAI, 4780–4789);
//     deterministic crowding (Mahfoud 1995); restricted tournament selection (Harik 1995, ICGA, 24–31); clearing (Pétrowski 1996);
//     NSGA-II (Deb, Pratap, Agarwal and Meyarivan 2002); ALPS, age layers reseeded from below (Hornby 2006, GECCO, 815–822);
//     MAP-Elites niches (Mouret and Clune 2015).
//   ages follow ALPS: a child is as old as its oldest parent, a member made new is of age 0, and every survivor ages by one a generation.
//   operators are chosen as credit.hpp says.
//
// What is generic: the genotype G is any copyable value; the operators, the distance, the key and the archive's judge are the
// caller's. What is assumed: objectives are minimised (heuresis's convention); strategies that need a distance (crowding,
// restricted tournament, clearing, novelty) are refused without one.
#pragma once

#include <heuresis/evolution/credit.hpp>
#include <heuresis/evolution/strategies.hpp>
#include <heuresis/multiobjective/dominance.hpp>
#include <heuresis/multiobjective/indicators.hpp>
#include <heuresis/search/draws.hpp>
#include <heuresis/search/evaluation.hpp>
#include <heuresis/search/protocol.hpp>
#include <heuresis/selection/lexicase.hpp>
#include <heuresis/selection/parents.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace heuresis::evolution {

using search::Evaluation;
using search::Ticket;

template <class G>
struct Member {
  Ticket ticket = 0;
  G genotype{};
  Evaluation evaluation;           // as told
  std::vector<double> score;       // what selection reads: the objectives, and −novelty when novelty is on
  std::string operation;           // the operator or source that made it
  std::vector<Ticket> parents;
  std::size_t born = 0;            // the generation it was made in
  std::size_t age = 0;             // in ALPS's sense
  std::size_t layer = 0;           // its ALPS layer
  std::size_t last_useful = 0;     // the last generation a child of it survived (its birth, until then)
  double novelty = 0;
  bool restarted = false;
};

template <class G>
struct Kept {
  Member<G> member;
  std::size_t admitted = 0;
  std::string note;                // what the judge said
};

template <class G>
struct Retired {
  Member<G> member;
  std::size_t admitted = 0, retired = 0;
  std::string note, reason;
};

template <class G>
struct Context {
  const std::vector<Member<G>>& population;
  const std::vector<Kept<G>>& archive;
  std::size_t generation;
};

template <class G>
struct Operators {
  struct Maker {
    std::string name;
    std::function<std::vector<G>(std::size_t, const Context<G>&, calx::random::Stream&)> make;
    double weight = 1;
  };
  struct Mutation {
    std::string name;
    std::function<std::optional<G>(const Member<G>&, const Context<G>&, calx::random::Stream&)> apply;
    double weight = 1;
  };
  struct Crossover {
    std::string name;
    std::function<std::optional<G>(const Member<G>&, const Member<G>&, const Context<G>&, calx::random::Stream&)> apply;
    double weight = 1;
  };
  std::vector<Maker> makers;
  std::vector<Mutation> mutations;
  std::vector<Crossover> crossovers;
  std::function<double(const G&, const G&)> distance;   // for crowding, restricted tournaments, clearing and novelty
  std::function<std::string(const G&)> key;             // identity, so that a genotype is evaluated once
};

template <class G>
struct Judge {
  std::function<std::optional<std::string>(const Member<G>&, const Context<G>&)> admit;     // a note to keep it, nothing to pass
  std::function<std::optional<std::string>(const Kept<G>&, const Context<G>&)> recheck;     // a reason to retire it, nothing to keep
};

struct Statistics {
  std::size_t generation = 0, evaluations = 0;
  std::vector<double> best, median;                  // of each objective over the population, as told (minimised)
  double diversity = std::numeric_limits<double>::quiet_NaN();  // the mean distance between members (a sample of pairs)
  std::vector<std::size_t> births, survivals;        // by operator (makers, mutations, crossovers in order)
  std::vector<double> probability;                   // the schedule's probabilities after the generation
  std::map<std::string, std::size_t> proposals, proposals_surviving;  // by source
  std::size_t archive = 0, admitted = 0, retired = 0, failures = 0, repeats = 0, reused = 0;
  bool improved = false, restarted = false;
};

struct Lineage {
  Ticket ticket = 0;
  std::size_t generation = 0;
  std::string operation;
  std::vector<Ticket> parents;
  std::vector<double> objectives;
  bool failed = false, survived = false, admitted = false;
};

// Everything an Evolution needs to resume between generations: kept by a caller that runs it a few generations a time.
template <class G>
struct State {
  std::size_t generation = 0, evaluations = 0, serial = 0, stall = 0;
  std::vector<Member<G>> population;
  std::vector<Kept<G>> archive;
  std::vector<Retired<G>> retired;
  std::vector<double> quality, probability, successes, failures;
  std::vector<std::size_t> count;
  std::size_t total = 0;
  std::vector<std::pair<std::string, Evaluation>> cache;  // every genotype evaluated, by its key, with its evaluation
  std::vector<Statistics> statistics;
  std::vector<Lineage> lineage;
  std::optional<Member<G>> best;                     // the best member ever, by the scoring
  bool stopped = false, reseed = false;
};

template <class G>
class Evolution {
 public:
  using Genotype = G;

  Evolution(Operators<G> ops, Strategy st, Judge<G> judge = {}, std::uint64_t seed = 0, std::uint32_t island = 0)
      : ops_(std::move(ops)), st_(std::move(st)), judge_(std::move(judge)), draws_(seed, island) {
    st_.validate();
    if (ops_.makers.empty()) throw std::invalid_argument("an evolution needs at least one generator to begin with");
    const bool needs_distance = st_.survival == Survival::deterministic_crowding || st_.survival == Survival::restricted_tournament ||
                                st_.survival == Survival::clearing || st_.novelty || st_.parents == Parents::novelty;
    if (needs_distance && !ops_.distance) throw std::invalid_argument(std::string("the strategy ") + (st_.novelty ? "novelty" : name(st_.survival)) + " needs a distance between genotypes");
    credits_.reset(weights());
  }

  Evolution(Operators<G> ops, Strategy st, Judge<G> judge, State<G> state, std::uint64_t seed = 0, std::uint32_t island = 0)
      : Evolution(std::move(ops), std::move(st), std::move(judge), seed, island) {
    s_ = std::move(state);
    if (!s_.quality.empty() && s_.quality.size() == credits_.size()) {
      credits_.quality = s_.quality, credits_.probability = s_.probability, credits_.successes = s_.successes, credits_.failures = s_.failures;
      credits_.count = s_.count, credits_.total = s_.total;
    }
    for (const auto& [k, e] : s_.cache) cache_[k] = e;
  }

  // ---- the ask–tell protocol ----

  std::vector<search::Candidate<G>> ask(std::size_t limit) {
    if (stopped()) return {};
    if (batch_.empty()) build();
    if (!batch_.empty() && told_ == batch_.size()) {  // every child was remembered: nothing to ask
      finish();
      return {};
    }
    std::vector<search::Candidate<G>> out;
    for (std::size_t k = 0; k < batch_.size() && out.size() < limit; k++) {
      auto& b = batch_[k];
      if (b.asked) continue;
      b.asked = true;
      out.push_back({b.member.ticket, b.member.genotype, {search::tag(b.member.operation), b.member.parents}});
    }
    return out;
  }

  void tell(std::span<const search::Told> told) {
    for (const auto& t : told) {
      auto it = index_.find(t.ticket);
      if (it == index_.end()) throw std::invalid_argument("told a ticket that was not asked");
      auto& b = batch_[it->second];
      if (b.told) throw std::invalid_argument("told a ticket twice");
      b.member.evaluation = t.evaluation;
      b.told = true;
      told_++;
    }
    if (!batch_.empty() && told_ == batch_.size()) finish();
  }

  bool stopped() const { return s_.stopped || (st_.generations > 0 && s_.generation >= st_.generations) || (st_.evaluations > 0 && s_.evaluations >= st_.evaluations); }

  // Proposals from outside, made by `source`: asked with the next generation, credited to the source. A proposal made from members
  // (a person's or a model's variation of them) names their tickets in `parents`, one list for each genotype, or none: its lineage
  // then records them, it inherits their age, and a parent whose proposed child survives is credited as useful.
  void propose(const std::vector<G>& genotypes, const std::string& source, const std::vector<std::vector<Ticket>>& parents = {}) {
    if (!parents.empty() && parents.size() != genotypes.size()) throw std::invalid_argument("heuresis::evolution: the parents of each proposal, or of none");
    for (std::size_t i = 0; i < genotypes.size(); i++) proposals_.push_back({genotypes[i], source, parents.empty() ? std::vector<Ticket>{} : parents[i]});
  }

  // Evaluated members from elsewhere (another island): each replaces the worst member when it is not worse.
  void inject(std::span<const search::Individual<G>> in) {
    for (const auto& ind : in) {
      Member<G> m;
      m.ticket = ind.ticket, m.genotype = ind.genotype, m.evaluation = ind.evaluation, m.operation = "migrant";
      m.parents = ind.origin.parents, m.born = s_.generation, m.last_useful = s_.generation;
      m.score = scored(m);
      if (s_.population.empty()) { s_.population.push_back(m); continue; }
      const auto pos = positions(pointers(s_.population));
      std::size_t worst = 0;
      for (std::size_t i = 1; i < pos.size(); i++)
        if (pos[i] > pos[worst]) worst = i;
      if (!less_members(s_.population[worst], m)) s_.population[worst] = m;
    }
  }

  std::vector<search::Individual<G>> emigrants(std::size_t k) const {
    std::vector<search::Individual<G>> out;
    const auto pos = positions(pointers(s_.population));
    std::vector<std::size_t> o(pos.size());
    std::iota(o.begin(), o.end(), std::size_t{0});
    std::sort(o.begin(), o.end(), [&](std::size_t a, std::size_t b) { return pos[a] < pos[b]; });
    for (std::size_t i = 0; i < std::min(k, o.size()); i++) {
      const auto& m = s_.population[o[i]];
      out.push_back({m.ticket, m.genotype, m.evaluation, {search::tag(m.operation), m.parents}});
    }
    return out;
  }

  // One generation with the evaluator f(G) → Evaluation, serially: ask everything, evaluate, tell.
  template <class F>
  void step(F&& f) {
    const std::size_t before = s_.generation;
    auto cs = ask(std::numeric_limits<std::size_t>::max());
    if (cs.empty() && s_.generation != before) return;
    std::vector<search::Told> told;
    for (auto& c : cs) {
      Evaluation e;
      try { e = f(c.genotype); } catch (const std::exception& ex) { e = search::failure(ex.what()); }
      told.push_back({c.ticket, e});
    }
    tell(told);
  }

  // ---- what it holds ----
  const State<G>& state() const {
    s_.quality = credits_.quality, s_.probability = credits_.probability, s_.successes = credits_.successes, s_.failures = credits_.failures;
    s_.count = credits_.count, s_.total = credits_.total;
    return s_;
  }
  const std::vector<Member<G>>& population() const { return s_.population; }
  const std::vector<Kept<G>>& archive() const { return s_.archive; }
  const std::vector<Retired<G>>& retired() const { return s_.retired; }
  const std::vector<Statistics>& statistics() const { return s_.statistics; }
  const std::vector<Lineage>& lineage() const { return s_.lineage; }
  const Credits& credits() const { return credits_; }
  std::size_t generation() const { return s_.generation; }
  std::size_t evaluations() const { return s_.evaluations; }
  const Strategy& strategy() const { return st_; }
  std::vector<std::string> operator_names() const {
    std::vector<std::string> out;
    for (const auto& m : ops_.makers) out.push_back(m.name);
    for (const auto& m : ops_.mutations) out.push_back(m.name);
    for (const auto& c : ops_.crossovers) out.push_back(c.name);
    return out;
  }
  // The members of the population, best first by the scoring.
  std::vector<std::size_t> ranking() const {
    const auto pos = positions(pointers(s_.population));
    std::vector<std::size_t> o(pos.size());
    std::iota(o.begin(), o.end(), std::size_t{0});
    std::sort(o.begin(), o.end(), [&](std::size_t a, std::size_t b) { return pos[a] < pos[b]; });
    return o;
  }

 private:
  struct Pending {
    Member<G> member;
    int op = -1;                  // the operator's index, or −1 for a proposal
    std::vector<double> parent_best;  // the best parent's score, for credit
    bool has_parent = false;
    bool asked = false, told = false, reused = false;
  };
  struct Proposal { G genotype; std::string source; std::vector<Ticket> parents; };

  Operators<G> ops_;
  Strategy st_;
  Judge<G> judge_;
  search::Draws draws_;
  mutable State<G> s_;
  Credits credits_;
  std::unordered_map<std::string, Evaluation> cache_;
  std::vector<Pending> batch_;
  std::unordered_map<Ticket, std::size_t> index_;
  std::size_t told_ = 0;
  std::vector<Proposal> proposals_;

  static constexpr std::uint32_t kVary = search::tag("heuresis.evolution.vary");
  static constexpr std::uint32_t kSurvive = search::tag("heuresis.evolution.survive");
  static constexpr std::uint32_t kSeed = search::tag("heuresis.evolution.seed");

  // ---- operators and their weights ----
  std::vector<double> weights() const {
    const std::size_t M = ops_.makers.size(), U = ops_.mutations.size(), C = ops_.crossovers.size();
    auto sum = [](const auto& v) { double s = 0; for (const auto& x : v) s += std::max(0.0, x.weight); return s; };
    double gm = st_.generator_rate, gc = C ? st_.crossover_rate : 0.0, gu = 1 - st_.generator_rate - (C ? st_.crossover_rate : 0.0);
    if (U == 0) { gc += std::max(0.0, gu); gu = 0; }
    if (C == 0 && U == 0) gm = 1;
    gu = std::max(0.0, gu);
    std::vector<double> w;
    const double sm = sum(ops_.makers), su = sum(ops_.mutations), sc = sum(ops_.crossovers);
    for (const auto& m : ops_.makers) w.push_back(sm > 0 ? gm * std::max(0.0, m.weight) / sm : 0.0);
    for (const auto& m : ops_.mutations) w.push_back(su > 0 ? gu * std::max(0.0, m.weight) / su : 0.0);
    for (const auto& c : ops_.crossovers) w.push_back(sc > 0 ? gc * std::max(0.0, c.weight) / sc : 0.0);
    (void)M;
    return w;
  }

  // ---- ordering ----
  std::vector<double> scored(const Member<G>& m) const {
    std::vector<double> s = m.evaluation.failed() ? std::vector<double>() : m.evaluation.objectives;
    if (st_.novelty && !m.evaluation.failed()) s.push_back(-m.novelty);
    return s;
  }
  double scalar(const Member<G>& m) const {
    if (m.evaluation.failed()) return std::numeric_limits<double>::infinity();
    if (st_.scoring == Scoring::weighted) {
      double v = 0;
      for (std::size_t k = 0; k < m.score.size(); k++) v += (k < st_.weights.size() ? st_.weights[k] : 1.0) * m.score[k];
      return v;
    }
    return m.score.empty() ? std::numeric_limits<double>::infinity() : m.score[0];
  }
  // a before b, for the scorings that are total orders (all but pareto).
  bool less_members(const Member<G>& a, const Member<G>& b) const {
    const bool fa = a.evaluation.failed(), fb = b.evaluation.failed();
    if (fa || fb) return !fa && fb;
    const double va = a.evaluation.violation(), vb = b.evaluation.violation();
    if (va != vb) return va < vb;
    if (st_.scoring == Scoring::lexicographic || st_.scoring == Scoring::pareto) {
      for (std::size_t k = 0; k < std::min(a.score.size(), b.score.size()); k++) {
        if (a.score[k] < b.score[k] - st_.lexicographic_tolerance) return true;
        if (b.score[k] < a.score[k] - st_.lexicographic_tolerance) return false;
      }
      return false;
    }
    return scalar(a) < scalar(b);
  }
  static std::vector<const Member<G>*> pointers(const std::vector<Member<G>>& v) {
    std::vector<const Member<G>*> p;
    for (const auto& m : v) p.push_back(&m);
    return p;
  }
  // Each member's position in the order (0 the best): Pareto fronts with crowding, or the total order; ties by ticket.
  std::vector<std::size_t> positions(const std::vector<const Member<G>*>& pool) const {
    const std::size_t n = pool.size();
    std::vector<std::size_t> o(n);
    std::iota(o.begin(), o.end(), std::size_t{0});
    if (st_.scoring == Scoring::pareto && n > 0) {
      std::vector<multiobjective::Point> pts(n);
      std::size_t m = 0;
      for (auto* p : pool)
        if (!p->score.empty()) { m = p->score.size(); break; }
      for (std::size_t i = 0; i < n; i++)
        pts[i] = pool[i]->score.size() == m && m > 0 && pool[i]->evaluation.feasible() ? pool[i]->score : multiobjective::Point(std::max<std::size_t>(m, 1), std::numeric_limits<double>::infinity());
      const auto fronts = multiobjective::non_dominated_sort(std::span<const multiobjective::Point>(pts));
      std::vector<double> crowd(n, 0.0);
      for (const auto& f : fronts.fronts) {
        const auto d = multiobjective::crowding_distance(pts, f);
        for (std::size_t k = 0; k < f.size(); k++) crowd[f[k]] = d[k];
      }
      std::stable_sort(o.begin(), o.end(), [&](std::size_t a, std::size_t b) {
        if (fronts.rank[a] != fronts.rank[b]) return fronts.rank[a] < fronts.rank[b];
        if (crowd[a] != crowd[b]) return crowd[a] > crowd[b];
        return pool[a]->ticket < pool[b]->ticket;
      });
    } else {
      std::stable_sort(o.begin(), o.end(), [&](std::size_t a, std::size_t b) {
        if (less_members(*pool[a], *pool[b])) return true;
        if (less_members(*pool[b], *pool[a])) return false;
        return pool[a]->ticket < pool[b]->ticket;
      });
    }
    std::vector<std::size_t> pos(n);
    for (std::size_t r = 0; r < n; r++) pos[o[r]] = r;
    return pos;
  }

  // ---- parents ----
  const Member<G>& select(const std::vector<const Member<G>*>& pool, const std::vector<std::size_t>& pos, calx::random::Stream& s) const {
    if (!s_.archive.empty() && st_.from_archive > 0 && s.uniform() < st_.from_archive) return s_.archive[s.below(s_.archive.size())].member;
    const std::size_t n = pool.size();
    if (n == 0) throw std::logic_error("no parent to select");
    auto by_pos = [&](std::size_t a, std::size_t b) { return pos[a] < pos[b]; };
    switch (st_.parents) {
      case Parents::uniform: return *pool[s.below(n)];
      case Parents::tournament: return *pool[selection::tournament(n, std::max<std::size_t>(1, st_.tournament), by_pos, s)];
      case Parents::truncation: {
        const std::size_t top = std::max<std::size_t>(1, static_cast<std::size_t>(std::ceil(st_.truncation * static_cast<double>(n))));
        const std::size_t r = s.below(top);
        for (std::size_t i = 0; i < n; i++)
          if (pos[i] == r) return *pool[i];
        return *pool[0];
      }
      case Parents::ranking: {
        const auto p = selection::linear_ranking(n, st_.ranking_pressure);
        std::vector<std::size_t> ordered(n);
        for (std::size_t i = 0; i < n; i++) ordered[pos[i]] = i;
        return *pool[selection::by_rank(ordered, p, s)];
      }
      case Parents::roulette:
      case Parents::boltzmann: {
        std::vector<double> v(n);
        double lo = std::numeric_limits<double>::infinity(), hi = -lo;
        for (std::size_t i = 0; i < n; i++) {
          v[i] = st_.scoring == Scoring::pareto ? static_cast<double>(pos[i]) : scalar(*pool[i]);
          if (std::isfinite(v[i])) lo = std::min(lo, v[i]), hi = std::max(hi, v[i]);
        }
        const double spread = hi > lo ? hi - lo : 1.0;
        std::vector<double> w(n);
        for (std::size_t i = 0; i < n; i++) {
          if (!std::isfinite(v[i])) { w[i] = 0; continue; }
          w[i] = st_.parents == Parents::roulette ? (hi - v[i]) + 0.05 * spread : std::exp(-(v[i] - lo) / (std::max(st_.temperature, 1e-9) * spread));
        }
        if (std::accumulate(w.begin(), w.end(), 0.0) <= 0) return *pool[s.below(n)];
        return *pool[selection::roulette(w, s)];
      }
      case Parents::lexicase:
      case Parents::epsilon_lexicase: {
        std::vector<std::vector<double>> errors(n);
        std::size_t m = 0;
        for (auto* p : pool) m = std::max(m, p->score.size());
        for (std::size_t i = 0; i < n; i++) {
          errors[i] = pool[i]->score;
          errors[i].resize(m, std::numeric_limits<double>::infinity());
        }
        if (m == 0) return *pool[s.below(n)];
        const auto eps = st_.parents == Parents::epsilon_lexicase ? selection::case_epsilons(errors) : std::vector<double>{};
        return *pool[selection::lexicase(errors, s, eps)];
      }
      default: {  // novelty: a tournament on novelty
        std::size_t best = s.below(n);
        for (std::size_t t = 1; t < std::max<std::size_t>(1, st_.tournament); t++) {
          const std::size_t c = s.below(n);
          if (pool[c]->novelty > pool[best]->novelty) best = c;
        }
        return *pool[best];
      }
    }
  }

  // ---- making a generation ----
  Context<G> context() const { return Context<G>{s_.population, s_.archive, s_.generation}; }

  std::string key_of(const G& g) const { return ops_.key ? ops_.key(g) : std::string(); }

  const Evaluation* remembered(const std::string& key) const {
    if (key.empty()) return nullptr;
    auto it = cache_.find(key);
    return it == cache_.end() ? nullptr : &it->second;
  }

  Ticket next_ticket() { return search::make_ticket(draws_.island(), s_.serial++); }

  void add(Member<G> m, int op, std::vector<double> parent_best, bool has_parent) {
    m.ticket = next_ticket();
    m.born = s_.generation, m.last_useful = s_.generation;
    index_[m.ticket] = batch_.size();
    const Evaluation* known = remembered(key_of(m.genotype));
    Pending p{std::move(m), op, std::move(parent_best), has_parent, false, false, false};
    if (known) p.member.evaluation = *known, p.asked = p.told = p.reused = true, told_++;
    batch_.push_back(std::move(p));
  }

  void build() {
    index_.clear();
    told_ = 0;
    const auto ctx = context();
    std::unordered_set<std::string> batch_keys, living;
    for (const auto& m : s_.population) living.insert(key_of(m.genotype));
    for (const auto& k : s_.archive) living.insert(key_of(k.member.genotype));
    std::size_t repeats = 0;
    auto accept = [&](const G& g) {
      const std::string k = key_of(g);
      if (!k.empty() && (living.count(k) || batch_keys.count(k))) return false;
      if (!k.empty()) batch_keys.insert(k);
      return true;
    };
    const bool first = s_.population.empty() || s_.reseed;
    if (first) {
      // Made new by the generators, in proportion to their weights.
      const std::size_t want = s_.reseed ? st_.mu - std::min(st_.mu, s_.population.size()) : st_.mu;
      double total = 0;
      for (const auto& mk : ops_.makers) total += std::max(0.0, mk.weight);
      std::size_t made = 0, k = 0;
      for (std::size_t i = 0; i < ops_.makers.size(); i++) {
        const auto& mk = ops_.makers[i];
        std::size_t n = i + 1 == ops_.makers.size() ? want - made : static_cast<std::size_t>(std::round(want * std::max(0.0, mk.weight) / (total > 0 ? total : 1)));
        n = std::min(n, want - made);
        if (n == 0) continue;
        auto s = draws_(kSeed, s_.generation, i);
        for (int attempt = 0; attempt < 3 && n > 0; attempt++) {
          std::vector<G> gs;
          try { gs = mk.make(n, ctx, s); } catch (const std::exception&) { gs.clear(); }
          for (const auto& g : gs) {
            if (n == 0) break;
            if (!accept(g)) { repeats++; continue; }
            Member<G> m;
            m.genotype = g, m.operation = mk.name, m.age = 0, m.restarted = s_.reseed;
            add(std::move(m), static_cast<int>(i), {}, false);
            made++, n--, k++;
          }
        }
      }
      s_.reseed = false;
    } else {
      const auto pool = pointers(s_.population);
      const auto pos = positions(pool);
      const std::size_t M = ops_.makers.size(), U = ops_.mutations.size();
      const std::size_t L = st_.survival == Survival::alps ? std::max<std::size_t>(1, st_.layers) : 1;
      for (std::size_t i = 0; i < st_.lambda; i++) {
        auto s = draws_(kVary, s_.generation, i);
        // ALPS: children of layer ℓ take parents from layers ℓ and ℓ − 1.
        std::vector<const Member<G>*> lp;
        std::vector<std::size_t> lpos;
        const std::size_t layer = L > 1 ? i % L : 0;
        if (L > 1) {
          for (std::size_t j = 0; j < pool.size(); j++)
            if (pool[j]->layer == layer || pool[j]->layer + 1 == layer) lp.push_back(pool[j]), lpos.push_back(pos[j]);
          if (lp.empty()) lp = pool, lpos = pos;
          // Re-rank within the layers' pool.
          std::vector<std::size_t> o(lp.size());
          std::iota(o.begin(), o.end(), std::size_t{0});
          std::sort(o.begin(), o.end(), [&](std::size_t a, std::size_t b) { return lpos[a] < lpos[b]; });
          for (std::size_t r = 0; r < o.size(); r++) lpos[o[r]] = r;
        }
        const auto& P = L > 1 ? lp : pool;
        const auto& Q = L > 1 ? lpos : pos;
        bool made = false;
        for (std::size_t attempt = 0; attempt <= st_.retries && !made; attempt++) {
          const std::size_t op = credits_.choose(st_, s);
          std::optional<G> child;
          std::vector<const Member<G>*> parents;
          try {
            if (op < M) {
              auto gs = ops_.makers[op].make(1, ctx, s);
              if (!gs.empty()) child = gs[0];
            } else if (op < M + U) {
              const auto& p = select(P, Q, s);
              parents = {&p};
              child = ops_.mutations[op - M].apply(p, ctx, s);
            } else {
              const auto& a = select(P, Q, s);
              const auto& b = select(P, Q, s);
              parents = {&a, &b};
              child = ops_.crossovers[op - M - U].apply(a, b, ctx, s);
            }
          } catch (const std::exception&) { child.reset(); }
          if (!child) continue;
          if (!accept(*child)) { repeats++; continue; }
          Member<G> m;
          m.genotype = *child, m.operation = operator_names()[op];
          std::vector<double> best;
          std::size_t age = 0;
          const Member<G>* bp = nullptr;
          for (auto* p : parents) {
            m.parents.push_back(p->ticket);
            age = std::max(age, p->age);
            if (!bp || less_members(*p, *bp)) bp = p;
          }
          m.age = parents.empty() ? 0 : age;
          m.layer = parents.empty() ? 0 : layer;
          add(std::move(m), static_cast<int>(op), bp ? bp->score : std::vector<double>{}, bp != nullptr);
          made = true;
        }
      }
    }
    // Proposals from outside.
    for (const auto& p : proposals_) {
      if (!accept(p.genotype)) { repeats++; continue; }
      Member<G> m;
      m.genotype = p.genotype, m.operation = p.source, m.age = 0, m.parents = p.parents;
      for (Ticket t : p.parents) {
        for (const auto& q : s_.population)
          if (q.ticket == t) m.age = std::max(m.age, q.age);
        for (const auto& q : s_.archive)
          if (q.member.ticket == t) m.age = std::max(m.age, q.member.age);
      }
      add(std::move(m), -1, {}, false);
    }
    proposals_.clear();
    repeats_ = repeats;
    if (batch_.empty()) {
      // Nothing new could be made: the space is exhausted as far as the operators reach.
      s_.stopped = true;
    }
  }

  std::size_t repeats_ = 0;

  // ---- after a generation is told ----
  void finish() {
    Statistics st;
    st.generation = s_.generation;
    const std::size_t K = credits_.size();
    st.births.assign(K, 0), st.survivals.assign(K, 0);
    std::vector<Member<G>> children;
    std::vector<Pending> pend = std::move(batch_);
    batch_.clear(), index_.clear(), told_ = 0;
    for (auto& p : pend) {
      const std::string k = key_of(p.member.genotype);
      if (p.reused) st.reused++;
      else {
        s_.evaluations++;
        if (p.member.evaluation.failed()) st.failures++;
        if (!k.empty() && !cache_.count(k)) cache_[k] = p.member.evaluation, s_.cache.emplace_back(k, p.member.evaluation);
      }
      if (p.op >= 0) st.births[static_cast<std::size_t>(p.op)]++;
      else st.proposals[p.member.operation]++;
      children.push_back(p.member);
    }
    st.repeats = repeats_;
    // Novelty against the population, the children and the archive.
    if (st_.novelty) novelty(children);
    for (auto& c : children) c.score = scored(c);
    for (auto& m : s_.population) m.score = scored(m);

    const bool first = s_.population.empty();
    std::vector<Member<G>> before = s_.population;
    survive(children);

    // Who survived, and the credit of their makers.
    std::unordered_set<Ticket> alive;
    for (const auto& m : s_.population) alive.insert(m.ticket);
    std::unordered_map<Ticket, std::size_t> in_pop, in_archive;
    for (std::size_t i = 0; i < s_.population.size(); i++) in_pop[s_.population[i].ticket] = i;
    for (std::size_t i = 0; i < s_.archive.size(); i++) in_archive[s_.archive[i].member.ticket] = i;
    for (std::size_t c = 0; c < children.size(); c++) {
      const auto& ch = children[c];
      const bool survived = alive.count(ch.ticket) > 0;
      Lineage lin{ch.ticket, s_.generation, ch.operation, ch.parents, ch.evaluation.objectives, ch.evaluation.failed(), survived, false};
      s_.lineage.push_back(lin);
      if (survived) {
        for (Ticket pt : ch.parents) {
          if (auto it = in_pop.find(pt); it != in_pop.end()) s_.population[it->second].last_useful = s_.generation;
          if (auto it = in_archive.find(pt); it != in_archive.end()) s_.archive[it->second].member.last_useful = s_.generation;
        }
      }
      const int op = pend[c].op;
      if (op >= 0) {
        if (survived) st.survivals[static_cast<std::size_t>(op)]++;
        if (!first) credits_.reward(static_cast<std::size_t>(op), reward(ch, pend[c], survived), st_);
      } else if (survived) {
        st.proposals_surviving[ch.operation]++;
      }
    }
    // Ages.
    for (auto& m : s_.population) m.age++;

    // The archive.
    archive(children, st);

    // The best, and stagnation.
    const auto pos = positions(pointers(s_.population));
    std::size_t b = 0;
    for (std::size_t i = 0; i < pos.size(); i++)
      if (pos[i] == 0) b = i;
    if (!s_.population.empty()) {
      const auto& now = s_.population[b];
      if (!s_.best || less_members(now, *s_.best) || (st_.scoring == Scoring::pareto && !s_.best)) {
        st.improved = !s_.best || less_members(now, *s_.best);
        s_.best = now;
      }
    }
    if (st.improved || first) s_.stall = 0;
    else s_.stall++;
    if (st_.stagnation > 0 && s_.stall >= st_.stagnation) {
      if (st_.restart_on_stagnation) {
        // Keep the elites; the rest is made new at the next generation.
        std::vector<std::size_t> o = ranking();
        std::vector<Member<G>> keep;
        for (std::size_t i = 0; i < std::min(st_.elites, o.size()); i++) keep.push_back(s_.population[o[i]]);
        s_.population = keep;
        s_.reseed = true, s_.stall = 0, st.restarted = true;
      } else {
        s_.stopped = true;
      }
    }
    // Statistics.
    statistics(st);
    st.probability = credits_.probability;
    st.evaluations = s_.evaluations;
    s_.statistics.push_back(st);
    s_.generation++;
    (void)before;
  }

  double reward(const Member<G>& child, const Pending& p, bool survived) const {
    if (child.evaluation.failed()) return 0;
    switch (st_.credit) {
      case Credit::survival: return survived ? 1 : 0;
      case Credit::improvement: {
        if (!p.has_parent) return survived ? 1 : 0;
        Member<G> parent;
        parent.score = p.parent_best;
        parent.evaluation.objectives = p.parent_best;
        if (st_.scoring == Scoring::pareto) {
          if (parent.score.size() != child.score.size()) return 0;
          return multiobjective::dominates(child.score, parent.score) ? 1 : 0;
        }
        const double a = scalar(parent), c = scalar(child);
        if (!std::isfinite(a)) return 1;
        const double scale = std::max(std::abs(a), 1e-12);
        return std::clamp((a - c) / scale, 0.0, 1.0);
      }
      default: {
        if (!p.has_parent) return survived ? 1 : 0;
        Member<G> parent;
        parent.score = p.parent_best;
        parent.evaluation.objectives = p.parent_best;
        if (st_.scoring == Scoring::pareto) return parent.score.size() == child.score.size() && multiobjective::dominates(child.score, parent.score) ? 1 : 0;
        return less_members(child, parent) ? 1 : 0;
      }
    }
  }

  void novelty(std::vector<Member<G>>& children) {
    std::vector<const G*> others;
    for (const auto& m : s_.population) others.push_back(&m.genotype);
    for (const auto& c : children) others.push_back(&c.genotype);
    for (const auto& k : s_.archive) others.push_back(&k.member.genotype);
    auto nov = [&](const G& g) {
      std::vector<double> d;
      for (auto* o : others)
        if (o != &g) d.push_back(ops_.distance(g, *o));
      const std::size_t k = std::min(st_.novelty_k, d.size());
      if (k == 0) return 0.0;
      std::partial_sort(d.begin(), d.begin() + static_cast<long>(k), d.end());
      return std::accumulate(d.begin(), d.begin() + static_cast<long>(k), 0.0) / static_cast<double>(k);
    };
    for (auto& c : children) c.novelty = nov(c.genotype);
    for (auto& m : s_.population) m.novelty = nov(m.genotype);
  }

  // ---- survival ----
  std::vector<Member<G>> best_of(std::vector<Member<G>> pool, std::size_t n) const {
    const auto pos = positions(pointers(pool));
    std::vector<std::size_t> o(pool.size());
    std::iota(o.begin(), o.end(), std::size_t{0});
    std::sort(o.begin(), o.end(), [&](std::size_t a, std::size_t b) { return pos[a] < pos[b]; });
    std::vector<Member<G>> out;
    for (std::size_t i = 0; i < std::min(n, o.size()); i++) out.push_back(pool[o[i]]);
    return out;
  }

  void survive(std::vector<Member<G>>& children) {
    auto& P = s_.population;
    const std::size_t mu = st_.mu;
    if (P.empty() && st_.survival != Survival::niches) {
      P = best_of(children, mu);
      return;
    }
    auto joined = [&]() {
      std::vector<Member<G>> all = P;
      all.insert(all.end(), children.begin(), children.end());
      return all;
    };
    auto worst_index = [&](const std::vector<Member<G>>& v) {
      const auto pos = positions(pointers(v));
      std::size_t w = 0;
      for (std::size_t i = 1; i < pos.size(); i++)
        if (pos[i] > pos[w]) w = i;
      return w;
    };
    auto nearest = [&](const G& g, const std::vector<Member<G>>& v, const std::vector<std::size_t>& among) {
      std::size_t best = among.empty() ? 0 : among[0];
      double bd = std::numeric_limits<double>::infinity();
      for (std::size_t i : among) {
        const double d = ops_.distance(g, v[i].genotype);
        if (d < bd) bd = d, best = i;
      }
      return best;
    };
    switch (st_.survival) {
      case Survival::generational: {
        auto elite = best_of(P, std::min(st_.elites, P.size()));
        auto rest = best_of(children, mu - std::min(mu, elite.size()));
        if (elite.size() + rest.size() < mu) {  // too few children: the best of the rest of the parents fill
          std::vector<Member<G>> others;
          std::unordered_set<Ticket> used;
          for (auto& e : elite) used.insert(e.ticket);
          for (auto& m : P)
            if (!used.count(m.ticket)) others.push_back(m);
          auto fill = best_of(others, mu - elite.size() - rest.size());
          rest.insert(rest.end(), fill.begin(), fill.end());
        }
        P = elite;
        P.insert(P.end(), rest.begin(), rest.end());
        return;
      }
      case Survival::plus: P = best_of(joined(), mu); return;
      case Survival::comma: P = best_of(children, mu); return;
      case Survival::replace_worst:
        for (auto& c : children) {
          P.push_back(c);
          if (P.size() > mu) P.erase(P.begin() + static_cast<long>(worst_index(P)));
        }
        return;
      case Survival::aging:
        for (auto& c : children) {
          P.push_back(c);
          while (P.size() > mu) {
            std::size_t oldest = 0;
            for (std::size_t i = 1; i < P.size(); i++)
              if (P[i].born < P[oldest].born || (P[i].born == P[oldest].born && P[i].ticket < P[oldest].ticket)) oldest = i;
            P.erase(P.begin() + static_cast<long>(oldest));
          }
        }
        return;
      case Survival::deterministic_crowding:
        for (auto& c : children) {
          std::vector<std::size_t> rivals;
          for (Ticket t : c.parents)
            for (std::size_t i = 0; i < P.size(); i++)
              if (P[i].ticket == t) rivals.push_back(i);
          if (rivals.empty()) { rivals.resize(P.size()); std::iota(rivals.begin(), rivals.end(), std::size_t{0}); }
          const std::size_t r = nearest(c.genotype, P, rivals);
          if (!less_members(P[r], c)) P[r] = c;  // the child takes its nearest parent's place unless worse
        }
        return;
      case Survival::restricted_tournament: {
        std::size_t k = 0;
        for (auto& c : children) {
          auto s = draws_(kSurvive, s_.generation, k++);
          std::vector<std::size_t> window;
          for (std::size_t w = 0; w < std::max<std::size_t>(1, st_.window); w++) window.push_back(s.below(P.size()));
          const std::size_t r = nearest(c.genotype, P, window);
          if (P.size() < mu) P.push_back(c);
          else if (less_members(c, P[r])) P[r] = c;
        }
        return;
      }
      case Survival::clearing: {
        auto all = joined();
        const auto pos = positions(pointers(all));
        std::vector<double> values(all.size());
        for (std::size_t i = 0; i < all.size(); i++) values[i] = static_cast<double>(pos[i]);
        const auto winners = diversity_clearing(values, all);
        std::vector<Member<G>> won, lost;
        for (std::size_t i = 0; i < all.size(); i++) (winners[i] ? won : lost).push_back(all[i]);
        P = best_of(won, mu);
        if (P.size() < mu) {
          auto fill = best_of(lost, mu - P.size());
          P.insert(P.end(), fill.begin(), fill.end());
        }
        return;
      }
      case Survival::nsga2: {
        Strategy copy = st_;
        auto all = joined();
        // Fronts with crowding, whatever the scoring.
        std::vector<multiobjective::Point> pts(all.size());
        std::size_t m = 0;
        for (const auto& x : all)
          if (!x.score.empty()) { m = x.score.size(); break; }
        for (std::size_t i = 0; i < all.size(); i++) pts[i] = all[i].score.size() == m && m > 0 && all[i].evaluation.feasible() ? all[i].score : multiobjective::Point(std::max<std::size_t>(m, 1), std::numeric_limits<double>::infinity());
        const auto fronts = multiobjective::non_dominated_sort(std::span<const multiobjective::Point>(pts));
        std::vector<Member<G>> next;
        for (const auto& f : fronts.fronts) {
          if (next.size() + f.size() <= mu) { for (auto i : f) next.push_back(all[i]); continue; }
          const auto d = multiobjective::crowding_distance(pts, f);
          std::vector<std::size_t> o(f.size());
          std::iota(o.begin(), o.end(), std::size_t{0});
          std::stable_sort(o.begin(), o.end(), [&](std::size_t a, std::size_t b) { return d[a] > d[b]; });
          for (std::size_t k = 0; next.size() < mu && k < o.size(); k++) next.push_back(all[f[o[k]]]);
          break;
        }
        P = next;
        (void)copy;
        return;
      }
      case Survival::alps: { alps(children); return; }
      case Survival::niches: { niches(children); return; }
    }
  }

  std::vector<char> diversity_clearing(const std::vector<double>& values, const std::vector<Member<G>>& all) const {
    // heuresis::diversity::clearing, read through the representation's distance (diversity is not a dependency of this module's
    // header set, so its rule is restated here in the same form: values minimised, ties to the earlier index).
    const std::size_t n = values.size();
    std::vector<std::size_t> o(n);
    std::iota(o.begin(), o.end(), std::size_t{0});
    std::stable_sort(o.begin(), o.end(), [&](std::size_t a, std::size_t b) { return values[a] < values[b]; });
    std::vector<char> cleared(n, 0), winner(n, 0);
    for (std::size_t a = 0; a < n; a++) {
      const std::size_t i = o[a];
      if (cleared[i]) continue;
      winner[i] = 1;
      std::size_t winners = 1;
      for (std::size_t b = a + 1; b < n; b++) {
        const std::size_t j = o[b];
        if (cleared[j] || winner[j] || ops_.distance(all[i].genotype, all[j].genotype) >= st_.radius) continue;
        if (winners < std::max<std::size_t>(1, st_.capacity)) { winner[j] = 1; winners++; }
        else cleared[j] = 1;
      }
    }
    return winner;
  }

  // ALPS: layer ℓ holds members of age at most gap · 2^ℓ (the top layer, any age). Each layer keeps its best μ/L. A member too old
  // for its layer is offered to the next; the bottom layer is made new every `gap` generations.
  void alps(std::vector<Member<G>>& children) {
    const std::size_t L = std::max<std::size_t>(1, st_.layers), each = std::max<std::size_t>(1, st_.mu / L);
    auto limit = [&](std::size_t l) { return l + 1 >= L ? std::numeric_limits<std::size_t>::max() : st_.age_gap * (std::size_t{1} << l); };
    std::vector<std::vector<Member<G>>> layer(L);
    for (auto& m : s_.population) layer[std::min(m.layer, L - 1)].push_back(m);
    for (auto& c : children) layer[std::min(c.layer, L - 1)].push_back(c);
    // From the bottom up: the too old move up.
    for (std::size_t l = 0; l < L; l++) {
      std::vector<Member<G>> stay, up;
      for (auto& m : layer[l]) ((m.age + 1 > limit(l)) ? up : stay).push_back(m);
      layer[l] = best_of(stay, each);
      if (l + 1 < L)
        for (auto& m : up) { m.layer = l + 1; layer[l + 1].push_back(m); }
    }
    s_.population.clear();
    for (auto& l : layer) s_.population.insert(s_.population.end(), l.begin(), l.end());
    if (s_.generation > 0 && s_.generation % st_.age_gap == 0) {
      // Reseed the bottom layer at the next generation: its members are dropped and made new.
      std::vector<Member<G>> keep;
      for (auto& m : s_.population)
        if (m.layer > 0) keep.push_back(m);
      if (keep.size() >= 2) { s_.population = keep; s_.reseed = true; }
    }
  }

  // MAP-Elites: the best member of each cell of the descriptors' grid; the population is the elites.
  void niches(std::vector<Member<G>>& children) {
    std::map<std::vector<std::size_t>, Member<G>> cell;
    auto cell_of = [&](const Member<G>& m) -> std::optional<std::vector<std::size_t>> {
      const auto& d = m.evaluation.descriptors;
      if (m.evaluation.failed() || d.size() != st_.bins.size()) return std::nullopt;
      std::vector<std::size_t> c(d.size());
      for (std::size_t k = 0; k < d.size(); k++) {
        const double u = (d[k] - st_.lower[k]) / (st_.upper[k] - st_.lower[k]);
        c[k] = static_cast<std::size_t>(std::clamp(u, 0.0, 1.0 - 1e-12) * static_cast<double>(st_.bins[k]));
      }
      return c;
    };
    auto offer = [&](const Member<G>& m) {
      const auto c = cell_of(m);
      if (!c) return;
      auto it = cell.find(*c);
      if (it == cell.end() || less_members(m, it->second)) cell[*c] = m;
    };
    for (auto& m : s_.population) offer(m);
    for (auto& c : children) offer(c);
    s_.population.clear();
    for (auto& [k, m] : cell) s_.population.push_back(m);
  }

  // ---- the archive ----
  void archive(const std::vector<Member<G>>& children, Statistics& st) {
    const auto ctx = context();
    std::unordered_set<std::string> kept_keys;
    for (const auto& k : s_.archive) kept_keys.insert(key_of(k.member.genotype));
    for (const auto& r : s_.retired) kept_keys.insert(key_of(r.member.genotype));
    if (judge_.admit) {
      for (const auto& c : children) {
        if (c.evaluation.failed()) continue;
        const std::string k = key_of(c.genotype);
        if (!k.empty() && kept_keys.count(k)) continue;
        std::optional<std::string> note;
        try { note = judge_.admit(c, ctx); } catch (const std::exception&) { note.reset(); }
        if (!note) continue;
        Member<G> m = c;
        m.last_useful = s_.generation;
        s_.archive.push_back({m, s_.generation, *note});
        kept_keys.insert(k);
        st.admitted++;
        for (auto& l : s_.lineage)
          if (l.ticket == m.ticket) l.admitted = true;
      }
    }
    std::vector<Kept<G>> keep;
    for (auto& k : s_.archive) {
      std::optional<std::string> why;
      if (st_.recheck && judge_.recheck && k.admitted < s_.generation) {
        try { why = judge_.recheck(k, ctx); } catch (const std::exception& e) { why = std::string("its recheck failed: ") + e.what(); }
      }
      const std::size_t idle = s_.generation - std::max(k.admitted, k.member.last_useful);
      if (!why && st_.retire_after > 0 && idle > st_.retire_after)
        why = "no child of it survived in " + std::to_string(idle) + " generations";
      if (!why && st_.retire_age > 0 && s_.generation - k.admitted > st_.retire_age)
        why = "kept " + std::to_string(s_.generation - k.admitted) + " generations, more than " + std::to_string(st_.retire_age);
      if (why) {
        s_.retired.push_back({k.member, k.admitted, s_.generation, k.note, *why});
        st.retired++;
      } else {
        keep.push_back(k);
      }
    }
    // Over capacity: the longest idle go first.
    while (keep.size() > st_.archive_capacity) {
      std::size_t w = 0;
      for (std::size_t i = 1; i < keep.size(); i++)
        if (keep[i].member.last_useful < keep[w].member.last_useful) w = i;
      s_.retired.push_back({keep[w].member, keep[w].admitted, s_.generation, keep[w].note, "the archive was full, and it had gone longest without use"});
      keep.erase(keep.begin() + static_cast<long>(w));
      st.retired++;
    }
    s_.archive = std::move(keep);
    st.archive = s_.archive.size();
  }

  void statistics(Statistics& st) const {
    const auto& P = s_.population;
    std::size_t m = 0;
    for (const auto& x : P)
      if (!x.evaluation.failed()) { m = x.evaluation.objectives.size(); break; }
    st.best.assign(m, std::numeric_limits<double>::quiet_NaN()), st.median.assign(m, std::numeric_limits<double>::quiet_NaN());
    for (std::size_t k = 0; k < m; k++) {
      std::vector<double> v;
      for (const auto& x : P)
        if (!x.evaluation.failed() && x.evaluation.objectives.size() == m) v.push_back(x.evaluation.objectives[k]);
      if (v.empty()) continue;
      std::sort(v.begin(), v.end());
      st.best[k] = v.front(), st.median[k] = v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
    }
    if (ops_.distance && P.size() > 1) {
      double sum = 0;
      std::size_t n = 0;
      const std::size_t step = std::max<std::size_t>(1, P.size() * P.size() / 2 / 400);
      std::size_t idx = 0;
      for (std::size_t i = 0; i < P.size(); i++)
        for (std::size_t j = i + 1; j < P.size(); j++, idx++)
          if (idx % step == 0) sum += ops_.distance(P[i].genotype, P[j].genotype), n++;
      st.diversity = n ? sum / static_cast<double>(n) : 0.0;
    }
  }
};

// A serial run to the end of the strategy's budget.
template <class G, class F>
void evolve(Evolution<G>& e, F&& f) {
  while (!e.stopped()) {
    const std::size_t before = e.generation();
    e.step(f);
    if (e.generation() == before) break;  // nothing could be made
  }
}

}  // namespace heuresis::evolution
