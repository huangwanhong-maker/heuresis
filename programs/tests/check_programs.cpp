// Checks of heuresis programs: typed generation (every term well-sorted, of its sort, within its depth); the
// covariance of dimensionally consistent terms under a change of units, exactly; ramped half-and-half's depths and
// full trees' sizes; variation keeping sorts and limits; lexicographic parsimony; grammatical evolution's mapping
// on a worked example with wrapping; derivations, crossovers and mutations of a grammar yielding words that Earley's
// parser accepts; the exact rediscovery of d = ½ a t² from typed data; generational order-independence.

#include <heuresis/programs/programs.hpp>
#include <heuresis/search/run.hpp>

#include <morphe/formal/earley.hpp>

#include <calx/verification/check.hpp>

#include <algorithm>
#include <cmath>

using namespace heuresis::programs;
using heuresis::search::Draws;
using heuresis::search::tag;
namespace dim = physis::units::dim;

namespace {

// +, −, ×, ÷, the square, √, sin, exp; lengths, times and masses; the constant 2 and an ephemeral constant.
Signature typed_signature() {
  Signature sig(3);
  for (auto p : {library::add(), library::subtract(), library::multiply(), library::divide(), library::square(), library::root(), library::sine(), library::exponential()}) sig.add(p);
  sig.add(Variable{"L", of(dim::length)});
  sig.add(Variable{"T", of(dim::time)});
  sig.add(Variable{"M", of(dim::mass)});
  sig.add(Constant{"2", dimensionless(), 2});
  sig.add(Constant{"c", dimensionless(), 0, true, -1, 1});
  return sig;
}

// The power of two by which a quantity of dimension d scales when the kilogram is halved (λ = 2), the metre
// quartered (λ = 4) and the second doubled (λ = ½): 2^{e_kg + 2 e_m − e_s}.
int scale_exponent(const physis::units::Dimension& d) { return d.e[0] + 2 * d.e[1] - d.e[2]; }

}  // namespace

int main() {
  calx::verification::Suite s;
  const Draws draws(31);
  const Signature sig = typed_signature();
  const std::vector<Sort> targets{of(dim::length), of(dim::velocity), of(dim::acceleration), dimensionless(), of(dim::force), of(dim::area), of(dim::length / dim::mass)};

  // Typed generation.
  std::vector<Tree> trees;
  {
    bool sorted = true, rooted = true, shallow = true, all_generable = true;
    for (const auto& t : targets) all_generable = all_generable && sig.generable(t, 4);
    for (int k = 0; k < 3000; k++) {
      auto r = draws(tag("check.programs.generate"), 0, k);
      const Sort& target = targets[k % targets.size()];
      const std::size_t d = 2 + (k / 7) % 5;
      Tree t = generate(sig, target, d, k % 2, r);
      sorted = sorted && well_sorted(sig, t);
      rooted = rooted && t[0].sort == target;
      shallow = shallow && depth(sig, t) <= d;
      trees.push_back(std::move(t));
    }
    s.require("every target sort is generable within depth 4", all_generable);
    s.require("3000 generated terms are well-sorted", sorted);
    s.require("each has its target sort", rooted);
    s.require("each is within its depth", shallow);
    const Sort metre[2] = {of(dim::length), of(dim::length)};
    s.require("the square cannot head a term of dimension m, √ cannot take one", sig.argument_options(4, of(dim::length), 3).empty() && !sig.argument_options(4, of(dim::area), 3).empty() && !sig.synthesise(5, std::span<const Sort>(metre, 1)));
  }

  // Covariance under a change of units, exactly: each input of dimension D scaled by 2^k(D), the value by 2^k(dim t).
  {
    std::size_t tested = 0, exact = 0;
    for (const auto& t : trees) {
      const std::vector<double> x{1.7, 0.6, 2.3};
      const std::vector<double> y{std::ldexp(x[0], scale_exponent(dim::length)), std::ldexp(x[1], scale_exponent(dim::time)), std::ldexp(x[2], scale_exponent(dim::mass))};
      const double a = interpret(sig, t, x), b = interpret(sig, t, y);
      if (!std::isfinite(a) || !std::isfinite(b) || a == 0 || std::fabs(a) < 1e-200 || std::fabs(a) > 1e200) continue;
      tested++;
      exact += b == std::ldexp(a, scale_exponent(t[0].sort.dimension));
    }
    s.require("terms tested for covariance: " + std::to_string(tested), tested > 2000);
    s.require("a dimensionally consistent term is covariant under a change of units, bit for bit", exact == tested);
  }

  // Ramped half-and-half on an untyped signature: full terms are complete binary trees of their depth.
  {
    Signature u;
    u.add(library::add());
    u.add(library::multiply());
    u.add(Variable{"x", dimensionless()});
    std::vector<int> per_depth(7, 0);
    bool full_ok = true, grow_ok = true;
    for (std::size_t k = 0; k < 600; k++) {
      auto r = draws(tag("check.programs.ramp"), 0, k);
      const auto [d, is_full] = ramp(k, 2, 6);
      const Tree t = ramped(u, dimensionless(), k, 2, 6, r);
      per_depth[d]++;
      if (is_full) full_ok = full_ok && depth(u, t) == d && t.size() == (std::size_t{1} << (d + 1)) - 1;
      else grow_ok = grow_ok && depth(u, t) <= d;
    }
    s.require("ramped half-and-half: 120 terms at each depth from 2 to 6", per_depth[2] == 120 && per_depth[4] == 120 && per_depth[6] == 120);
    s.require("full terms are complete binary trees of size 2^{d+1} − 1", full_ok);
    s.require("grown terms are within their depth", grow_ok);
  }

  // Variation keeps sorts and limits.
  {
    Limits lim;
    lim.depth = 8;
    lim.size = 120;
    bool ok = true, within_limits = true;
    std::size_t changed = 0;
    for (int k = 0; k < 3000; k++) {
      auto r = draws(tag("check.programs.vary"), 0, k);
      const Tree& a = trees[k % trees.size()];
      const Tree& b = trees[(k * 7 + 3) % trees.size()];
      const auto [c1, c2] = subtree_crossover(sig, a, b, lim, r);
      const Tree m1 = subtree_mutation(sig, a, lim, r), m2 = point_mutation(sig, a, 0.3, r), m3 = hoist_mutation(sig, a, r), m4 = shrink_mutation(sig, a, r);
      for (const Tree* t : {&c1, &m1, &m2, &m3, &m4}) ok = ok && well_sorted(sig, *t) && (*t)[0].sort == a[0].sort;
      ok = ok && well_sorted(sig, c2) && c2[0].sort == b[0].sort;
      for (const Tree* t : {&c1, &m1}) within_limits = within_limits && (((*t).size() <= lim.size && depth(sig, *t) <= lim.depth) || *t == a);
      changed += m2.size() == a.size() && !std::equal(m2.begin(), m2.end(), a.begin(), [](const Node& x, const Node& y) { return x.symbol == y.symbol && x.value == y.value; });
    }
    s.require("crossover and the four mutations keep every term well-sorted and of its sort", ok);
    s.require("children beyond the limits are replaced by their parents", within_limits);
    s.require("point mutation changes symbols while keeping the shape", changed > 1000);
  }

  // Lexicographic parsimony: among equal evaluations, the smaller term.
  {
    const std::vector<heuresis::search::Evaluation> e{heuresis::search::scalar(1), heuresis::search::scalar(1), heuresis::search::scalar(0.5)};
    const std::vector<std::size_t> sizes{9, 3, 40};
    const auto less = lexicographic_parsimony(e, sizes);
    s.require("lexicographic parsimony: fitness first, then size", less(2, 1) && less(1, 0) && !less(0, 1));
  }

  // Grammatical evolution: E → E O E | V, O → + | *, V → x | y; codons (4, 5, 3, 2, 9).
  Grammar g;
  g.terminals = 4;  // x, y, +, *
  const Symbol E = g.add_nonterminal("E"), O = g.add_nonterminal("O"), V = g.add_nonterminal("V");
  g.add(E, {E, O, E});
  g.add(E, {V});
  g.add(O, {2});
  g.add(O, {3});
  g.add(V, {0});
  g.add(V, {1});
  {
    const std::vector<std::int64_t> codons{4, 5, 3, 2, 9};
    const Mapping m = map_codons(g, codons, 1);
    // By hand: E —4 mod 2 = 0→ E O E; E —5→ V; V —3→ y; O —2→ +; E —9→ V; V, wrapping, —4→ x.
    const Derivation expected{{E, 0}, {E, 1}, {V, 5}, {1, -1}, {O, 2}, {2, -1}, {E, 1}, {V, 4}, {0, -1}};
    bool same = m.derivation.size() == expected.size();
    for (std::size_t i = 0; same && i < expected.size(); i++) same = m.derivation[i].symbol == expected[i].symbol && m.derivation[i].rule == expected[i].rule;
    s.require("grammatical evolution maps (4, 5, 3, 2, 9) to y + x", m.valid && m.word == std::vector<Symbol>{1, 2, 0});
    s.require("reading six codons with one wrap", m.codons_read == 6 && m.wraps == 1);
    s.require("its derivation is the leftmost derivation worked by hand", same);
    s.require("without wrapping the same codons map to no word", !map_codons(g, codons, 0).valid);
  }

  // Words of the grammar: every yield is accepted by Earley's parser.
  {
    const morphe::formal::Earley parser(g);
    const GrammarSpace space(g);
    s.require("the least depths: V and O 1, E 2", space.least_depth(V) == 1 && space.least_depth(O) == 1 && space.least_depth(E) == 2);
    bool grown = true, varied = true, mapped = true;
    std::size_t valid = 0;
    std::vector<Derivation> pool;
    for (int k = 0; k < 1000; k++) {
      auto r = draws(tag("check.programs.grammar"), 0, k);
      const std::size_t d = 2 + k % 7;
      Derivation t = space.grow(E, d, r);
      grown = grown && space.depth(t) <= d && parser.accepts(space.yield(t));
      pool.push_back(std::move(t));
    }
    for (int k = 0; k < 1000; k++) {
      auto r = draws(tag("check.programs.grammar.vary"), 0, k);
      const auto [c1, c2] = space.crossover(pool[k], pool[(k * 13 + 1) % 1000], 10, r);
      const Derivation m = space.mutate(pool[k], 10, r);
      for (const Derivation* t : {&c1, &c2, &m}) varied = varied && space.depth(*t) <= 10 && parser.accepts(space.yield(*t));
      std::vector<std::int64_t> codons(12);
      for (auto& c : codons) c = static_cast<std::int64_t>(r.below(256));
      const Mapping mp = map_codons(g, codons, 2);
      if (mp.valid) {
        valid++;
        mapped = mapped && parser.accepts(mp.word) && space.yield(mp.derivation) == mp.word;
      }
    }
    s.require("grown derivations are within their depth and yield words of the language", grown);
    s.require("crossover and mutation of derivations yield words of the language", varied);
    s.require("every valid mapping of random codons is a word of the language, its derivation's yield (" + std::to_string(valid) + " valid)", mapped && valid > 100);
  }

  // Rediscovery of d = ½ a t² from data with units: a in m s⁻², t in s, the term of sort m.
  {
    Signature law(3);
    for (auto p : {library::add(), library::subtract(), library::multiply(), library::divide()}) law.add(p);
    law.add(Variable{"a", of(dim::acceleration)});
    law.add(Variable{"t", of(dim::time)});
    law.add(Constant{"2", dimensionless(), 2});
    law.add(Constant{"c", dimensionless(), 0, true, -1, 1});
    Cases cases;
    for (int i = 0; i < 20; i++) {
      const double a = 1 + 0.45 * i, t = 0.5 + 0.23 * ((i * 7) % 20);
      cases.inputs.push_back({a, t});
      cases.targets.push_back(0.5 * a * t * t);
    }
    double ymax = 0;
    for (double y : cases.targets) ymax = std::max(ymax, y);
    GpOptions o;
    o.target = of(dim::length);
    o.population = 300;
    o.max_depth = 5;
    o.tournament = 5;
    GeneticProgramming gp(law, o, Draws(4));
    const auto out = heuresis::search::run(gp, [&](const Tree& t) { return regression(law, t, cases); }, heuresis::search::Budget{300 * 40, INFINITY, 1e-24});
    const bool found = out.best && std::sqrt(out.best->evaluation.value()) <= 1e-12 * ymax;
    s.require("GP finds d = ½ a t² exactly within 40 generations: " + (out.best ? to_string(law, out.best->genotype) : std::string("none")) + " after " + std::to_string(static_cast<long>(out.evaluations)) + " evaluations", found);
    s.require("the law found has the sort of a length", out.best && out.best->genotype[0].sort == of(dim::length) && well_sorted(law, out.best->genotype));

    // The same data searched without dimensions: every symbol dimensionless.
    Signature plain(3);
    for (auto p : {library::add(), library::subtract(), library::multiply(), library::divide()}) plain.add(p);
    plain.add(Variable{"a", dimensionless()});
    plain.add(Variable{"t", dimensionless()});
    plain.add(Constant{"2", dimensionless(), 2});
    plain.add(Constant{"c", dimensionless(), 0, true, -1, 1});
    int typed_hits = 0, plain_hits = 0;
    double typed_evals = 0, plain_evals = 0;
    for (std::uint64_t seed = 100; seed < 110; seed++) {
      GpOptions ot = o, op = o;
      op.target = dimensionless();
      GeneticProgramming a(law, ot, Draws(seed)), b(plain, op, Draws(seed));
      const auto oa = heuresis::search::run(a, [&](const Tree& t) { return regression(law, t, cases); }, heuresis::search::Budget{300 * 20, INFINITY, 1e-24});
      const auto ob = heuresis::search::run(b, [&](const Tree& t) { return regression(plain, t, cases); }, heuresis::search::Budget{300 * 20, INFINITY, 1e-24});
      typed_hits += oa.ending == heuresis::search::Ending::target;
      plain_hits += ob.ending == heuresis::search::Ending::target;
      typed_evals += oa.evaluations;
      plain_evals += ob.evaluations;
    }
    s.require("typed by dimension, GP finds the law on " + std::to_string(typed_hits) + " of 10 seeds within 20 generations (untyped: " + std::to_string(plain_hits) + "; evaluations " +
                  std::to_string(static_cast<long>(typed_evals)) + " against " + std::to_string(static_cast<long>(plain_evals)) + ")",
              typed_hits >= plain_hits);
  }

  // A law that the initial population cannot hold (its terms have depth ≤ 2, the law needs 3): E = ½ m v² + m g h.
  {
    Signature law(3);
    for (auto p : {library::add(), library::subtract(), library::multiply(), library::divide()}) law.add(p);
    law.add(Variable{"m", of(dim::mass)});
    law.add(Variable{"v", of(dim::velocity)});
    law.add(Variable{"g", of(dim::acceleration)});
    law.add(Variable{"h", of(dim::length)});
    law.add(Constant{"2", dimensionless(), 2});
    Cases cases;
    for (int i = 0; i < 30; i++) {
      const double m = 0.5 + 0.1 * i, v = 0.3 + 0.17 * ((i * 11) % 30), g = 1 + 0.3 * ((i * 7) % 30), h = 0.2 + 0.13 * ((i * 13) % 30);
      cases.inputs.push_back({m, v, g, h});
      cases.targets.push_back(0.5 * m * v * v + m * g * h);
    }
    double ymax = 0;
    for (double y : cases.targets) ymax = std::max(ymax, y);
    GpOptions o;
    o.target = of(dim::energy);
    o.population = 500;
    o.min_depth = 1;
    o.max_depth = 2;  // too shallow to hold the law: it must be bred
    o.tournament = 5;
    GeneticProgramming gp(law, o, Draws(12));
    const auto out = heuresis::search::run(gp, [&](const Tree& t) { return regression(law, t, cases); }, heuresis::search::Budget{500 * 60, INFINITY, 1e-22 * ymax * ymax});
    const bool found = out.best && std::sqrt(out.best->evaluation.value()) <= 1e-11 * ymax;
    s.require("GP evolves E = ½ m v² + m g h exactly: " + (out.best ? to_string(law, out.best->genotype) : std::string("none")) + " in generation " + std::to_string(static_cast<long>(out.evaluations) / 500), found && out.evaluations > 500);
  }

  // Generational order-independence: telling in reverse gives the same populations.
  {
    Signature u;
    u.add(library::add());
    u.add(library::multiply());
    u.add(Variable{"x", dimensionless()});
    u.add(Constant{"1", dimensionless(), 1});
    Cases c;
    for (int i = 0; i < 10; i++) { c.inputs.push_back({0.1 * i}); c.targets.push_back(std::pow(0.1 * i, 3) + 0.1 * i); }
    GpOptions o;
    o.population = 60;
    GeneticProgramming a(u, o, Draws(9)), b(u, o, Draws(9));
    for (int gen = 0; gen < 6; gen++) {
      auto ca = a.ask(heuresis::search::unlimited), cb = b.ask(heuresis::search::unlimited);
      std::vector<heuresis::search::Told> ta, tb;
      for (auto& x : ca) ta.push_back({x.ticket, regression(u, x.genotype, c)});
      for (auto& x : cb) tb.push_back({x.ticket, regression(u, x.genotype, c)});
      std::reverse(tb.begin(), tb.end());
      a.tell(ta);
      b.tell(tb);
    }
    bool same = a.population().size() == b.population().size();
    for (std::size_t i = 0; same && i < a.population().size(); i++) same = a.population()[i].ticket == b.population()[i].ticket && a.population()[i].genotype.size() == b.population()[i].genotype.size();
    s.require("telling in reverse gives the same populations after six generations", same && a.generation() == 6);
  }

  return s.done();
}
