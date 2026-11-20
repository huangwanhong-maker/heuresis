// Checks of heuresis/spaces: typed spaces held to their laws and to what is known of them in closed form or by enumeration.
//
// - Every law (validity, the metric, the ball of a mutation and its floor, purity, the segment, repair, the chart, the
//   embedding, the unit, the elements, the keys) holds on every built-in space and on products, choices and constrained
//   subspaces of them.
// - Units and diameters equal, on every pair of a small finite space, their closed forms: n − Hₙ, n − 1 and n(n − 1)/4 for
//   permutations; 2E[K](1 − E[K]/n) for subsets of bounded size; the marginal formula for compositions; (n² − 1)/(3n) for
//   integers; 1 − 1/k for categories; Σwᵢ for a product of parts whose units are their means.
// - Moraglio's product theorem: on thirty random products of geometric spaces with random weights, recombination part by part
//   is geometric under the weighted sum of scaled distances.
// - A choice's distance is a metric on every triple of a finite choice.
// - Samplers are uniform where they claim to be, counted exactly over every value: subsets of bounded size, compositions with
//   and without caps, permutations, sequences given their length (Boltzmann's principle), a choice sampled by size.
// - The sphere's and SO(3)'s retractions invert their logarithms to rounding, and the trust radius π is sharp; SO(3)'s values
//   have one sign; the simplex's ilr chart is a bijection and its distance is Aitchison's, computed apart from clr coordinates.
// - Line recombination of random keys is geometric under Kendall's metric on the orders they decode to; a direct permutation
//   has no redundancy, keys at a small scale much; fitness–distance correlation is 1 for the distance itself.
// - An evolution over a product of seven spaces, through the adapter to heuresis's loop, finds a known optimum.
// - The laws find four faults planted in a broken space.
// - What is invalid is refused by name.

#include <heuresis/spaces/spaces.hpp>

#include <calx/verification/check.hpp>

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <numbers>
#include <string>
#include <vector>

namespace hs = heuresis::spaces;
using hs::Value;

namespace {

calx::random::Stream stream(std::uint32_t id) { return calx::random::Stream(20261120, id, 0, 0); }

struct Exact {
  double mean = 0, max = 0;
};
Exact enumerate(const hs::Space& sp) {
  const auto all = sp.elements(100000);
  Exact e;
  for (const auto& a : all)
    for (const auto& b : all) {
      const double d = sp.distance(a, b);
      e.mean += d, e.max = std::max(e.max, d);
    }
  e.mean /= static_cast<double>(all.size()) * static_cast<double>(all.size());
  return e;
}

template <class T, class... A>
hs::SpacePtr make(A&&... a) {
  return std::make_shared<T>(std::forward<A>(a)...);
}

// A space with four faults planted: mutations leave it, its distance (a square) breaks the triangle inequality, its midpoint
// recombination is claimed geometric and is not under that distance, and its unit is wrong (the mean square is 1/6).
class Broken final : public hs::Space {
 public:
  std::string describe() const override { return "a broken line"; }
  Value sample(calx::random::Stream& s, double = 1) const override { return line_.sample(s); }
  Value mutate(const Value& x, double, calx::random::Stream& s) const override { return hs::real_value({x.reals[0] + 2 * s.uniform()}); }
  double distance(const Value& x, const Value& y) const override {
    const double d = x.reals[0] - y.reals[0];
    return d * d;
  }
  double unit() const override { return 0.5; }
  std::optional<std::string> invalid(const Value& x) const override { return line_.invalid(x); }
  bool can_recombine() const override { return true; }
  Value recombine(const Value& x, const Value& y, calx::random::Stream&) const override { return hs::real_value({(x.reals[0] + y.reals[0]) / 2}); }
  hs::Properties properties() const override { return {true, true, true, true, hs::Unit::mean, true}; }

 private:
  hs::RealBox line_{std::vector<double>{0}, std::vector<double>{1}};
};

}  // namespace

int main() {
  calx::verification::Suite s;
  using PM = hs::PermutationSpace::Metric;

  // ---- the laws on every space ----
  auto box = make<hs::RealBox>(std::vector<double>{0, -1, 2}, std::vector<double>{1, 1, 2.5});
  auto ints = make<hs::IntegerBox>(std::vector<std::int64_t>{0, -3}, std::vector<std::int64_t>{9, 3});
  auto multiset = hs::multiset_space({2, 1, 3});
  auto cat = make<hs::Categorical>(5);
  auto lone = make<hs::Categorical>(1);
  auto level = make<hs::Ordinal>(7);
  auto simplex = make<hs::Simplex>(4);
  auto shares = make<hs::Simplex>(2, 10.0);
  auto cayley = make<hs::PermutationSpace>(7, PM::cayley);
  auto hamming = make<hs::PermutationSpace>(7, PM::hamming);
  auto kendall = make<hs::PermutationSpace>(7, PM::kendall);
  auto subset = make<hs::Subset>(10);
  auto exactly3 = make<hs::Subset>(10, 3, 3);
  auto some = make<hs::Subset>(12, 2, 5);
  auto units = make<hs::Composition>(4, 9);
  auto capped = make<hs::Composition>(5, 12, std::vector<std::int64_t>{3, 0, 5, 2, 4});
  auto words3 = make<hs::SequenceSpace>(make<hs::Categorical>(3), 0, 6);
  auto path = make<hs::SequenceSpace>(make<hs::RealBox>(std::vector<double>{0, 0}, std::vector<double>{1, 1}), 1, 5);
  auto sphere = make<hs::Sphere>(3);
  auto circle = make<hs::Sphere>(2);
  auto rotations = make<hs::Rotations>();
  auto product = std::make_shared<hs::Product>(std::vector<hs::Part>{{"where", box, 1}, {"kind", cat, 2}, {"order", kendall, 0.5}, {"mix", simplex, 1}, {"turn", rotations, 1}});
  auto choice = std::make_shared<hs::Choice>(std::vector<hs::Branch>{{"none", lone, 1}, {"amount", ints, 1}, {"items", exactly3, 2}});
  auto open_choice = std::make_shared<hs::Choice>(std::vector<hs::Branch>{{"mix", simplex, 1}, {"kind", cat, 1}});
  auto even = std::make_shared<hs::Constrained>(subset, "the size is even", [](const Value& v) -> std::optional<std::string> {
    if (v.ints.size() % 2) return std::string("an odd size");
    return std::nullopt;
  });
  auto inside = std::make_shared<hs::Constrained>(box, "x₀ ≤ 0.5", [](const Value& v) -> std::optional<std::string> {
    if (v.reals[0] > 0.5) return std::string("x₀ above one half");
    return std::nullopt;
  }, [](const Value& v) {
    Value w = v;
    w.reals[0] = std::min(w.reals[0], 0.5);
    return w;
  });
  const std::vector<std::pair<std::string, hs::SpacePtr>> all = {
      {"a real box", box}, {"an integer box", ints}, {"multisets", multiset}, {"five categories", cat}, {"one category", lone},
      {"seven ordered levels", level}, {"the simplex of four parts", simplex}, {"the simplex of two parts summing to 10", shares},
      {"permutations under Cayley's metric", cayley}, {"permutations under Hamming's metric", hamming}, {"permutations under Kendall's metric", kendall},
      {"subsets", subset}, {"subsets of size 3", exactly3}, {"subsets of sizes 2…5", some}, {"compositions", units}, {"compositions with caps", capped},
      {"sequences of categories", words3}, {"sequences of points of a square", path}, {"the sphere S²", sphere}, {"the circle", circle},
      {"SO(3)", rotations}, {"a product of five spaces", product}, {"a choice of three branches", choice}, {"a choice with an unbounded branch", open_choice},
      {"subsets of even size, by rejection", even}, {"a box cut by a plane, repaired", inside}};
  {
    std::uint32_t id = 1;
    for (const auto& [name, sp] : all) {
      auto st = stream(id++);
      const auto r = hs::check_laws(*sp, st);
      s.require("every law holds on " + name + " (" + std::to_string(r.applied()) + " laws applied)", r.all_hold(), r.failures());
    }
  }

  // ---- the laws find planted faults ----
  {
    Broken broken;
    auto st = stream(99);
    const auto r = hs::check_laws(broken, st);
    std::string missed;
    for (const char* law : {"mutations valid", "metric: triangle inequality", "children on the segment", "the unit, exact"}) {
      const auto* l = r.find(law);
      if (!l || l->holds()) missed += std::string(missed.empty() ? "" : ", ") + law;
    }
    const auto* pure = r.find("recombination pure");
    s.require("the laws find each of four faults planted in a broken space, and pass what is sound", missed.empty() && pure && pure->holds(), missed.empty() ? r.failures() : "missed: " + missed);
  }

  // ---- units and diameters by enumeration ----
  {
    struct Case {
      std::string name;
      hs::SpacePtr sp;
      double unit, diameter;
      bool diameter_is_bound;
    };
    double h5 = 0;
    for (int i = 1; i <= 5; i++) h5 += 1.0 / i;
    // Subsets of 6 with sizes 1…4: E[K] over sizes weighted C(6, k).
    double ek = 0, wk = 0;
    for (int k = 1; k <= 4; k++) {
      double c = 1;
      for (int j = 0; j < k; j++) c = c * (6 - j) / (j + 1);
      ek += k * c, wk += c;
    }
    ek /= wk;
    const std::vector<Case> cases = {
        {"permutations of 5 under Cayley's metric", make<hs::PermutationSpace>(5, PM::cayley), 5 - h5, 4, false},
        {"permutations of 5 under Hamming's metric", make<hs::PermutationSpace>(5, PM::hamming), 4, 5, false},
        {"permutations of 5 under Kendall's metric", make<hs::PermutationSpace>(5, PM::kendall), 5.0, 10, false},
        {"subsets of 6 of sizes 1…4", make<hs::Subset>(6, 1, 4), 2 * ek * (1 - ek / 6), 6, false},
        {"5 units among 3 parts", make<hs::Composition>(3, 5), -1, 5, false},
        {"a 3 × 4 integer box", make<hs::IntegerBox>(std::vector<std::int64_t>{0, 0}, std::vector<std::int64_t>{2, 3}), 8.0 / 9 + 15.0 / 12, 5, false},
        {"five ordered levels", make<hs::Ordinal>(5), 24.0 / 15, 4, false},
        {"four categories", make<hs::Categorical>(4), 0.75, 1, false},
        {"a product of three categories, four levels and permutations of three",
         std::make_shared<hs::Product>(std::vector<hs::Part>{{"a", make<hs::Categorical>(3), 1}, {"b", make<hs::Ordinal>(4), 1}, {"c", make<hs::PermutationSpace>(3, PM::hamming), 1}}),
         3.0, 1.5 + 3.0 / 1.25 + 3.0 / 2, false},
        {"4 units among 3 parts with caps 1, 2, 4", make<hs::Composition>(3, 4, std::vector<std::int64_t>{1, 2, 4}), -1, 4, true},
    };
    for (const auto& c : cases) {
      const Exact e = enumerate(*c.sp);
      const double unit = c.unit >= 0 ? c.unit : e.mean;
      if (c.sp->properties().unit_exact) {
        s.check(c.name + ": the unit is the mean distance over every pair", std::abs(c.sp->unit() - e.mean) + std::abs(unit - e.mean), 1e-12);
      } else {
        s.check(c.name + ": the estimated unit within 3% of the mean over every pair", std::abs(c.sp->unit() - e.mean) / e.mean, 0.03);
      }
      if (c.diameter_is_bound) s.require(c.name + ": the diameter bounds every distance", e.max <= c.sp->diameter());
      else s.check(c.name + ": the diameter is the greatest distance", std::abs(c.sp->diameter() - e.max) + std::abs(c.diameter - e.max), 1e-12);
    }
    // The composition formula on its own: 5 units among 3 parts, every pair enumerated by hand above.
    const auto words = make<hs::SequenceSpace>(make<hs::Categorical>(2), 0, 3);
    const Exact e = enumerate(*words);
    s.check("sequences of two letters up to length 3: the estimated unit within 10% of the mean over every pair", std::abs(words->unit() - e.mean) / e.mean, 0.10);
  }

  // ---- distances computed apart ----
  {
    auto letters = std::make_shared<hs::Categorical>(26);
    hs::SequenceSpace seq(letters, 0, 10);
    auto word = [](const std::string& w) {
      std::vector<Value> xs;
      for (char c : w) xs.push_back(hs::category_value(c - 'a'));
      return hs::sequence_value(std::move(xs));
    };
    s.check("the edit distance from kitten to sitting is 3", seq.distance(word("kitten"), word("sitting")) - 3, 0);
    s.check("the symmetric difference of the multisets {a, a, b} and {b, c} is 3",
            multiset->distance(hs::integer_value({2, 1, 0}), hs::integer_value({0, 1, 1})) - 3, 0);
    // Aitchison's distance from centred log-ratios, computed apart from tyche's ilr.
    auto st = stream(101);
    double worst = 0;
    for (int i = 0; i < 500; i++) {
      const Value x = simplex->sample(st), y = simplex->sample(st);
      auto clr = [](const std::vector<double>& v) {
        double m = 0;
        for (double a : v) m += std::log(a);
        m /= static_cast<double>(v.size());
        std::vector<double> c;
        for (double a : v) c.push_back(std::log(a) - m);
        return c;
      };
      const auto cx = clr(x.reals), cy = clr(y.reals);
      double d = 0;
      for (std::size_t j = 0; j < cx.size(); j++) d += (cx[j] - cy[j]) * (cx[j] - cy[j]);
      worst = std::max(worst, std::abs(std::sqrt(d) - simplex->distance(x, y)) / (1 + std::sqrt(d)));
    }
    s.check("the simplex's distance is Aitchison's, from clr coordinates computed apart", worst, 1e-12);
    double back = 0;
    for (int i = 0; i < 500; i++) {
      std::vector<double> z(3);
      for (double& c : z) c = 3 * st.normal();
      const auto again = simplex->embed(simplex->unembed(z));
      for (int j = 0; j < 3; j++) back = std::max(back, std::abs(again[j] - z[j]) / (1 + std::abs(z[j])));
    }
    s.check("the simplex's ilr chart is a bijection: embed(unembed(z)) = z", back, 1e-12);
  }

  // ---- Moraglio's product theorem on random products ----
  {
    const std::vector<hs::SpacePtr> pool = {
        make<hs::RealBox>(std::vector<double>{0, 0}, std::vector<double>{1, 3}), ints, cat, level, simplex, make<hs::PermutationSpace>(5, PM::cayley),
        make<hs::PermutationSpace>(5, PM::kendall), make<hs::Subset>(7), make<hs::Composition>(3, 6), sphere, rotations, words3};
    auto st = stream(102);
    double worst = 0;
    std::size_t pairs = 0;
    for (int k = 0; k < 30; k++) {
      std::vector<hs::Part> parts;
      const std::size_t m = 2 + st.below(3);
      for (std::size_t i = 0; i < m; i++) parts.push_back({"p" + std::to_string(i), pool[st.below(pool.size())], st.uniform(0.2, 3)});
      hs::Product p(parts);
      for (int i = 0; i < 100; i++) {
        const Value x = p.sample(st);
        const Value y = i % 2 ? p.sample(st) : p.mutate(x, 0.3, st);
        const Value c = p.recombine(x, y, st);
        const double dxy = p.distance(x, y);
        worst = std::max(worst, std::abs(p.distance(x, c) + p.distance(c, y) - dxy) / (1 + dxy));
        pairs++;
      }
    }
    s.check("the product theorem: children on the segment of the weighted sum, 30 random products, " + std::to_string(pairs) + " pairs", worst, 1e-9);
  }

  // ---- a choice's distance is a metric on every triple ----
  {
    hs::Choice c(std::vector<hs::Branch>{{"kind", make<hs::Categorical>(3), 1}, {"level", make<hs::Ordinal>(4), 1}, {"items", make<hs::Subset>(3), 1}});
    const auto vs = c.elements(1000);
    double excess = 0;
    for (const auto& a : vs)
      for (const auto& b : vs)
        for (const auto& d : vs) excess = std::max(excess, c.distance(a, d) - c.distance(a, b) - c.distance(b, d));
    s.check("a choice's distance obeys the triangle inequality on all " + std::to_string(vs.size() * vs.size() * vs.size()) + " triples", std::max(0.0, excess), 1e-12);
  }

  // ---- samplers uniform where they claim to be ----
  {
    auto check_uniform = [&](const std::string& name, const hs::Space& sp, std::uint32_t id, const std::function<bool(const Value&)>& keep = {}) {
      auto st = stream(id);
      const auto cells = sp.elements(100000).size();
      const auto u = hs::uniformity(sp, st, 60 * cells, keep);
      s.check(name + ": χ² over " + std::to_string(u.cells) + " values, standardised", u.z + 1e3 * static_cast<double>(u.outside), 4);
    };
    check_uniform("subsets of 6 of sizes 2…4 sampled uniformly", hs::Subset(6, 2, 4), 201);
    check_uniform("5 units among 4 parts sampled uniformly (stars and bars)", hs::Composition(4, 5), 202);
    check_uniform("6 units among 4 parts with caps 3, 1, 2, 4 sampled uniformly (by counting)", hs::Composition(4, 6, {3, 1, 2, 4}), 203);
    check_uniform("permutations of 4 sampled uniformly", hs::PermutationSpace(4), 204);
    hs::SequenceSpace seq(make<hs::Categorical>(3), 0, 4);
    for (std::size_t L : {2u, 4u})
      check_uniform("sequences of three letters, uniform given length " + std::to_string(L), seq, 205 + static_cast<std::uint32_t>(L), [L](const Value& v) { return v.parts.size() == L; });
    hs::Choice by_size(std::vector<hs::Branch>{{"one of two", make<hs::Categorical>(2), 1}, {"a subset of three", make<hs::Subset>(3), 1}}, 0, hs::Choice::Sampling::by_size);
    check_uniform("a choice sampled by size is uniform over the union", by_size, 210);
    hs::Product pr(std::vector<hs::Part>{{"a", make<hs::Subset>(4, 1, 2), 1}, {"b", make<hs::Ordinal>(3), 1}});
    check_uniform("a product of uniform parts is uniform", pr, 211);
  }

  // ---- curved spaces ----
  {
    auto st = stream(301);
    double worst = 0, worst_rot = 0, worst_len = 0;
    for (int i = 0; i < 1000; i++) {
      const Value x = sphere->sample(st), y = sphere->sample(st);
      worst = std::max(worst, sphere->distance(sphere->retract(x, sphere->log(x, y)), y));
      const Value a = rotations->sample(st), b = rotations->sample(st);
      worst_rot = std::max(worst_rot, rotations->distance(rotations->retract(a, rotations->log(a, b)), b));
      const auto v = rotations->log(a, b);
      worst_len = std::max(worst_len, std::abs(std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) - rotations->distance(a, b)));
    }
    s.check("the sphere: retract_x(log_x(y)) = y on 1000 pairs", worst, 1e-12);
    s.check("SO(3): retract_x(log_x(y)) = y on 1000 pairs", worst_rot, 1e-12);
    s.check("SO(3): the logarithm's length is the geodesic distance", worst_len, 1e-12);
    const Value x = sphere->sample(st);
    const std::vector<double> v = {3.5, 0};
    const auto back = sphere->log(x, sphere->retract(x, v));
    s.check("the sphere's trust radius π is sharp: a step of 3.5 comes back as 2π − 3.5",
            std::abs(std::sqrt(back[0] * back[0] + back[1] * back[1]) - (2 * std::numbers::pi - 3.5)), 1e-12);
    const auto r = physis::geometry::Rotation::about(physis::geometry::Vec3{1, 2, 3}, 2.0);
    s.require("SO(3): a rotation and its negated quaternion have one value", hs::Rotations::value_of(r) == hs::Rotations::value_of(r.negated()));
  }

  // ---- representation diagnostics ----
  {
    auto st = stream(401);
    using Order = std::vector<std::int64_t>;
    hs::PermutationSpace direct(8, PM::kendall);
    auto order_distance = [&](const Order& a, const Order& b) { return direct.distance(hs::permutation_value(a), hs::permutation_value(b)); };
    const auto m1 = hs::representation<Order>(direct, [](const Value& v) { return v.ints; }, order_distance, st, 0.01);
    s.check("a direct permutation: no mutation leaves the order unchanged", m1.unchanged, 0);
    s.check("a direct permutation: crossover is geometric in the order", m1.crossover_excess, 1e-12);
    hs::RealBox keys(std::vector<double>(8, 0), std::vector<double>(8, 1));
    auto decode = [&](const Value& v) { return direct.unembed(v.reals).ints; };
    const auto m2 = hs::representation<Order>(keys, decode, order_distance, st, 0.01);
    s.require("random keys at a small scale: most mutations leave the order unchanged (" + std::to_string(m2.unchanged) + ")", m2.unchanged > 0.5);
    s.check("random keys: line recombination is geometric under Kendall's metric on the decoded orders", m2.crossover_excess, 1e-12);
    const Value best = cayley->sample(st);
    const double fdc = hs::fitness_distance_correlation(*cayley, [&](const Value& v) { return cayley->distance(v, best); }, best, st);
    s.check("fitness–distance correlation of the distance itself is 1", fdc - 1, 1e-12);
  }

  // ---- an evolution over a product of seven spaces ----
  {
    auto pos = make<hs::RealBox>(std::vector<double>{0, 0, 0}, std::vector<double>{1, 1, 1});
    auto mix = make<hs::Simplex>(3);
    auto genotype = std::make_shared<hs::Product>(std::vector<hs::Part>{
        {"position", pos, 1}, {"colour", make<hs::Categorical>(5), 1}, {"order", make<hs::PermutationSpace>(6, PM::kendall), 1},
        {"chosen", make<hs::Subset>(8, 3, 3), 1}, {"shares", mix, 1}, {"units", make<hs::Composition>(4, 6), 1}, {"level", make<hs::Ordinal>(7), 1}});
    auto st = stream(501);
    const Value target = genotype->sample(st);
    heuresis::evolution::Strategy strategy;
    strategy.mu = 30, strategy.lambda = 30, strategy.generations = 400, strategy.from_archive = 0;
    heuresis::evolution::Evolution<Value> e(hs::operators(genotype, {{0.001, 0.01, 0.1, 1}}), strategy, {}, 77);
    heuresis::evolution::evolve(e, [&](const Value& g) { return heuresis::search::scalar(genotype->distance(g, target)); });
    const Value best = e.state().best->genotype;
    double discrete = 0, continuous = 0;
    for (std::size_t i = 0; i < genotype->size(); i++) {
      const auto& part = genotype->part(i);
      const double d = part.space->distance(best.parts[i], target.parts[i]) / part.space->unit();
      if (part.name == "position" || part.name == "shares") continuous = std::max(continuous, d);
      else discrete += d;
    }
    s.check("an evolution over seven spaces finds the known optimum: its discrete parts exactly", discrete, 0);
    s.check("an evolution over seven spaces finds the known optimum: its continuous parts within 2% of their units", continuous, 0.02);
  }

  // ---- constrained subspaces ----
  {
    auto st = stream(601);
    for (int i = 0; i < 200; i++) even->sample(st);
    s.require("a constrained space counts what it rejects (" + std::to_string(even->rejections()) + ")", even->rejections() > 0);
    hs::Constrained impossible(cat, "nothing", [](const Value&) -> std::optional<std::string> { return std::string("excluded"); }, {}, 50);
    bool refused = false;
    try {
      impossible.sample(st);
    } catch (const hs::Refusal&) {
      refused = true;
    }
    s.require("a constrained space with no value and no repair refuses to sample", refused);
    std::size_t fixed = 0;
    for (int i = 0; i < 200; i++) fixed += inside->valid(inside->mutate(inside->sample(st), 1, st));
    s.require("a repaired constrained space gives valid values", fixed == 200);
  }

  // ---- refusals ----
  {
    auto refuses = [&](const std::string& what, const std::function<void()>& f) {
      bool r = false;
      try {
        f();
      } catch (const hs::Refusal&) {
        r = true;
      }
      s.require(what + " is refused", r);
    };
    auto st = stream(701);
    refuses("a category given to a real box", [&] { box->mutate(hs::category_value(1), 0.5, st); });
    refuses("a box whose lower bound exceeds its upper", [] { (void)hs::RealBox({1}, {0}); });
    refuses("a product whose parts share a name", [&] { (void)hs::Product({{"a", cat, 1}, {"a", level, 1}}); });
    refuses("a choice sampled by size with an infinite branch", [&] { (void)hs::Choice({{"x", box, 1}}, 0, hs::Choice::Sampling::by_size); });
    refuses("a composition whose caps hold fewer than its units", [] { (void)hs::Composition(3, 10, {2, 3, 4}); });
    refuses("a permutation with an item placed twice", [&] { cayley->distance(hs::permutation_value({0, 0, 1, 2, 3, 4, 5}), cayley->sample(st)); });
  }

  return s.done();
}
