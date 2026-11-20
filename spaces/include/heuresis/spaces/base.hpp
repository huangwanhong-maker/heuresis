// The base spaces: real boxes, integer boxes (and multisets), categories, ordered categories, the simplex in Aitchison's
// geometry, permutations under a stated metric, subsets, compositions of identical units, and sequences of any space's
// values under an edit distance. Each states its metric, its unit (a closed form where one exists, else an estimate it
// says is one), the ball its mutation draws from, and whether its recombination is geometric.
//
// Sources, by space:
// - Real boxes. Gaussian mutation with per-coordinate step σᵢ = scale·wᵢ/√6, so that at scale 1 the root-mean-square
//   step equals the unit √(Σwᵢ²/6), the root-mean-square distance between two uniform points (E(X − Y)² = w²/6 for X,
//   Y uniform on an interval of width w); reflection into the box. Line recombination: a uniform point of the segment,
//   the metric segment of a strictly convex norm (Moraglio 2007, §5).
// - Integer boxes. Rudolph's two-sided geometric step per coordinate (Rudolph 1994, PPSN III, §3), its mean absolute
//   step scale times the coordinate's unit (n² − 1)/(3n), the mean |X − Y| of two uniform integers among n; reflection
//   into the box; a step that leaves the value unchanged is replaced by a unit move, so that a mutation always moves.
//   Box recombination: each coordinate uniform between the parents', the segment of the ℓ¹ metric (Moraglio 2007, §5).
//   A multiset of n kinds with multiplicities bounded is the same space: its symmetric difference is the ℓ¹ distance.
// - Categories. The discrete metric; its unit 1 − 1/k. Ordered categories: the path metric |i − j|, as an integer box
//   of one coordinate.
// - The simplex. Positive compositions of a total T in Aitchison's geometry (Aitchison 1986, The Statistical Analysis of
//   Compositional Data; Egozcue et al. 2003, Math. Geol. 35, 279–300): the distance is the Euclidean distance between
//   isometric log-ratio coordinates (tyche's `compositions`), a global chart; uniform sampling by Dirichlet(1, …, 1)
//   (normalised exponentials); the unit π√((D − 1)/3), exactly the root-mean-square Aitchison distance between two
//   uniform compositions, since each centred log-ratio coordinate of a uniform composition is log E − mean log E with
//   Var(log E) = π²/6 for E exponential. Recombination along the Aitchison geodesic, a line in ilr coordinates.
// - Permutations. A value lists the item at each position. Under Cayley's metric (transpositions; unit n − Hₙ, since a
//   uniform permutation has Hₙ cycles on average), Hamming's (positions that differ; unit n − 1, since it has one fixed
//   point on average) or Kendall's (adjacent transpositions; unit n(n − 1)/4). Mutation by moves of the metric's
//   generators. Under Cayley's and Hamming's metric, cycle crossover with each cycle taken from either parent at random
//   is geometric: a child takes whole cycles of the parents' relative permutation, so its distances to them add up
//   (Moraglio and Poli 2005, "Topological crossover for the permutation representation", GECCO workshops); under
//   Kendall's, a child is a random point of a bubble-sort path from one parent to the other, each step an adjacent
//   transposition that removes one inversion relative to the other parent.
// - Subsets. The symmetric difference |A Δ B|; uniform sampling among the subsets whose size lies in the stated bounds
//   (the size drawn with weights C(n, k), then a uniform subset of that size by Floyd's algorithm); recombination
//   uniform between A ∩ B and A ∪ B with its count drawn exactly within the bounds: the segment of the symmetric
//   difference (Moraglio 2007, §5).
// - Compositions. N identical units among D parts with optional caps; the transport distance ½Σ|xᵢ − yᵢ| (units
//   moved); uniform sampling (by stars and bars without caps, by counting with them); recombination between the
//   parents' minima and maxima with the remaining units drawn hypergeometrically, the segment of the ℓ¹ metric on the
//   hyperplane Σxᵢ = N.
// - Sequences. Values of an element space E of bounded length; the edit distance with unit insertions and deletions and
//   substitution cost d_E/diam(E) (or min(1, d_E/unit(E)) where E is unbounded), a metric as the shortest-path metric of
//   a metric cost; sampling with the length uniform up to a size and each element from E, uniform given the length when
//   E's sampler is uniform (Boltzmann's principle, Duchon, Flajolet, Louchard and Schaeffer 2004, Combin. Probab.
//   Comput. 13, 577–625); homologous recombination along an optimal alignment, each column from either parent, geometric
//   since the columns' costs split between the two distances.
//
// Every operator first checks its arguments with `invalid` and refuses an invalid one by name.

#pragma once

#include <heuresis/spaces/space.hpp>
#include <heuresis/variation/distance.hpp>
#include <heuresis/variation/integer.hpp>
#include <heuresis/variation/permutation.hpp>

#include <tyche/compositions/balances.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <numeric>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace heuresis::spaces {

namespace detail {

inline void require(const Space& sp, const Value& x, const char* who) {
  if (auto why = sp.invalid(x)) throw Refusal(std::string(who) + " on " + sp.describe() + ": " + *why);
}

// The number of elementary moves at a scale, for a space whose unit is `reach` moves: 1 + Binomial(2m, ½) with
// m = round(max(0, scale·reach − 1)), mean about scale·reach, at least one and at most move_bound.
inline std::size_t move_bound(double scale, double reach) {
  const double mu = std::max(1.0, clamp_scale(scale) * reach);
  return 1 + 2 * static_cast<std::size_t>(std::llround(mu - 1));
}
inline std::size_t draw_moves(double scale, double reach, calx::random::Stream& s) {
  const std::size_t b = move_bound(scale, reach);
  return b == 1 ? 1 : 1 + static_cast<std::size_t>(s.binomial(static_cast<std::int64_t>(b - 1), 0.5));
}

inline double reflect(double x, double l, double h) {
  const double w = h - l;
  if (!(w > 0)) return l;
  double t = std::fmod(x - l, 2 * w);
  if (t < 0) t += 2 * w;
  if (t > w) t = 2 * w - t;
  return l + t;
}
inline std::int64_t reflect(std::int64_t x, std::int64_t l, std::int64_t h) {
  const std::int64_t w = h - l;
  if (w <= 0) return l;
  std::int64_t t = (x - l) % (2 * w);
  if (t < 0) t += 2 * w;
  if (t > w) t = 2 * w - t;
  return l + t;
}

// A uniform k-subset of {0, …, n − 1}, sorted (Floyd's algorithm; Bentley and Floyd 1987, CACM 30, 754–757).
inline std::vector<std::int64_t> floyd(std::size_t n, std::size_t k, calx::random::Stream& s) {
  std::set<std::size_t> chosen;
  for (std::size_t j = n - k; j < n; j++) {
    const std::size_t t = static_cast<std::size_t>(s.below(j + 1));
    if (!chosen.insert(t).second) chosen.insert(j);
  }
  return {chosen.begin(), chosen.end()};
}

// An index drawn with probability proportional to its weight (all non-negative, not all zero).
inline std::size_t pick(const std::vector<double>& w, calx::random::Stream& s) {
  double total = 0;
  for (double v : w) total += v;
  double u = s.uniform() * total;
  for (std::size_t i = 0; i < w.size(); i++) {
    if (u < w[i]) return i;
    u -= w[i];
  }
  for (std::size_t i = w.size(); i-- > 0;)
    if (w[i] > 0) return i;
  return 0;
}

// Weights proportional to C(m, j) for j in [a, b], computed in logarithms and scaled so that the largest is 1.
inline std::vector<double> binomial_weights(std::size_t m, std::size_t a, std::size_t b) {
  std::vector<double> lw(b - a + 1);
  double l = 0;  // log C(m, 0)
  for (std::size_t j = 0; j <= b; j++) {
    if (j >= a) lw[j - a] = l;
    if (j < m) l += std::log(static_cast<double>(m - j)) - std::log(static_cast<double>(j + 1));
  }
  const double top = *std::max_element(lw.begin(), lw.end());
  for (double& v : lw) v = std::exp(v - top);
  return lw;
}

// Exact binomial coefficient, or nullopt where it exceeds 2⁶².
inline std::optional<std::uint64_t> binomial_exact(std::uint64_t n, std::uint64_t k) {
  if (k > n) return 0;
  k = std::min(k, n - k);
  unsigned __int128 c = 1;
  for (std::uint64_t i = 1; i <= k; i++) {
    c = c * (n - k + i) / i;
    if (c > (static_cast<unsigned __int128>(1) << 62)) return std::nullopt;
  }
  return static_cast<std::uint64_t>(c);
}

inline variation::Permutation to_permutation(const std::vector<std::int64_t>& v) {
  variation::Permutation p(v.size());
  for (std::size_t i = 0; i < p.size(); i++) p[i] = static_cast<std::uint32_t>(v[i]);
  return p;
}
inline variation::Permutation to_permutation(const Value& v) { return to_permutation(v.ints); }

}  // namespace detail

// ---------------------------------------------------------------------------------------------------- real boxes ----

class RealBox final : public Space {
 public:
  RealBox(std::vector<double> lower, std::vector<double> upper) : l_(std::move(lower)), h_(std::move(upper)) {
    if (l_.size() != h_.size() || l_.empty()) throw Refusal("a real box needs as many lower as upper bounds, at least one");
    double s2 = 0;
    for (std::size_t i = 0; i < l_.size(); i++) {
      if (!std::isfinite(l_[i]) || !std::isfinite(h_[i]) || h_[i] < l_[i]) throw Refusal("a real box's bounds must be finite with lower ≤ upper");
      s2 += (h_[i] - l_[i]) * (h_[i] - l_[i]);
    }
    unit_ = std::sqrt(s2 / 6);
    diameter_ = std::sqrt(s2);
  }
  std::size_t dimension() const { return l_.size(); }
  const std::vector<double>& lower() const { return l_; }
  const std::vector<double>& upper() const { return h_; }
  std::string describe() const override { return "a box in ℝ^" + std::to_string(l_.size()); }

  Value sample(calx::random::Stream& s, double = 1) const override {
    std::vector<double> x(l_.size());
    for (std::size_t i = 0; i < x.size(); i++) x[i] = l_[i] + (h_[i] - l_[i]) * s.uniform();
    return real_value(std::move(x));
  }
  Value mutate(const Value& x, double scale, calx::random::Stream& s) const override {
    detail::require(*this, x, "mutate");
    const double c = detail::clamp_scale(scale) / std::sqrt(6.0);
    std::vector<double> y = x.reals;
    for (std::size_t i = 0; i < y.size(); i++) y[i] = detail::reflect(y[i] + c * (h_[i] - l_[i]) * s.normal(), l_[i], h_[i]);
    return real_value(std::move(y));
  }
  double distance(const Value& x, const Value& y) const override {
    detail::require(*this, x, "distance");
    detail::require(*this, y, "distance");
    return variation::euclidean(x.reals, y.reals);
  }
  double unit() const override { return unit_; }
  std::optional<std::string> invalid(const Value& x) const override {
    if (x.kind != Value::Kind::real) return std::string("a ") + name_of(x.kind) + " value, not a real vector";
    if (x.reals.size() != l_.size()) return "a vector of " + std::to_string(x.reals.size()) + " coordinates, not " + std::to_string(l_.size());
    for (std::size_t i = 0; i < l_.size(); i++)
      if (!(x.reals[i] >= l_[i] && x.reals[i] <= h_[i])) return "coordinate " + std::to_string(i) + " outside [" + std::to_string(l_[i]) + ", " + std::to_string(h_[i]) + "]";
    return std::nullopt;
  }

  bool can_recombine() const override { return true; }
  Value recombine(const Value& x, const Value& y, calx::random::Stream& s) const override {
    detail::require(*this, x, "recombine");
    detail::require(*this, y, "recombine");
    const double t = s.uniform();
    std::vector<double> c(l_.size());
    for (std::size_t i = 0; i < c.size(); i++) c[i] = x.reals[i] == y.reals[i] ? x.reals[i] : std::clamp(x.reals[i] + t * (y.reals[i] - x.reals[i]), l_[i], h_[i]);
    return real_value(std::move(c));
  }
  Properties properties() const override { return {true, true, true, true, Unit::rms, true}; }
  bool can_repair() const override { return true; }
  Value repair(const Value& x) const override {
    if (valid(x)) return x;
    if (x.kind != Value::Kind::real || x.reals.size() != l_.size()) throw Refusal("repair on " + describe() + ": a value of another shape");
    std::vector<double> y = x.reals;
    for (std::size_t i = 0; i < y.size(); i++) y[i] = std::isnan(y[i]) ? l_[i] : std::clamp(y[i], l_[i], h_[i]);
    return real_value(std::move(y));
  }
  double change_floor() const override { return diameter_ > 0 ? 1 : 0; }
  double diameter() const override { return diameter_; }

  bool has_chart() const override { return true; }
  std::size_t chart_dimension() const override { return l_.size(); }
  std::vector<double> log(const Value& x, const Value& y) const override {
    detail::require(*this, x, "log");
    detail::require(*this, y, "log");
    std::vector<double> v(l_.size());
    for (std::size_t i = 0; i < v.size(); i++) v[i] = y.reals[i] - x.reals[i];
    return v;
  }
  Value retract(const Value& x, std::span<const double> v) const override {
    detail::require(*this, x, "retract");
    if (v.size() != l_.size()) throw Refusal("retract on " + describe() + ": a tangent vector of another dimension");
    std::vector<double> y(l_.size());
    for (std::size_t i = 0; i < y.size(); i++) y[i] = detail::reflect(x.reals[i] + v[i], l_[i], h_[i]);
    return real_value(std::move(y));
  }
  double trust_radius(const Value& x) const override {
    detail::require(*this, x, "trust_radius");
    double r = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < l_.size(); i++) r = std::min({r, x.reals[i] - l_[i], h_[i] - x.reals[i]});
    return r;
  }
  std::optional<std::size_t> embedding_dimension() const override { return l_.size(); }
  std::vector<double> embed(const Value& x) const override {
    detail::require(*this, x, "embed");
    std::vector<double> z(l_.size());
    for (std::size_t i = 0; i < z.size(); i++) z[i] = h_[i] > l_[i] ? (x.reals[i] - l_[i]) / (h_[i] - l_[i]) : 0;
    return z;
  }
  Value unembed(std::span<const double> z) const override {
    if (z.size() != l_.size()) throw Refusal("unembed on " + describe() + ": coordinates of another dimension");
    std::vector<double> x(l_.size());
    for (std::size_t i = 0; i < x.size(); i++) {
      const double t = std::isnan(z[i]) ? 0 : std::clamp(z[i], 0.0, 1.0);
      x[i] = t == 1 ? h_[i] : l_[i] + t * (h_[i] - l_[i]);
    }
    return real_value(std::move(x));
  }

 private:
  std::vector<double> l_, h_;
  double unit_ = 0, diameter_ = 0;
};

// ------------------------------------------------------------------------------------------------- integer boxes ----

class IntegerBox final : public Space {
 public:
  IntegerBox(std::vector<std::int64_t> lower, std::vector<std::int64_t> upper, std::string what = {})
      : l_(std::move(lower)), h_(std::move(upper)), what_(std::move(what)) {
    if (l_.size() != h_.size() || l_.empty()) throw Refusal("an integer box needs as many lower as upper bounds, at least one");
    for (std::size_t i = 0; i < l_.size(); i++) {
      if (h_[i] < l_[i]) throw Refusal("an integer box's bounds must have lower ≤ upper");
      if (h_[i] - l_[i] > (std::int64_t{1} << 52)) throw Refusal("an integer box's coordinates must span at most 2⁵² values");
      const double n = static_cast<double>(h_[i] - l_[i] + 1);
      unit_ += (n * n - 1) / (3 * n);
      diameter_ += static_cast<double>(h_[i] - l_[i]);
      if (h_[i] > l_[i]) movable_ = true;
    }
  }
  std::size_t dimension() const { return l_.size(); }
  std::string describe() const override { return what_.empty() ? "a box in ℤ^" + std::to_string(l_.size()) : what_; }

  Value sample(calx::random::Stream& s, double = 1) const override {
    std::vector<std::int64_t> x(l_.size());
    for (std::size_t i = 0; i < x.size(); i++) x[i] = l_[i] + static_cast<std::int64_t>(s.below(static_cast<std::uint64_t>(h_[i] - l_[i] + 1)));
    return integer_value(std::move(x));
  }
  Value mutate(const Value& x, double scale, calx::random::Stream& s) const override {
    detail::require(*this, x, "mutate");
    std::vector<std::int64_t> y = x.ints;
    const double sc = detail::clamp_scale(scale);
    for (std::size_t i = 0; i < y.size(); i++) {
      if (h_[i] == l_[i]) continue;
      const double n = static_cast<double>(h_[i] - l_[i] + 1);
      const double p = variation::rudolph_p(sc * (n * n - 1) / (3 * n), 1);
      const auto g1 = static_cast<std::int64_t>(s.geometric(p)), g2 = static_cast<std::int64_t>(s.geometric(p));
      y[i] = detail::reflect(y[i] + g1 - g2, l_[i], h_[i]);
    }
    if (y == x.ints && movable_) unit_move(y, s);
    return integer_value(std::move(y));
  }
  double distance(const Value& x, const Value& y) const override {
    detail::require(*this, x, "distance");
    detail::require(*this, y, "distance");
    double d = 0;
    for (std::size_t i = 0; i < l_.size(); i++) d += static_cast<double>(x.ints[i] > y.ints[i] ? x.ints[i] - y.ints[i] : y.ints[i] - x.ints[i]);
    return d;
  }
  double unit() const override { return unit_; }
  std::optional<std::string> invalid(const Value& x) const override {
    if (x.kind != Value::Kind::integer) return std::string("a ") + name_of(x.kind) + " value, not an integer vector";
    if (x.ints.size() != l_.size()) return "a vector of " + std::to_string(x.ints.size()) + " coordinates, not " + std::to_string(l_.size());
    for (std::size_t i = 0; i < l_.size(); i++)
      if (x.ints[i] < l_[i] || x.ints[i] > h_[i]) return "coordinate " + std::to_string(i) + " = " + std::to_string(x.ints[i]) + " outside [" + std::to_string(l_[i]) + ", " + std::to_string(h_[i]) + "]";
    return std::nullopt;
  }

  bool can_recombine() const override { return true; }
  Value recombine(const Value& x, const Value& y, calx::random::Stream& s) const override {
    detail::require(*this, x, "recombine");
    detail::require(*this, y, "recombine");
    std::vector<std::int64_t> c(l_.size());
    for (std::size_t i = 0; i < c.size(); i++) {
      const auto a = std::min(x.ints[i], y.ints[i]), b = std::max(x.ints[i], y.ints[i]);
      c[i] = a + static_cast<std::int64_t>(s.below(static_cast<std::uint64_t>(b - a + 1)));
    }
    return integer_value(std::move(c));
  }
  Properties properties() const override { return {true, true, true, true, Unit::mean, true}; }
  bool can_repair() const override { return true; }
  Value repair(const Value& x) const override {
    if (valid(x)) return x;
    if (x.kind != Value::Kind::integer || x.ints.size() != l_.size()) throw Refusal("repair on " + describe() + ": a value of another shape");
    std::vector<std::int64_t> y = x.ints;
    for (std::size_t i = 0; i < y.size(); i++) y[i] = std::clamp(y[i], l_[i], h_[i]);
    return integer_value(std::move(y));
  }
  double change_floor() const override { return movable_ ? 1 : 0; }
  double diameter() const override { return diameter_; }
  std::optional<std::size_t> embedding_dimension() const override { return l_.size(); }
  std::vector<double> embed(const Value& x) const override {
    detail::require(*this, x, "embed");
    std::vector<double> z(l_.size());
    for (std::size_t i = 0; i < z.size(); i++) z[i] = (static_cast<double>(x.ints[i] - l_[i]) + 0.5) / static_cast<double>(h_[i] - l_[i] + 1);
    return z;
  }
  Value unembed(std::span<const double> z) const override {
    if (z.size() != l_.size()) throw Refusal("unembed on " + describe() + ": coordinates of another dimension");
    std::vector<std::int64_t> x(l_.size());
    for (std::size_t i = 0; i < x.size(); i++) {
      const double n = static_cast<double>(h_[i] - l_[i] + 1);
      const double t = std::isnan(z[i]) ? 0 : std::clamp(std::floor(z[i] * n), 0.0, n - 1);
      x[i] = l_[i] + static_cast<std::int64_t>(t);
    }
    return integer_value(std::move(x));
  }
  std::optional<std::size_t> cardinality() const override {
    unsigned __int128 c = 1;
    for (std::size_t i = 0; i < l_.size(); i++) {
      c *= static_cast<unsigned __int128>(h_[i] - l_[i] + 1);
      if (c > (static_cast<unsigned __int128>(1) << 62)) return std::nullopt;
    }
    return static_cast<std::size_t>(c);
  }
  std::vector<Value> elements(std::size_t limit) const override {
    const auto n = cardinality();
    if (!n || *n > limit) throw Refusal(describe() + " has more values than the limit");
    std::vector<Value> out;
    std::vector<std::int64_t> x = l_;
    for (std::size_t k = 0; k < *n; k++) {
      out.push_back(integer_value(x));
      for (std::size_t i = x.size(); i-- > 0;) {
        if (x[i] < h_[i]) { x[i]++; break; }
        x[i] = l_[i];
      }
    }
    return out;
  }

 private:
  void unit_move(std::vector<std::int64_t>& y, calx::random::Stream& s) const {
    std::vector<std::size_t> open;
    for (std::size_t i = 0; i < y.size(); i++)
      if (h_[i] > l_[i]) open.push_back(i);
    const std::size_t i = open[s.below(open.size())];
    if (y[i] == l_[i]) y[i]++;
    else if (y[i] == h_[i]) y[i]--;
    else y[i] += s.uniform() < 0.5 ? -1 : 1;
  }
  std::vector<std::int64_t> l_, h_;
  std::string what_;
  double unit_ = 0, diameter_ = 0;
  bool movable_ = false;
};

// Multisets of n kinds, each kind's multiplicity in [0, cap]: the symmetric difference of multisets is the ℓ¹ distance.
inline std::shared_ptr<IntegerBox> multiset_space(std::vector<std::int64_t> caps) {
  std::vector<std::int64_t> zero(caps.size(), 0);
  const std::string what = "multisets of " + std::to_string(caps.size()) + " kinds";
  return std::make_shared<IntegerBox>(std::move(zero), std::move(caps), what);
}

// ---------------------------------------------------------------------------------------------------- categories ----

class Categorical final : public Space {
 public:
  explicit Categorical(std::size_t k, std::vector<std::string> names = {}) : k_(k), names_(std::move(names)) {
    if (k_ == 0) throw Refusal("a categorical space needs at least one category");
    if (!names_.empty() && names_.size() != k_) throw Refusal("a categorical space's names must be one per category");
  }
  std::size_t size() const { return k_; }
  const std::vector<std::string>& names() const { return names_; }
  std::string describe() const override { return std::to_string(k_) + " categories"; }

  Value sample(calx::random::Stream& s, double = 1) const override { return category_value(static_cast<std::int64_t>(s.below(k_))); }
  Value mutate(const Value& x, double, calx::random::Stream& s) const override {
    detail::require(*this, x, "mutate");
    if (k_ == 1) return x;
    auto j = static_cast<std::int64_t>(s.below(k_ - 1));
    if (j >= x.ints[0]) j++;
    return category_value(j);
  }
  double distance(const Value& x, const Value& y) const override {
    detail::require(*this, x, "distance");
    detail::require(*this, y, "distance");
    return x.ints[0] == y.ints[0] ? 0 : 1;
  }
  double unit() const override { return 1 - 1.0 / static_cast<double>(k_); }
  std::optional<std::string> invalid(const Value& x) const override {
    if (x.kind != Value::Kind::category || x.ints.size() != 1) return std::string("a ") + name_of(x.kind) + " value, not a category";
    if (x.ints[0] < 0 || x.ints[0] >= static_cast<std::int64_t>(k_)) return "category " + std::to_string(x.ints[0]) + " outside 0…" + std::to_string(k_ - 1);
    return std::nullopt;
  }
  bool can_recombine() const override { return true; }
  Value recombine(const Value& x, const Value& y, calx::random::Stream& s) const override {
    detail::require(*this, x, "recombine");
    detail::require(*this, y, "recombine");
    return s.uniform() < 0.5 ? x : y;
  }
  Properties properties() const override { return {true, true, true, true, Unit::mean, true}; }
  bool can_repair() const override { return true; }
  Value repair(const Value& x) const override {
    if (valid(x)) return x;
    if (x.kind != Value::Kind::category || x.ints.size() != 1) throw Refusal("repair on " + describe() + ": a value of another shape");
    return category_value(std::clamp<std::int64_t>(x.ints[0], 0, static_cast<std::int64_t>(k_) - 1));
  }
  std::optional<double> mutation_radius(double) const override { return 1.0; }
  double change_floor() const override { return k_ > 1 ? 1 : 0; }
  double diameter() const override { return k_ > 1 ? 1 : 0; }
  std::optional<std::size_t> embedding_dimension() const override { return k_; }
  std::vector<double> embed(const Value& x) const override {
    detail::require(*this, x, "embed");
    std::vector<double> z(k_, 0);
    z[static_cast<std::size_t>(x.ints[0])] = 1;
    return z;
  }
  Value unembed(std::span<const double> z) const override {
    if (z.size() != k_) throw Refusal("unembed on " + describe() + ": coordinates of another dimension");
    std::size_t best = 0;
    for (std::size_t i = 1; i < k_; i++)
      if (z[i] > z[best]) best = i;
    return category_value(static_cast<std::int64_t>(best));
  }
  std::optional<std::size_t> cardinality() const override { return k_; }
  std::vector<Value> elements(std::size_t limit) const override {
    if (k_ > limit) throw Refusal(describe() + " has more values than the limit");
    std::vector<Value> out;
    for (std::size_t i = 0; i < k_; i++) out.push_back(category_value(static_cast<std::int64_t>(i)));
    return out;
  }

 private:
  std::size_t k_;
  std::vector<std::string> names_;
};

// Ordered categories: k levels under the path metric |i − j|, varied as an integer box of one coordinate.
class Ordinal final : public Space {
 public:
  explicit Ordinal(std::size_t k, std::vector<std::string> names = {})
      : k_(k), names_(std::move(names)), line_({0}, {static_cast<std::int64_t>(k == 0 ? 0 : k - 1)}) {
    if (k_ == 0) throw Refusal("an ordinal space needs at least one level");
    if (!names_.empty() && names_.size() != k_) throw Refusal("an ordinal space's names must be one per level");
  }
  std::size_t size() const { return k_; }
  const std::vector<std::string>& names() const { return names_; }
  std::string describe() const override { return std::to_string(k_) + " ordered levels"; }

  Value sample(calx::random::Stream& s, double = 1) const override { return category_value(static_cast<std::int64_t>(s.below(k_))); }
  Value mutate(const Value& x, double scale, calx::random::Stream& s) const override {
    detail::require(*this, x, "mutate");
    return category_value(line_.mutate(integer_value({x.ints[0]}), scale, s).ints[0]);
  }
  double distance(const Value& x, const Value& y) const override {
    detail::require(*this, x, "distance");
    detail::require(*this, y, "distance");
    return static_cast<double>(x.ints[0] > y.ints[0] ? x.ints[0] - y.ints[0] : y.ints[0] - x.ints[0]);
  }
  double unit() const override { return line_.unit(); }
  std::optional<std::string> invalid(const Value& x) const override {
    if (x.kind != Value::Kind::category || x.ints.size() != 1) return std::string("a ") + name_of(x.kind) + " value, not a level";
    if (x.ints[0] < 0 || x.ints[0] >= static_cast<std::int64_t>(k_)) return "level " + std::to_string(x.ints[0]) + " outside 0…" + std::to_string(k_ - 1);
    return std::nullopt;
  }
  bool can_recombine() const override { return true; }
  Value recombine(const Value& x, const Value& y, calx::random::Stream& s) const override {
    detail::require(*this, x, "recombine");
    detail::require(*this, y, "recombine");
    return category_value(line_.recombine(integer_value({x.ints[0]}), integer_value({y.ints[0]}), s).ints[0]);
  }
  Properties properties() const override { return {true, true, true, true, Unit::mean, true}; }
  bool can_repair() const override { return true; }
  Value repair(const Value& x) const override {
    if (valid(x)) return x;
    if (x.kind != Value::Kind::category || x.ints.size() != 1) throw Refusal("repair on " + describe() + ": a value of another shape");
    return category_value(std::clamp<std::int64_t>(x.ints[0], 0, static_cast<std::int64_t>(k_) - 1));
  }
  double change_floor() const override { return k_ > 1 ? 1 : 0; }
  double diameter() const override { return static_cast<double>(k_ - 1); }
  std::optional<std::size_t> embedding_dimension() const override { return 1; }
  std::vector<double> embed(const Value& x) const override {
    detail::require(*this, x, "embed");
    return line_.embed(integer_value({x.ints[0]}));
  }
  Value unembed(std::span<const double> z) const override { return category_value(line_.unembed(z).ints[0]); }
  std::optional<std::size_t> cardinality() const override { return k_; }
  std::vector<Value> elements(std::size_t limit) const override {
    if (k_ > limit) throw Refusal(describe() + " has more values than the limit");
    std::vector<Value> out;
    for (std::size_t i = 0; i < k_; i++) out.push_back(category_value(static_cast<std::int64_t>(i)));
    return out;
  }

 private:
  std::size_t k_;
  std::vector<std::string> names_;
  IntegerBox line_;
};

// ------------------------------------------------------------------------------------------------------ simplex ----

class Simplex final : public Space {
 public:
  explicit Simplex(std::size_t parts, double total = 1) : D_(parts), T_(total), basis_(tyche::compositions::pivot_basis(parts < 2 ? 2 : parts)) {
    if (D_ < 2) throw Refusal("a simplex needs at least two parts");
    if (!(T_ > 0) || !std::isfinite(T_)) throw Refusal("a simplex's total must be positive and finite");
  }
  std::size_t parts() const { return D_; }
  double total() const { return T_; }
  std::string describe() const override { return "the simplex of " + std::to_string(D_) + " parts"; }

  Value sample(calx::random::Stream& s, double = 1) const override {
    std::vector<double> e(D_);
    double sum = 0;
    for (double& v : e) sum += (v = s.exponential());
    for (double& v : e) v = T_ * v / sum;
    return real_value(close(std::move(e)));
  }
  Value mutate(const Value& x, double scale, calx::random::Stream& s) const override {
    detail::require(*this, x, "mutate");
    std::vector<double> z = tyche::compositions::ilr(x.reals, basis_);
    const double sigma = detail::clamp_scale(scale) * std::numbers::pi / std::sqrt(3.0);
    for (double& v : z) v += sigma * s.normal();
    return from_ilr(z);
  }
  double distance(const Value& x, const Value& y) const override {
    detail::require(*this, x, "distance");
    detail::require(*this, y, "distance");
    return variation::euclidean(tyche::compositions::ilr(x.reals, basis_), tyche::compositions::ilr(y.reals, basis_));
  }
  double unit() const override { return std::numbers::pi * std::sqrt(static_cast<double>(D_ - 1) / 3); }
  std::optional<std::string> invalid(const Value& x) const override {
    if (x.kind != Value::Kind::real) return std::string("a ") + name_of(x.kind) + " value, not a composition";
    if (x.reals.size() != D_) return "a composition of " + std::to_string(x.reals.size()) + " parts, not " + std::to_string(D_);
    double sum = 0;
    for (std::size_t i = 0; i < D_; i++) {
      if (!(x.reals[i] > 0) || !std::isfinite(x.reals[i])) return "part " + std::to_string(i) + " is not positive and finite";
      sum += x.reals[i];
    }
    if (std::abs(sum - T_) > 1e-9 * T_) return "parts summing to " + std::to_string(sum) + ", not " + std::to_string(T_);
    return std::nullopt;
  }
  bool can_recombine() const override { return true; }
  Value recombine(const Value& x, const Value& y, calx::random::Stream& s) const override {
    detail::require(*this, x, "recombine");
    detail::require(*this, y, "recombine");
    if (x == y) return x;
    const auto zx = tyche::compositions::ilr(x.reals, basis_), zy = tyche::compositions::ilr(y.reals, basis_);
    const double t = s.uniform();
    std::vector<double> z(zx.size());
    for (std::size_t i = 0; i < z.size(); i++) z[i] = zx[i] + t * (zy[i] - zx[i]);
    return from_ilr(z);
  }
  // Respectful is not claimed: what two compositions share has no meaning independent of the chosen basis.
  Properties properties() const override { return {true, true, false, true, Unit::rms, true}; }
  bool can_repair() const override { return true; }
  Value repair(const Value& x) const override {
    if (valid(x)) return x;
    if (x.kind != Value::Kind::real || x.reals.size() != D_) throw Refusal("repair on " + describe() + ": a value of another shape");
    std::vector<double> y = x.reals;
    double positive = 0;
    for (double v : y)
      if (v > 0 && std::isfinite(v)) positive += v;
    const double floor = 1e-9 * (positive > 0 ? positive : 1);
    double sum = 0;
    for (double& v : y) sum += (v = (v > 0 && std::isfinite(v)) ? std::max(v, floor) : floor);
    for (double& v : y) v = T_ * v / sum;
    return real_value(close(std::move(y)));
  }
  double change_floor() const override { return 1; }

  bool has_chart() const override { return true; }
  std::size_t chart_dimension() const override { return D_ - 1; }
  std::vector<double> log(const Value& x, const Value& y) const override {
    detail::require(*this, x, "log");
    detail::require(*this, y, "log");
    auto zx = tyche::compositions::ilr(x.reals, basis_);
    const auto zy = tyche::compositions::ilr(y.reals, basis_);
    for (std::size_t i = 0; i < zx.size(); i++) zx[i] = zy[i] - zx[i];
    return zx;
  }
  Value retract(const Value& x, std::span<const double> v) const override {
    detail::require(*this, x, "retract");
    if (v.size() != D_ - 1) throw Refusal("retract on " + describe() + ": a tangent vector of another dimension");
    auto z = tyche::compositions::ilr(x.reals, basis_);
    for (std::size_t i = 0; i < z.size(); i++) z[i] += v[i];
    return from_ilr(z);
  }
  double trust_radius(const Value&) const override { return std::numeric_limits<double>::infinity(); }
  // Standardised ilr coordinates: each has variance one under uniform sampling (π²/6 before scaling).
  std::optional<std::size_t> embedding_dimension() const override { return D_ - 1; }
  bool embedding_bijective() const override { return true; }
  std::vector<double> embed(const Value& x) const override {
    detail::require(*this, x, "embed");
    auto z = tyche::compositions::ilr(x.reals, basis_);
    for (double& v : z) v /= kStandard;
    return z;
  }
  Value unembed(std::span<const double> z) const override {
    if (z.size() != D_ - 1) throw Refusal("unembed on " + describe() + ": coordinates of another dimension");
    std::vector<double> w(z.begin(), z.end());
    for (double& v : w) v *= kStandard;
    return from_ilr(w);
  }

 private:
  static constexpr double kStandard = std::numbers::pi / 2.449489742783178;  // π/√6
  // Positive parts summing to T up to rounding: parts that underflowed are lifted to the least positive double and the
  // whole is scaled back to T, so that what is returned is valid.
  std::vector<double> close(std::vector<double> x) const {
    for (double& v : x)
      if (!(v > 0)) v = std::numeric_limits<double>::min();
    double sum = 0;
    for (double v : x) sum += v;
    for (double& v : x) v = T_ * (v / sum);
    return x;
  }
  Value from_ilr(const std::vector<double>& z) const {
    for (double v : z)
      if (!std::isfinite(v)) throw Refusal("a composition beyond the range of doubles (an ilr coordinate not finite)");
    return real_value(close(tyche::compositions::ilr_inverse(z, basis_, T_)));
  }
  std::size_t D_;
  double T_;
  tyche::compositions::Basis basis_;
};

// -------------------------------------------------------------------------------------------------- permutations ----

class PermutationSpace final : public Space {
 public:
  enum class Metric { cayley, hamming, kendall };
  explicit PermutationSpace(std::size_t n, Metric m = Metric::cayley) : n_(n), m_(m) {
    if (n_ == 0) throw Refusal("a permutation space needs at least one item");
    if (n_ > (std::size_t{1} << 31)) throw Refusal("a permutation space of more than 2³¹ items");
  }
  std::size_t size() const { return n_; }
  Metric metric() const { return m_; }
  std::string describe() const override {
    const char* name = m_ == Metric::cayley ? "Cayley's" : m_ == Metric::hamming ? "Hamming's" : "Kendall's";
    return "permutations of " + std::to_string(n_) + " under " + name + " metric";
  }

  Value sample(calx::random::Stream& s, double = 1) const override {
    const auto p = variation::random_permutation(n_, s);
    return permutation_value(std::vector<std::int64_t>(p.begin(), p.end()));
  }
  Value mutate(const Value& x, double scale, calx::random::Stream& s) const override {
    detail::require(*this, x, "mutate");
    if (n_ < 2) return x;
    for (;;) {
      std::vector<std::int64_t> y = x.ints;
      const std::size_t k = detail::draw_moves(scale, reach(), s);
      for (std::size_t j = 0; j < k; j++) move(y, s);
      if (y != x.ints) return permutation_value(std::move(y));
    }
  }
  double distance(const Value& x, const Value& y) const override {
    detail::require(*this, x, "distance");
    detail::require(*this, y, "distance");
    const auto p = detail::to_permutation(x), q = detail::to_permutation(y);
    switch (m_) {
      case Metric::cayley: return static_cast<double>(variation::cayley_distance(p, q));
      case Metric::hamming: return static_cast<double>(variation::hamming(p, q));
      case Metric::kendall: return static_cast<double>(variation::kendall_distance(p, q));
    }
    return 0;
  }
  double unit() const override {
    const double n = static_cast<double>(n_);
    switch (m_) {
      case Metric::cayley: {
        double h = 0;
        for (std::size_t i = 1; i <= n_; i++) h += 1.0 / static_cast<double>(i);
        return n - h;
      }
      case Metric::hamming: return n - 1;
      case Metric::kendall: return n * (n - 1) / 4;
    }
    return 0;
  }
  std::optional<std::string> invalid(const Value& x) const override {
    if (x.kind != Value::Kind::permutation) return std::string("a ") + name_of(x.kind) + " value, not a permutation";
    if (x.ints.size() != n_) return "a permutation of " + std::to_string(x.ints.size()) + " items, not " + std::to_string(n_);
    std::vector<char> seen(n_, 0);
    for (auto v : x.ints) {
      if (v < 0 || v >= static_cast<std::int64_t>(n_)) return "item " + std::to_string(v) + " outside 0…" + std::to_string(n_ - 1);
      if (seen[static_cast<std::size_t>(v)]++) return "item " + std::to_string(v) + " placed twice";
    }
    return std::nullopt;
  }
  bool can_recombine() const override { return true; }
  Value recombine(const Value& x, const Value& y, calx::random::Stream& s) const override {
    detail::require(*this, x, "recombine");
    detail::require(*this, y, "recombine");
    return permutation_value(m_ == Metric::kendall ? bubble_child(x.ints, y.ints, s) : cycle_child(x.ints, y.ints, s));
  }
  Properties properties() const override { return {true, true, true, true, Unit::mean, true}; }
  bool can_repair() const override { return true; }
  // Keeps the first placement of each item in range, in order, and appends the missing items in increasing order.
  Value repair(const Value& x) const override {
    if (valid(x)) return x;
    if (x.kind != Value::Kind::permutation) throw Refusal("repair on " + describe() + ": a value of another kind");
    std::vector<char> seen(n_, 0);
    std::vector<std::int64_t> y;
    for (auto v : x.ints)
      if (v >= 0 && v < static_cast<std::int64_t>(n_) && !seen[static_cast<std::size_t>(v)]) seen[static_cast<std::size_t>(v)] = 1, y.push_back(v);
    for (std::size_t i = 0; i < n_ && y.size() < n_; i++)
      if (!seen[i]) y.push_back(static_cast<std::int64_t>(i));
    return permutation_value(std::move(y));
  }
  std::optional<double> mutation_radius(double scale) const override {
    return static_cast<double>(detail::move_bound(scale, reach())) * (m_ == Metric::hamming ? 2 : 1);
  }
  double change_floor() const override { return n_ > 1 ? 1 : 0; }
  double diameter() const override {
    const double n = static_cast<double>(n_);
    if (n_ < 2) return 0;
    switch (m_) {
      case Metric::cayley: return n - 1;
      case Metric::hamming: return n;
      case Metric::kendall: return n * (n - 1) / 2;
    }
    return 0;
  }
  // Random keys: each item's key is (its position + ½)/n; the decoder sorts the items by key, ties by item.
  std::optional<std::size_t> embedding_dimension() const override { return n_; }
  std::vector<double> embed(const Value& x) const override {
    detail::require(*this, x, "embed");
    std::vector<double> z(n_);
    for (std::size_t i = 0; i < n_; i++) z[static_cast<std::size_t>(x.ints[i])] = (static_cast<double>(i) + 0.5) / static_cast<double>(n_);
    return z;
  }
  Value unembed(std::span<const double> z) const override {
    if (z.size() != n_) throw Refusal("unembed on " + describe() + ": keys of another dimension");
    std::vector<std::int64_t> p(n_);
    std::iota(p.begin(), p.end(), 0);
    std::stable_sort(p.begin(), p.end(), [&](std::int64_t a, std::int64_t b) {
      const double za = std::isnan(z[static_cast<std::size_t>(a)]) ? std::numeric_limits<double>::infinity() : z[static_cast<std::size_t>(a)];
      const double zb = std::isnan(z[static_cast<std::size_t>(b)]) ? std::numeric_limits<double>::infinity() : z[static_cast<std::size_t>(b)];
      return za < zb;
    });
    return permutation_value(std::move(p));
  }
  std::optional<std::size_t> cardinality() const override {
    if (n_ > 20) return std::nullopt;
    std::size_t f = 1;
    for (std::size_t i = 2; i <= n_; i++) f *= i;
    return f;
  }
  std::vector<Value> elements(std::size_t limit) const override {
    const auto c = cardinality();
    if (!c || *c > limit) throw Refusal(describe() + " has more values than the limit");
    std::vector<std::int64_t> p(n_);
    std::iota(p.begin(), p.end(), 0);
    std::vector<Value> out;
    do out.push_back(permutation_value(p));
    while (std::next_permutation(p.begin(), p.end()));
    return out;
  }

 private:
  double reach() const { return unit() / (m_ == Metric::hamming ? 2 : 1); }
  void move(std::vector<std::int64_t>& y, calx::random::Stream& s) const {
    if (m_ == Metric::kendall) {
      const std::size_t i = static_cast<std::size_t>(s.below(n_ - 1));
      std::swap(y[i], y[i + 1]);
    } else {
      const std::size_t i = static_cast<std::size_t>(s.below(n_));
      std::size_t j = static_cast<std::size_t>(s.below(n_ - 1));
      if (j >= i) j++;
      std::swap(y[i], y[j]);
    }
  }
  // Cycle crossover: the cycles of the positions under i → (where x holds y's item at i), each from x or y.
  std::vector<std::int64_t> cycle_child(const std::vector<std::int64_t>& x, const std::vector<std::int64_t>& y, calx::random::Stream& s) const {
    std::vector<std::size_t> pos_x(n_);
    for (std::size_t i = 0; i < n_; i++) pos_x[static_cast<std::size_t>(x[i])] = i;
    std::vector<char> seen(n_, 0);
    std::vector<std::int64_t> c = x;
    for (std::size_t i = 0; i < n_; i++) {
      if (seen[i]) continue;
      const bool from_y = x[i] != y[i] && s.uniform() < 0.5;
      for (std::size_t j = i; !seen[j]; j = pos_x[static_cast<std::size_t>(y[j])]) {
        seen[j] = 1;
        if (from_y) c[j] = y[j];
      }
    }
    return c;
  }
  // A random point of a bubble-sort path from x to y: t adjacent transpositions, each removing one inversion relative to
  // y, with t uniform in 0…kendall(x, y). The out-of-order adjacent places are kept up to date as the walk goes.
  std::vector<std::int64_t> bubble_child(const std::vector<std::int64_t>& x, const std::vector<std::int64_t>& y, calx::random::Stream& s) const {
    std::vector<std::size_t> rank(n_);
    for (std::size_t i = 0; i < n_; i++) rank[static_cast<std::size_t>(y[i])] = i;
    const auto d = variation::kendall_distance(detail::to_permutation(x), detail::to_permutation(y));
    std::uint64_t t = s.below(d + 1);
    std::vector<std::int64_t> c = x;
    if (n_ < 2) return c;
    std::vector<std::size_t> bad;
    std::vector<std::ptrdiff_t> where(n_ - 1, -1);
    auto out_of_order = [&](std::size_t i) { return rank[static_cast<std::size_t>(c[i])] > rank[static_cast<std::size_t>(c[i + 1])]; };
    auto update = [&](std::size_t i) {
      const bool b = out_of_order(i);
      if (b && where[i] < 0) where[i] = static_cast<std::ptrdiff_t>(bad.size()), bad.push_back(i);
      else if (!b && where[i] >= 0) {
        const std::size_t at = static_cast<std::size_t>(where[i]);
        bad[at] = bad.back();
        where[bad[at]] = static_cast<std::ptrdiff_t>(at);
        bad.pop_back();
        where[i] = -1;
      }
    };
    for (std::size_t i = 0; i + 1 < n_; i++) update(i);
    for (; t > 0 && !bad.empty(); t--) {
      const std::size_t i = bad[s.below(bad.size())];
      std::swap(c[i], c[i + 1]);
      update(i);
      if (i > 0) update(i - 1);
      if (i + 2 < n_) update(i + 1);
    }
    return c;
  }
  std::size_t n_;
  Metric m_;
};

// -------------------------------------------------------------------------------------------------------- subsets ----

class Subset final : public Space {
 public:
  explicit Subset(std::size_t n, std::size_t least = 0, std::size_t most = std::numeric_limits<std::size_t>::max())
      : n_(n), a_(least), b_(std::min(most, n)) {
    if (n_ == 0) throw Refusal("a subset space needs at least one item");
    if (a_ > b_) throw Refusal("a subset space's least size exceeds its greatest");
    sizes_ = detail::binomial_weights(n_, a_, b_);
    double total = 0, mean = 0;
    for (std::size_t k = a_; k <= b_; k++) total += sizes_[k - a_], mean += static_cast<double>(k) * sizes_[k - a_];
    mean /= total;
    unit_ = 2 * mean * (1 - mean / static_cast<double>(n_));
  }
  std::size_t items() const { return n_; }
  std::size_t least() const { return a_; }
  std::size_t most() const { return b_; }
  std::string describe() const override {
    std::string s = "subsets of " + std::to_string(n_) + " items";
    if (a_ > 0 || b_ < n_) s += a_ == b_ ? " of size " + std::to_string(a_) : " of sizes " + std::to_string(a_) + "…" + std::to_string(b_);
    return s;
  }

  Value sample(calx::random::Stream& s, double = 1) const override {
    const std::size_t k = a_ + detail::pick(sizes_, s);
    return set_value(detail::floyd(n_, k, s));
  }
  Value mutate(const Value& x, double scale, calx::random::Stream& s) const override {
    detail::require(*this, x, "mutate");
    if (!movable()) return x;
    for (;;) {
      std::vector<char> in(n_, 0);
      for (auto v : x.ints) in[static_cast<std::size_t>(v)] = 1;
      std::size_t size = x.ints.size();
      const std::size_t k = detail::draw_moves(scale, unit_, s);
      for (std::size_t j = 0; j < k; j++) {
        std::vector<int> kinds;  // 0 add, 1 remove, 2 swap
        if (size < b_) kinds.push_back(0);
        if (size > a_) kinds.push_back(1);
        if (size > 0 && size < n_) kinds.push_back(2);
        const int kind = kinds[s.below(kinds.size())];
        // The items to take out and to put in are both chosen before either is moved, so that a swap always swaps.
        const std::size_t out = kind != 0 ? nth(in, 1, s.below(size)) : n_;
        const std::size_t put = kind != 1 ? nth(in, 0, s.below(n_ - size)) : n_;
        if (out < n_) in[out] = 0;
        if (put < n_) in[put] = 1;
        size += kind == 0 ? 1 : 0;
        size -= kind == 1 ? 1 : 0;
      }
      std::vector<std::int64_t> y;
      for (std::size_t i = 0; i < n_; i++)
        if (in[i]) y.push_back(static_cast<std::int64_t>(i));
      if (y != x.ints) return set_value(std::move(y));
    }
  }
  double distance(const Value& x, const Value& y) const override {
    detail::require(*this, x, "distance");
    detail::require(*this, y, "distance");
    std::size_t common = 0, i = 0, j = 0;
    while (i < x.ints.size() && j < y.ints.size()) {
      if (x.ints[i] == y.ints[j]) common++, i++, j++;
      else if (x.ints[i] < y.ints[j]) i++;
      else j++;
    }
    return static_cast<double>(x.ints.size() + y.ints.size() - 2 * common);
  }
  double unit() const override { return unit_; }
  std::optional<std::string> invalid(const Value& x) const override {
    if (x.kind != Value::Kind::set) return std::string("a ") + name_of(x.kind) + " value, not a set";
    for (std::size_t i = 0; i < x.ints.size(); i++) {
      if (x.ints[i] < 0 || x.ints[i] >= static_cast<std::int64_t>(n_)) return "item " + std::to_string(x.ints[i]) + " outside 0…" + std::to_string(n_ - 1);
      if (i > 0 && x.ints[i] <= x.ints[i - 1]) return "items not listed once each in increasing order";
    }
    if (x.ints.size() < a_ || x.ints.size() > b_) return "a set of " + std::to_string(x.ints.size()) + " items, outside the sizes " + std::to_string(a_) + "…" + std::to_string(b_);
    return std::nullopt;
  }
  bool can_recombine() const override { return true; }
  Value recombine(const Value& x, const Value& y, calx::random::Stream& s) const override {
    detail::require(*this, x, "recombine");
    detail::require(*this, y, "recombine");
    std::vector<std::int64_t> common, differ;
    std::set_intersection(x.ints.begin(), x.ints.end(), y.ints.begin(), y.ints.end(), std::back_inserter(common));
    std::set_symmetric_difference(x.ints.begin(), x.ints.end(), y.ints.begin(), y.ints.end(), std::back_inserter(differ));
    const std::size_t m = differ.size(), base = common.size();
    const std::size_t lo = a_ > base ? a_ - base : 0, hi = std::min(m, b_ - base);
    const std::size_t j = lo + detail::pick(detail::binomial_weights(m, lo, hi), s);
    std::vector<std::int64_t> c = common;
    for (auto r : detail::floyd(m, j, s)) c.push_back(differ[static_cast<std::size_t>(r)]);
    std::sort(c.begin(), c.end());
    return set_value(std::move(c));
  }
  Properties properties() const override { return {true, true, true, true, Unit::mean, true}; }
  bool can_repair() const override { return true; }
  // Drops items out of range and repeats; then adds the least missing items, or drops the greatest, to reach the sizes.
  Value repair(const Value& x) const override {
    if (valid(x)) return x;
    if (x.kind != Value::Kind::set) throw Refusal("repair on " + describe() + ": a value of another kind");
    std::vector<char> in(n_, 0);
    for (auto v : x.ints)
      if (v >= 0 && v < static_cast<std::int64_t>(n_)) in[static_cast<std::size_t>(v)] = 1;
    return set_value(fit(in));
  }
  std::optional<double> mutation_radius(double scale) const override { return 2.0 * static_cast<double>(detail::move_bound(scale, unit_)); }
  double change_floor() const override { return movable() ? 1 : 0; }
  double diameter() const override {
    // max over sizes p, q of p + q − 2·max(0, p + q − n): s = p + q, s ≤ n gives s, s > n gives 2n − s.
    const std::size_t lo = 2 * a_, hi = 2 * b_;
    if (lo <= n_ && n_ <= hi) return static_cast<double>(n_);
    return hi < n_ ? static_cast<double>(hi) : static_cast<double>(2 * n_ - lo);
  }
  // The indicators; the decoder takes the items above ½, then adds the highest absent or drops the lowest present to
  // reach the sizes.
  std::optional<std::size_t> embedding_dimension() const override { return n_; }
  std::vector<double> embed(const Value& x) const override {
    detail::require(*this, x, "embed");
    std::vector<double> z(n_, 0);
    for (auto v : x.ints) z[static_cast<std::size_t>(v)] = 1;
    return z;
  }
  Value unembed(std::span<const double> z) const override {
    if (z.size() != n_) throw Refusal("unembed on " + describe() + ": indicators of another dimension");
    std::vector<std::size_t> order(n_);
    std::iota(order.begin(), order.end(), 0);
    auto val = [&](std::size_t i) { return std::isnan(z[i]) ? -std::numeric_limits<double>::infinity() : z[i]; };
    std::stable_sort(order.begin(), order.end(), [&](std::size_t p, std::size_t q) { return val(p) > val(q); });
    std::size_t k = 0;
    while (k < n_ && val(order[k]) > 0.5) k++;
    k = std::clamp(k, a_, b_);
    std::vector<std::int64_t> c;
    for (std::size_t i = 0; i < k; i++) c.push_back(static_cast<std::int64_t>(order[i]));
    std::sort(c.begin(), c.end());
    return set_value(std::move(c));
  }
  std::optional<std::size_t> cardinality() const override {
    std::uint64_t c = 0;
    for (std::size_t k = a_; k <= b_; k++) {
      const auto ck = detail::binomial_exact(n_, k);
      if (!ck || c + *ck > (std::uint64_t{1} << 62)) return std::nullopt;
      c += *ck;
    }
    return static_cast<std::size_t>(c);
  }
  std::vector<Value> elements(std::size_t limit) const override {
    const auto c = cardinality();
    if (!c || *c > limit || n_ > 62) throw Refusal(describe() + " has more values than the limit");
    std::vector<Value> out;
    for (std::uint64_t mask = 0; mask < (std::uint64_t{1} << n_); mask++) {
      const auto k = static_cast<std::size_t>(std::popcount(mask));
      if (k < a_ || k > b_) continue;
      std::vector<std::int64_t> v;
      for (std::size_t i = 0; i < n_; i++)
        if (mask >> i & 1) v.push_back(static_cast<std::int64_t>(i));
      out.push_back(set_value(std::move(v)));
    }
    return out;
  }

 private:
  bool movable() const { return a_ < b_ || (a_ > 0 && a_ < n_); }
  static std::size_t nth(const std::vector<char>& in, char want, std::uint64_t r) {
    for (std::size_t i = 0; i < in.size(); i++)
      if (in[i] == want && r-- == 0) return i;
    return in.size() - 1;
  }
  std::vector<std::int64_t> fit(std::vector<char> in) const {
    std::size_t k = 0;
    for (char c : in) k += c;
    for (std::size_t i = 0; i < n_ && k < a_; i++)
      if (!in[i]) in[i] = 1, k++;
    for (std::size_t i = n_; i-- > 0 && k > b_;)
      if (in[i]) in[i] = 0, k--;
    std::vector<std::int64_t> c;
    for (std::size_t i = 0; i < n_; i++)
      if (in[i]) c.push_back(static_cast<std::int64_t>(i));
    return c;
  }
  std::size_t n_, a_, b_;
  std::vector<double> sizes_;
  double unit_ = 0;
};

// --------------------------------------------------------------------------------------------------- compositions ----

class Composition final : public Space {
 public:
  // N identical units among D parts; caps empty means none.
  Composition(std::size_t parts, std::int64_t units, std::vector<std::int64_t> caps = {}) : D_(parts), N_(units), caps_(std::move(caps)) {
    if (D_ == 0) throw Refusal("a composition needs at least one part");
    if (N_ < 0) throw Refusal("a composition's units must be at least zero");
    if (N_ > (std::int64_t{1} << 40)) throw Refusal("a composition of more than 2⁴⁰ units");
    capped_ = !caps_.empty();
    if (!capped_) caps_.assign(D_, N_);
    if (caps_.size() != D_) throw Refusal("a composition's caps must be one per part");
    std::int64_t room = 0;
    std::size_t open = 0;
    for (auto& c : caps_) {
      if (c < 0) throw Refusal("a composition's caps must be at least zero");
      c = std::min(c, N_);
      room += c;
      open += c > 0;
    }
    if (room < N_) throw Refusal("a composition whose caps hold fewer than its units");
    movable_ = N_ >= 1 && room > N_ && open >= 2;
    if (capped_) count_table();
    unit_ = capped_ ? estimate_unit() : uncapped_unit();
  }
  std::size_t parts() const { return D_; }
  std::int64_t units() const { return N_; }
  const std::vector<std::int64_t>& caps() const { return caps_; }
  std::string describe() const override {
    return std::to_string(N_) + " units among " + std::to_string(D_) + " parts" + (capped_ ? " with caps" : "");
  }

  Value sample(calx::random::Stream& s, double = 1) const override {
    std::vector<std::int64_t> x(D_, 0);
    if (D_ == 1) {
      x[0] = N_;
      return integer_value(std::move(x));
    }
    if (!capped_) {
      // Stars and bars: D − 1 bars among N + D − 1 places, uniformly.
      const auto bars = detail::floyd(static_cast<std::size_t>(N_) + D_ - 1, D_ - 1, s);
      std::int64_t prev = -1;
      for (std::size_t i = 0; i < D_ - 1; i++) x[i] = bars[i] - prev - 1, prev = bars[i];
      x[D_ - 1] = static_cast<std::int64_t>(static_cast<std::size_t>(N_) + D_ - 1) - prev - 1;
      return integer_value(std::move(x));
    }
    std::int64_t m = N_;
    for (std::size_t d = 0; d < D_; d++) {
      const std::int64_t top = std::min(caps_[d], m);
      x[d] = pick_long(d + 1, m, top, s);
      m -= x[d];
    }
    return integer_value(std::move(x));
  }
  Value mutate(const Value& x, double scale, calx::random::Stream& s) const override {
    detail::require(*this, x, "mutate");
    if (!movable_) return x;
    for (;;) {
      std::vector<std::int64_t> y = x.ints;
      const std::size_t k = detail::draw_moves(scale, unit_, s);
      for (std::size_t j = 0; j < k; j++) move(y, s);
      if (y != x.ints) return integer_value(std::move(y));
    }
  }
  double distance(const Value& x, const Value& y) const override {
    detail::require(*this, x, "distance");
    detail::require(*this, y, "distance");
    std::int64_t d = 0;
    for (std::size_t i = 0; i < D_; i++) d += std::abs(x.ints[i] - y.ints[i]);
    return static_cast<double>(d / 2);
  }
  double unit() const override { return unit_; }
  std::optional<std::string> invalid(const Value& x) const override {
    if (x.kind != Value::Kind::integer) return std::string("a ") + name_of(x.kind) + " value, not a composition";
    if (x.ints.size() != D_) return "a composition of " + std::to_string(x.ints.size()) + " parts, not " + std::to_string(D_);
    std::int64_t sum = 0;
    for (std::size_t i = 0; i < D_; i++) {
      if (x.ints[i] < 0 || x.ints[i] > caps_[i]) return "part " + std::to_string(i) + " = " + std::to_string(x.ints[i]) + " outside 0…" + std::to_string(caps_[i]);
      sum += x.ints[i];
    }
    if (sum != N_) return "parts summing to " + std::to_string(sum) + ", not " + std::to_string(N_);
    return std::nullopt;
  }
  bool can_recombine() const override { return true; }
  // Between the parents' minima and maxima: the remaining units drawn without replacement from the room between them.
  Value recombine(const Value& x, const Value& y, calx::random::Stream& s) const override {
    detail::require(*this, x, "recombine");
    detail::require(*this, y, "recombine");
    std::vector<std::int64_t> c(D_);
    std::int64_t remaining = N_, pool = 0;
    for (std::size_t i = 0; i < D_; i++) {
      c[i] = std::min(x.ints[i], y.ints[i]);
      remaining -= c[i];
      pool += std::abs(x.ints[i] - y.ints[i]);
    }
    for (std::size_t i = 0; i < D_ && remaining > 0; i++) {
      const std::int64_t room = std::abs(x.ints[i] - y.ints[i]);
      if (room == 0) continue;
      const std::int64_t take = hypergeometric(remaining, room, pool - room, s);
      c[i] += take;
      remaining -= take;
      pool -= room;
    }
    return integer_value(std::move(c));
  }
  Properties properties() const override { return {true, true, true, true, Unit::mean, !capped_}; }
  bool can_repair() const override { return true; }
  // Clamps each part to its cap, then adds units to the parts with most room or removes them from the fullest, ties by
  // index.
  Value repair(const Value& x) const override {
    if (valid(x)) return x;
    if (x.kind != Value::Kind::integer || x.ints.size() != D_) throw Refusal("repair on " + describe() + ": a value of another shape");
    std::vector<std::int64_t> y(D_);
    std::int64_t sum = 0;
    for (std::size_t i = 0; i < D_; i++) sum += (y[i] = std::clamp<std::int64_t>(x.ints[i], 0, caps_[i]));
    while (sum < N_) {
      std::size_t best = 0;
      for (std::size_t i = 1; i < D_; i++)
        if (caps_[i] - y[i] > caps_[best] - y[best]) best = i;
      const std::int64_t add = std::min(N_ - sum, caps_[best] - y[best]);
      y[best] += add, sum += add;
    }
    while (sum > N_) {
      std::size_t best = 0;
      for (std::size_t i = 1; i < D_; i++)
        if (y[i] > y[best]) best = i;
      const std::int64_t cut = std::min(sum - N_, y[best]);
      y[best] -= cut, sum -= cut;
    }
    return integer_value(std::move(y));
  }
  std::optional<double> mutation_radius(double scale) const override { return static_cast<double>(detail::move_bound(scale, unit_)); }
  double change_floor() const override { return movable_ ? 1 : 0; }
  double diameter() const override {
    // Without caps exactly N (all units in one part against all in another); with caps N bounds it.
    return movable_ ? static_cast<double>(N_) : 0;
  }
  // The decoder apportions the N units in proportion to the coordinates' positive parts by largest remainders (Hamilton),
  // capped; embed gives the parts themselves, which the decoder returns exactly.
  std::optional<std::size_t> embedding_dimension() const override { return D_; }
  std::vector<double> embed(const Value& x) const override {
    detail::require(*this, x, "embed");
    return std::vector<double>(x.ints.begin(), x.ints.end());
  }
  Value unembed(std::span<const double> z) const override {
    if (z.size() != D_) throw Refusal("unembed on " + describe() + ": coordinates of another dimension");
    std::vector<double> w(D_);
    double total = 0;
    for (std::size_t i = 0; i < D_; i++) total += (w[i] = (std::isfinite(z[i]) && z[i] > 0 && caps_[i] > 0) ? z[i] : 0);
    if (!(total > 0)) {
      for (std::size_t i = 0; i < D_; i++) w[i] = caps_[i] > 0 ? 1 : 0;
      total = 0;
      for (double v : w) total += v;
    }
    std::vector<std::int64_t> y(D_);
    std::vector<std::pair<double, std::size_t>> rest;
    std::int64_t given = 0;
    for (std::size_t i = 0; i < D_; i++) {
      double q = static_cast<double>(N_) * w[i] / total;
      const double r = std::round(q);
      if (std::abs(q - r) < 1e-9 * std::max(1.0, q)) q = r;
      y[i] = std::min(static_cast<std::int64_t>(std::floor(q)), caps_[i]);
      given += y[i];
      rest.push_back({q - std::floor(q), i});
    }
    std::stable_sort(rest.begin(), rest.end(), [](auto a, auto b) { return a.first > b.first; });
    for (std::size_t j = 0; given < N_ && j < rest.size(); j++)
      if (y[rest[j].second] < caps_[rest[j].second]) y[rest[j].second]++, given++;
    return repair(integer_value(std::move(y)));
  }
  std::optional<std::size_t> cardinality() const override {
    // Counted exactly by parts, refusing past 2⁶².
    std::vector<unsigned __int128> f(static_cast<std::size_t>(N_) + 1, 0);
    f[0] = 1;
    const unsigned __int128 top = static_cast<unsigned __int128>(1) << 62;
    std::uint64_t work = 0;
    for (auto c : caps_) work += (static_cast<std::uint64_t>(c) + 1) * (static_cast<std::uint64_t>(N_) + 1);
    if (work > 50'000'000ULL) return std::nullopt;
    for (std::size_t d = 0; d < D_; d++) {
      std::vector<unsigned __int128> g(f.size(), 0);
      for (std::size_t m = 0; m < f.size(); m++)
        for (std::int64_t v = 0; v <= caps_[d] && static_cast<std::int64_t>(m) - v >= 0; v++) {
          g[m] += f[m - static_cast<std::size_t>(v)];
          if (g[m] > top) g[m] = top + 1;
        }
      f = std::move(g);
    }
    if (f[static_cast<std::size_t>(N_)] > top) return std::nullopt;
    return static_cast<std::size_t>(f[static_cast<std::size_t>(N_)]);
  }
  std::vector<Value> elements(std::size_t limit) const override {
    const auto c = cardinality();
    if (!c || *c > limit) throw Refusal(describe() + " has more values than the limit");
    std::vector<Value> out;
    std::vector<std::int64_t> x(D_, 0);
    enumerate(0, N_, x, out);
    return out;
  }

 private:
  void enumerate(std::size_t d, std::int64_t m, std::vector<std::int64_t>& x, std::vector<Value>& out) const {
    if (d + 1 == D_) {
      if (m <= caps_[d]) x[d] = m, out.push_back(integer_value(x));
      return;
    }
    for (std::int64_t v = std::min(caps_[d], m); v >= 0; v--) x[d] = v, enumerate(d + 1, m - v, x, out);
  }
  // Counts of ways to place m units in parts d…D−1, each row scaled by its greatest entry, in long double.
  void count_table() {
    const auto n = static_cast<std::size_t>(N_);
    std::uint64_t work = 0;
    for (auto c : caps_) work += (static_cast<std::uint64_t>(c) + 1) * (n + 1);
    if (work > 400'000'000ULL) throw Refusal("a composition with caps too large to count for uniform sampling");
    table_.assign(D_ + 1, std::vector<long double>(n + 1, 0));
    table_[D_][0] = 1;
    for (std::size_t d = D_; d-- > 0;) {
      long double top = 0;
      for (std::size_t m = 0; m <= n; m++) {
        long double sum = 0;
        for (std::int64_t v = 0; v <= caps_[d] && static_cast<std::int64_t>(m) - v >= 0; v++) sum += table_[d + 1][m - static_cast<std::size_t>(v)];
        table_[d][m] = sum;
        top = std::max(top, sum);
      }
      if (top > 0)
        for (auto& v : table_[d]) v /= top;
    }
  }
  std::int64_t pick_long(std::size_t row, std::int64_t m, std::int64_t top, calx::random::Stream& s) const {
    long double total = 0;
    for (std::int64_t v = 0; v <= top; v++) total += table_[row][static_cast<std::size_t>(m - v)];
    long double u = static_cast<long double>(s.uniform()) * total;
    for (std::int64_t v = 0; v <= top; v++) {
      const long double w = table_[row][static_cast<std::size_t>(m - v)];
      if (u < w) return v;
      u -= w;
    }
    for (std::int64_t v = top; v >= 0; v--)
      if (table_[row][static_cast<std::size_t>(m - v)] > 0) return v;
    return 0;
  }
  void move(std::vector<std::int64_t>& y, calx::random::Stream& s) const {
    std::vector<std::size_t> from, to;
    for (std::size_t i = 0; i < D_; i++) {
      if (y[i] > 0) from.push_back(i);
      if (y[i] < caps_[i]) to.push_back(i);
    }
    for (;;) {
      const std::size_t i = from[s.below(from.size())];
      std::vector<std::size_t> targets;
      for (auto j : to)
        if (j != i) targets.push_back(j);
      if (targets.empty()) continue;
      const std::size_t j = targets[s.below(targets.size())];
      y[i]--, y[j]++;
      return;
    }
  }
  // Successes in `draws` draws without replacement from `good` successes and `bad` failures.
  static std::int64_t hypergeometric(std::int64_t draws, std::int64_t good, std::int64_t bad, calx::random::Stream& s) {
    std::int64_t k = 0;
    for (std::int64_t i = 0; i < draws; i++) {
      if (static_cast<double>(good) > s.uniform() * static_cast<double>(good + bad)) k++, good--;
      else bad--;
    }
    return k;
  }
  // E|X − Y| summed over parts, halved, for X, Y independent with a uniform composition's marginal
  // P(X = j) = C(N − j + D − 2, D − 2)/C(N + D − 1, D − 1).
  double uncapped_unit() const {
    if (D_ < 2 || N_ == 0) return 0;
    const auto n = static_cast<std::size_t>(N_);
    std::vector<double> lp(n + 1);
    for (std::size_t j = 0; j <= n; j++) {
      // log C(N − j + D − 2, D − 2) by its product, relative values only.
      double l = 0;
      for (std::size_t t = 1; t <= D_ - 2; t++) l += std::log(static_cast<double>(n - j) + static_cast<double>(t)) - std::log(static_cast<double>(t));
      lp[j] = l;
    }
    const double top = *std::max_element(lp.begin(), lp.end());
    std::vector<double> p(n + 1);
    double total = 0;
    for (std::size_t j = 0; j <= n; j++) total += (p[j] = std::exp(lp[j] - top));
    for (double& v : p) v /= total;
    double below = 0, below_weighted = 0, e = 0;
    for (std::size_t j = 0; j <= n; j++) {
      e += 2 * p[j] * (static_cast<double>(j) * below - below_weighted);
      below += p[j];
      below_weighted += static_cast<double>(j) * p[j];
    }
    return static_cast<double>(D_) * e / 2;
  }
  // With caps the unit is estimated: the mean distance between 4000 pairs drawn by a fixed stream.
  double estimate_unit() const {
    if (!movable_) return 0;
    calx::random::Stream s(0x5ea5c0de2026ULL, 1, 2, 3);
    double sum = 0;
    const int pairs = 4000;
    for (int i = 0; i < pairs; i++) sum += distance(sample(s), sample(s));
    return sum / pairs;
  }
  std::size_t D_;
  std::int64_t N_;
  std::vector<std::int64_t> caps_;
  bool capped_ = false, movable_ = false;
  double unit_ = 0;
  std::vector<std::vector<long double>> table_;
};

// ------------------------------------------------------------------------------------------------------ sequences ----

class SequenceSpace final : public Space {
 public:
  SequenceSpace(SpacePtr element, std::size_t least, std::size_t most) : e_(std::move(element)), a_(least), b_(most) {
    if (!e_) throw Refusal("a sequence space needs an element space");
    if (a_ > b_) throw Refusal("a sequence space's least length exceeds its greatest");
    if (b_ > 100000) throw Refusal("a sequence space of lengths beyond 100000");
    const double diam = e_->diameter(), u = e_->unit();
    bounded_ = std::isfinite(diam) && diam > 0;
    scale_ = bounded_ ? diam : (u > 0 ? u : 1);
    blend_ = bounded_ && e_->can_recombine() && e_->properties().geometric;
    // The unit is estimated: the mean distance between 300 pairs drawn by a fixed stream.
    calx::random::Stream s(0x5e9e1ce2026ULL, 4, 5, 6);
    double sum = 0;
    const int pairs = 300;
    for (int i = 0; i < pairs; i++) sum += distance(sample(s), sample(s));
    unit_ = sum / pairs;
  }
  const SpacePtr& element() const { return e_; }
  std::size_t least() const { return a_; }
  std::size_t most() const { return b_; }
  std::string describe() const override { return "sequences of " + std::to_string(a_) + "…" + std::to_string(b_) + " of " + e_->describe(); }

  Value sample(calx::random::Stream& s, double size = 1) const override {
    const auto top = a_ + static_cast<std::size_t>(std::llround(detail::clamp_scale(size) * static_cast<double>(b_ - a_)));
    const std::size_t length = a_ + static_cast<std::size_t>(s.below(top - a_ + 1));
    std::vector<Value> xs;
    for (std::size_t i = 0; i < length; i++) xs.push_back(e_->sample(s));
    return sequence_value(std::move(xs));
  }
  Value mutate(const Value& x, double scale, calx::random::Stream& s) const override {
    detail::require(*this, x, "mutate");
    if (a_ == b_ && b_ == 0) return x;
    for (int attempt = 0; attempt < 64; attempt++) {
      std::vector<Value> y = x.parts;
      const std::size_t k = detail::draw_moves(scale, unit_, s);
      for (std::size_t j = 0; j < k; j++) {
        std::vector<int> kinds;  // 0 insert, 1 delete, 2 substitute
        if (y.size() < b_) kinds.push_back(0);
        if (y.size() > a_) kinds.push_back(1);
        if (!y.empty() && e_->change_floor() > 0) kinds.push_back(2);
        if (kinds.empty()) break;
        const int kind = kinds[s.below(kinds.size())];
        if (kind == 0) y.insert(y.begin() + static_cast<std::ptrdiff_t>(s.below(y.size() + 1)), e_->sample(s));
        else if (kind == 1) y.erase(y.begin() + static_cast<std::ptrdiff_t>(s.below(y.size())));
        else {
          const std::size_t i = static_cast<std::size_t>(s.below(y.size()));
          y[i] = e_->mutate(y[i], scale, s);
        }
      }
      if (y != x.parts) return sequence_value(std::move(y));
    }
    return x;
  }
  double distance(const Value& x, const Value& y) const override {
    detail::require(*this, x, "distance");
    detail::require(*this, y, "distance");
    return table(x.parts, y.parts).back().back();
  }
  double unit() const override { return unit_; }
  std::optional<std::string> invalid(const Value& x) const override {
    if (x.kind != Value::Kind::sequence) return std::string("a ") + name_of(x.kind) + " value, not a sequence";
    if (x.parts.size() < a_ || x.parts.size() > b_) return "a sequence of length " + std::to_string(x.parts.size()) + ", outside " + std::to_string(a_) + "…" + std::to_string(b_);
    for (std::size_t i = 0; i < x.parts.size(); i++)
      if (auto why = e_->invalid(x.parts[i])) return "element " + std::to_string(i) + ": " + *why;
    return std::nullopt;
  }
  bool can_recombine() const override { return true; }
  Value recombine(const Value& x, const Value& y, calx::random::Stream& s) const override {
    detail::require(*this, x, "recombine");
    detail::require(*this, y, "recombine");
    // An optimal alignment, read back from the table: columns of (x's element or none, y's element or none).
    const auto t = table(x.parts, y.parts);
    std::vector<std::pair<std::ptrdiff_t, std::ptrdiff_t>> cols;
    std::size_t i = x.parts.size(), j = y.parts.size();
    while (i > 0 || j > 0) {
      if (i > 0 && j > 0 && std::abs(t[i][j] - (t[i - 1][j - 1] + cost(x.parts[i - 1], y.parts[j - 1]))) <= 1e-12 * (1 + t[i][j])) {
        cols.push_back({static_cast<std::ptrdiff_t>(i - 1), static_cast<std::ptrdiff_t>(j - 1)}), i--, j--;
      } else if (i > 0 && std::abs(t[i][j] - (t[i - 1][j] + 1)) <= 1e-12 * (1 + t[i][j])) {
        cols.push_back({static_cast<std::ptrdiff_t>(i - 1), -1}), i--;
      } else {
        cols.push_back({-1, static_cast<std::ptrdiff_t>(j - 1)}), j--;
      }
    }
    std::reverse(cols.begin(), cols.end());
    // Each column from x's side (1) or y's (0); an insertion or deletion column taken from its empty side adds nothing.
    std::vector<char> from_x(cols.size());
    std::size_t length = 0;
    for (std::size_t c = 0; c < cols.size(); c++) {
      from_x[c] = s.uniform() < 0.5;
      const bool present = from_x[c] ? cols[c].first >= 0 : cols[c].second >= 0;
      length += present;
    }
    // Bring the length within bounds by flipping indel columns toward the side that removes or adds an element.
    std::vector<std::size_t> indel;
    for (std::size_t c = 0; c < cols.size(); c++)
      if (cols[c].first < 0 || cols[c].second < 0) indel.push_back(c);
    for (std::size_t r = indel.size(); r > 1; r--) std::swap(indel[r - 1], indel[s.below(r)]);
    for (auto c : indel) {
      const bool present = from_x[c] ? cols[c].first >= 0 : cols[c].second >= 0;
      if (length > b_ && present) from_x[c] = !from_x[c], length--;
      else if (length < a_ && !present) from_x[c] = !from_x[c], length++;
    }
    std::vector<Value> child;
    for (std::size_t c = 0; c < cols.size(); c++) {
      const auto [p, q] = cols[c];
      if (p >= 0 && q >= 0) {
        const Value& u = x.parts[static_cast<std::size_t>(p)];
        const Value& v = y.parts[static_cast<std::size_t>(q)];
        if (u == v) child.push_back(u);
        else if (blend_) child.push_back(e_->recombine(u, v, s));
        else child.push_back(from_x[c] ? u : v);
      } else if (from_x[c] && p >= 0) child.push_back(x.parts[static_cast<std::size_t>(p)]);
      else if (!from_x[c] && q >= 0) child.push_back(y.parts[static_cast<std::size_t>(q)]);
    }
    return sequence_value(std::move(child));
  }
  Properties properties() const override { return {true, true, true, true, Unit::mean, false}; }
  std::optional<double> mutation_radius(double scale) const override { return static_cast<double>(detail::move_bound(scale, unit_)); }
  double change_floor() const override { return (a_ < b_ || (b_ > 0 && e_->change_floor() > 0)) ? 1 : 0; }
  double diameter() const override { return static_cast<double>(b_); }  // a bound: no column costs more than one
  std::optional<std::size_t> cardinality() const override {
    const auto k = e_->cardinality();
    if (!k) return std::nullopt;
    const unsigned __int128 top = static_cast<unsigned __int128>(1) << 62;
    unsigned __int128 total = 0, power = 1;  // power = k^L
    for (std::size_t L = 0; L <= b_; L++) {
      if (L >= a_) total += power;
      if (total > top) return std::nullopt;
      if (L < b_) {
        power *= *k;
        if (power > top) return std::nullopt;
      }
    }
    return static_cast<std::size_t>(total);
  }
  std::vector<Value> elements(std::size_t limit) const override {
    const auto c = cardinality();
    if (!c || *c > limit) throw Refusal(describe() + " has more values than the limit");
    const auto alphabet = e_->elements(limit);
    std::vector<Value> out;
    std::vector<Value> cur;
    grow(alphabet, cur, out);
    return out;
  }

 private:
  void grow(const std::vector<Value>& alphabet, std::vector<Value>& cur, std::vector<Value>& out) const {
    if (cur.size() >= a_) out.push_back(sequence_value(cur));
    if (cur.size() == b_) return;
    for (const auto& v : alphabet) {
      cur.push_back(v);
      grow(alphabet, cur, out);
      cur.pop_back();
    }
  }
  double cost(const Value& u, const Value& v) const {
    if (u == v) return 0;
    const double d = e_->distance(u, v) / scale_;
    return bounded_ ? d : std::min(1.0, d);
  }
  std::vector<std::vector<double>> table(const std::vector<Value>& x, const std::vector<Value>& y) const {
    std::vector<std::vector<double>> t(x.size() + 1, std::vector<double>(y.size() + 1, 0));
    for (std::size_t i = 0; i <= x.size(); i++) t[i][0] = static_cast<double>(i);
    for (std::size_t j = 0; j <= y.size(); j++) t[0][j] = static_cast<double>(j);
    for (std::size_t i = 1; i <= x.size(); i++)
      for (std::size_t j = 1; j <= y.size(); j++)
        t[i][j] = std::min({t[i - 1][j] + 1, t[i][j - 1] + 1, t[i - 1][j - 1] + cost(x[i - 1], y[j - 1])});
    return t;
  }
  SpacePtr e_;
  std::size_t a_, b_;
  bool bounded_ = false, blend_ = false;
  double scale_ = 1, unit_ = 0;
};

}  // namespace heuresis::spaces
