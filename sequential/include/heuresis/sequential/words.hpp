// Words with fixed letter counts: orders in which a letter may recur, such as a picking sequence (each agent picks a stated number
// of times), a serial order of categories with quotas, or grants repeated in batches (decision 0009). A mechanism whose outcome
// depends on an order (picking sequences, serial dictatorships, the precedence of reserve categories) is searched over such words,
// its evaluation running the mechanism wherever it is written.
//
// The metric is the least number of exchanges of adjacent letters that turns one word into another. Labelling the occurrences of
// each letter in order (the k-th a of one word matched to the k-th a of the other) makes the two words permutations of the same
// labels, and the least number of exchanges is then Kendall's distance between them: an exchange of two equal letters is never
// needed, so matching occurrences in order is optimal (Diaconis 1988, Group Representations in Probability and Statistics, ch. 6,
// for metrics on such words). The unit is the mean distance between two uniform words, Σ over letters a < b and occurrences k, l
// of 2p(1 − p), with p the chance that the k-th a precedes the l-th b, which is hypergeometric:
//   p = Σ_{j ≥ k} C(k + l − 1, j) C(m + n − k − l + 1, m − j) / C(m + n, m)
// for m occurrences of a and n of b (at least k of the first k + l − 1 letters of the {a, b} subword are a). Mutation exchanges
// adjacent different letters; recombination is a point of a bubble-sort path between the labelled words, which never exchanges two
// occurrences of one letter (both words have them in order), so its children are words with canonical labels on the segment.

#pragma once

#include <heuresis/sequential/grants.hpp>

#include <heuresis/spaces/base.hpp>
#include <heuresis/variation/distance.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace heuresis::sequential {

class WordSpace final : public spaces::Space {
 public:
  explicit WordSpace(std::vector<std::size_t> counts, std::vector<std::string> names = {}) : counts_(std::move(counts)), names_(std::move(names)) {
    if (counts_.empty()) throw Refusal("words need at least one letter");
    if (!names_.empty() && names_.size() != counts_.size()) throw Refusal("a word space's names must be one per letter");
    std::size_t kinds = 0;
    for (auto c : counts_) n_ += c, kinds += c > 0;
    if (n_ == 0) throw Refusal("words of no letters");
    movable_ = kinds >= 2;
    offset_.assign(counts_.size(), 0);
    for (std::size_t a = 1; a < counts_.size(); a++) offset_[a] = offset_[a - 1] + counts_[a - 1];
    for (std::size_t a = 0; a < counts_.size(); a++)
      for (std::size_t i = 0; i < counts_[a]; i++) letter_of_label_.push_back(a);
    unit_ = exact_unit();
    for (std::size_t a = 0; a < counts_.size(); a++)
      for (std::size_t b = a + 1; b < counts_.size(); b++) diameter_ += static_cast<double>(counts_[a] * counts_[b]);
  }
  const std::vector<std::size_t>& counts() const { return counts_; }
  std::size_t length() const { return n_; }
  std::string describe() const override {
    std::string s = "words with letter counts (";
    for (std::size_t a = 0; a < counts_.size(); a++) s += (a ? ", " : "") + std::to_string(counts_[a]);
    return s + ")";
  }
  // The occurrences labelled in order: the k-th occurrence of letter a is label offset(a) + k.
  variation::Permutation labelled(const spaces::Value& x) const {
    std::vector<std::size_t> seen(counts_.size(), 0);
    variation::Permutation p(n_);
    for (std::size_t i = 0; i < n_; i++) {
      const auto a = static_cast<std::size_t>(x.ints[i]);
      p[i] = static_cast<std::uint32_t>(offset_[a] + seen[a]++);
    }
    return p;
  }
  spaces::Value word_of(const variation::Permutation& p) const {
    std::vector<std::int64_t> w(n_);
    for (std::size_t i = 0; i < n_; i++) w[i] = static_cast<std::int64_t>(letter_of_label_[p[i]]);
    return spaces::integer_value(std::move(w));
  }

  spaces::Value sample(calx::random::Stream& s, double = 1) const override {
    std::vector<std::int64_t> w;
    for (std::size_t a = 0; a < counts_.size(); a++)
      for (std::size_t i = 0; i < counts_[a]; i++) w.push_back(static_cast<std::int64_t>(a));
    for (std::size_t i = w.size(); i > 1; i--) std::swap(w[i - 1], w[s.below(i)]);
    return spaces::integer_value(std::move(w));
  }
  spaces::Value mutate(const spaces::Value& x, double scale, calx::random::Stream& s) const override {
    spaces::detail::require(*this, x, "mutate");
    if (!movable_) return x;
    for (;;) {
      std::vector<std::int64_t> y = x.ints;
      const std::size_t k = spaces::detail::draw_moves(scale, unit_, s);
      for (std::size_t j = 0; j < k; j++) {
        std::vector<std::size_t> places;
        for (std::size_t i = 0; i + 1 < n_; i++)
          if (y[i] != y[i + 1]) places.push_back(i);
        const std::size_t i = places[s.below(places.size())];
        std::swap(y[i], y[i + 1]);
      }
      if (y != x.ints) return spaces::integer_value(std::move(y));
    }
  }
  double distance(const spaces::Value& x, const spaces::Value& y) const override {
    spaces::detail::require(*this, x, "distance");
    spaces::detail::require(*this, y, "distance");
    return static_cast<double>(variation::kendall_distance(labelled(x), labelled(y)));
  }
  double unit() const override { return unit_; }
  std::optional<std::string> invalid(const spaces::Value& x) const override {
    if (x.kind != spaces::Value::Kind::integer) return std::string("a ") + spaces::name_of(x.kind) + " value, not a word";
    if (x.ints.size() != n_) return "a word of length " + std::to_string(x.ints.size()) + ", not " + std::to_string(n_);
    std::vector<std::size_t> seen(counts_.size(), 0);
    for (auto v : x.ints) {
      if (v < 0 || v >= static_cast<std::int64_t>(counts_.size())) return "letter " + std::to_string(v) + " outside 0…" + std::to_string(counts_.size() - 1);
      seen[static_cast<std::size_t>(v)]++;
    }
    for (std::size_t a = 0; a < counts_.size(); a++)
      if (seen[a] != counts_[a]) return "letter " + std::to_string(a) + " " + std::to_string(seen[a]) + " times, not " + std::to_string(counts_[a]);
    return std::nullopt;
  }
  bool can_recombine() const override { return true; }
  spaces::Value recombine(const spaces::Value& x, const spaces::Value& y, calx::random::Stream& s) const override {
    spaces::detail::require(*this, x, "recombine");
    spaces::detail::require(*this, y, "recombine");
    // A random point of a bubble-sort path between the labelled words, as for permutations under Kendall's metric.
    spaces::PermutationSpace orders(n_, spaces::PermutationSpace::Metric::kendall);
    const auto px = labelled(x), py = labelled(y);
    const auto c = orders.recombine(spaces::permutation_value(std::vector<std::int64_t>(px.begin(), px.end())), spaces::permutation_value(std::vector<std::int64_t>(py.begin(), py.end())), s);
    variation::Permutation pc(n_);
    for (std::size_t i = 0; i < n_; i++) pc[i] = static_cast<std::uint32_t>(c.ints[i]);
    return word_of(pc);
  }
  spaces::Properties properties() const override { return {true, true, true, true, spaces::Unit::mean, true}; }
  std::optional<double> mutation_radius(double scale) const override { return static_cast<double>(spaces::detail::move_bound(scale, unit_)); }
  double change_floor() const override { return movable_ ? 1 : 0; }
  double diameter() const override { return diameter_; }
  // Random keys on the labelled occurrences.
  std::optional<std::size_t> embedding_dimension() const override { return n_; }
  std::vector<double> embed(const spaces::Value& x) const override {
    spaces::detail::require(*this, x, "embed");
    const auto p = labelled(x);
    std::vector<double> z(n_);
    for (std::size_t i = 0; i < n_; i++) z[p[i]] = (static_cast<double>(i) + 0.5) / static_cast<double>(n_);
    return z;
  }
  spaces::Value unembed(std::span<const double> z) const override {
    if (z.size() != n_) throw Refusal("unembed on " + describe() + ": keys of another dimension");
    variation::Permutation p(n_);
    for (std::size_t i = 0; i < n_; i++) p[i] = static_cast<std::uint32_t>(i);
    std::stable_sort(p.begin(), p.end(), [&](std::uint32_t a, std::uint32_t b) {
      const double inf = std::numeric_limits<double>::infinity();
      const double za = std::isnan(z[a]) ? inf : z[a], zb = std::isnan(z[b]) ? inf : z[b];
      return za < zb;
    });
    return word_of(p);
  }
  std::optional<std::size_t> cardinality() const override {
    // The multinomial coefficient, refused past 2⁶².
    unsigned __int128 c = 1;
    std::size_t placed = 0;
    for (auto m : counts_)
      for (std::size_t i = 1; i <= m; i++) {
        placed++;
        c = c * placed / i;
        if (c > (static_cast<unsigned __int128>(1) << 62)) return std::nullopt;
      }
    return static_cast<std::size_t>(c);
  }
  std::vector<spaces::Value> elements(std::size_t limit) const override {
    const auto c = cardinality();
    if (!c || *c > limit) throw Refusal(describe() + " has more values than the limit");
    std::vector<std::int64_t> w;
    for (std::size_t a = 0; a < counts_.size(); a++)
      for (std::size_t i = 0; i < counts_[a]; i++) w.push_back(static_cast<std::int64_t>(a));
    std::vector<spaces::Value> out;
    do out.push_back(spaces::integer_value(w));
    while (std::next_permutation(w.begin(), w.end()));
    return out;
  }

 private:
  double exact_unit() const {
    // log C(n, k) by sums of logarithms, exactly enough for weights summing to one.
    auto logc = [](std::size_t n, std::size_t k) -> double {
      if (k > n) return -std::numeric_limits<double>::infinity();
      double l = 0;
      for (std::size_t i = 1; i <= k; i++) l += std::log(static_cast<double>(n - k + i)) - std::log(static_cast<double>(i));
      return l;
    };
    double u = 0;
    for (std::size_t a = 0; a < counts_.size(); a++)
      for (std::size_t b = a + 1; b < counts_.size(); b++) {
        const std::size_t m = counts_[a], n = counts_[b];
        if (m == 0 || n == 0) continue;
        const double total = logc(m + n, m);
        for (std::size_t k = 1; k <= m; k++)
          for (std::size_t l = 1; l <= n; l++) {
            const std::size_t first = k + l - 1;
            double p = 0;
            for (std::size_t j = k; j <= std::min(m, first); j++) {
              if (m - j > m + n - first) continue;
              p += std::exp(logc(first, j) + logc(m + n - first, m - j) - total);
            }
            p = std::clamp(p, 0.0, 1.0);
            u += 2 * p * (1 - p);
          }
      }
    return u;
  }
  std::vector<std::size_t> counts_;
  std::vector<std::string> names_;
  std::size_t n_ = 0;
  std::vector<std::size_t> offset_, letter_of_label_;
  bool movable_ = false;
  double unit_ = 0, diameter_ = 0;
};

}  // namespace heuresis::sequential
