// Mallows's laws on orders (Mallows 1957, "Non-null ranking models", Biometrika 44, 114–130): P(σ) ∝ φ^{d(σ, σ₀)} about a centre σ₀
// with dispersion φ ∈ [0, 1], under Kendall's, Cayley's or Hamming's metric, sampled exactly (decision 0009). Each metric is
// unchanged when the items are renamed, so σ = σ₀ ∘ π with π drawn about the identity has d(σ, σ₀) = d(π, id).
//
// - Kendall's: the repeated insertion model (Doignon, Pekeč and Regenwetter 2004, "The repeated insertion model for rankings",
//   Psychometrika 69, 33–54). Item j is inserted among the j items placed so far at a place that puts i of them after it with
//   probability ∝ φ^i, i = 0, …, j; the insertions are a bijection with orders and their inversions add up, so the law is exact.
//   Z = ∏_{j=0}^{n−1} Σ_{i=0}^{j} φ^i.
// - Cayley's: d = n − c(π) with c the number of cycles, so P ∝ θ^{c(π)} with θ = 1/φ, Ewens's law, sampled by the Chinese
//   restaurant process (Feller's coupling; Arratia, Barbour and Tavaré 2003, Logarithmic Combinatorial Structures, ch. 1): item i
//   opens a new cycle with probability θ/(θ + i − 1), else follows a uniformly chosen earlier item in its cycle.
//   Z = ∏_{i=1}^{n−1} (1 + iφ).
// - Hamming's: d = n − f(π) with f the fixed points. P(f = k) ∝ C(n, k) D(n − k) φ^{n−k}, D the derangement numbers; the fixed
//   points are a uniform k-set and the rest a uniform derangement, drawn by rejection of uniform permutations.
//
// A Mallows law under Kendall's metric is fitted to orders by the Borda centre (the items by mean position, the approximation of
// the Kemeny centre used by Mallows estimation of distribution algorithms; Ceberio, Irurozki, Mendiburu and Lozano 2014, Progress
// in Artificial Intelligence 2, 103–117) and by solving E_φ[d] = the mean distance for φ, which is increasing in φ.

#pragma once

#include <heuresis/sequential/grants.hpp>

#include <heuresis/spaces/base.hpp>
#include <heuresis/variation/distance.hpp>
#include <heuresis/variation/permutation.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

namespace heuresis::sequential {

using Metric = spaces::PermutationSpace::Metric;

class Mallows {
 public:
  Mallows(variation::Permutation centre, double phi, Metric metric = Metric::kendall) : centre_(std::move(centre)), phi_(phi), metric_(metric) {
    if (centre_.empty() || !variation::is_permutation(centre_)) throw Refusal("a Mallows law needs a permutation for its centre");
    if (!(phi_ >= 0 && phi_ <= 1)) throw Refusal("a Mallows law's dispersion must lie in [0, 1]");
    if (metric_ == Metric::hamming) hamming_weights();
  }
  const variation::Permutation& centre() const { return centre_; }
  double phi() const { return phi_; }
  Metric metric() const { return metric_; }
  std::size_t size() const { return centre_.size(); }

  variation::Permutation sample(calx::random::Stream& s) const {
    const variation::Permutation pi = metric_ == Metric::kendall ? kendall(s) : metric_ == Metric::cayley ? cayley(s) : hamming(s);
    variation::Permutation sigma(pi.size());
    for (std::size_t i = 0; i < pi.size(); i++) sigma[i] = centre_[pi[i]];
    return sigma;
  }
  double distance_to_centre(const variation::Permutation& p) const {
    switch (metric_) {
      case Metric::kendall: return static_cast<double>(variation::kendall_distance(p, centre_));
      case Metric::cayley: return static_cast<double>(variation::cayley_distance(p, centre_));
      case Metric::hamming: return static_cast<double>(variation::hamming(p, centre_));
    }
    return 0;
  }
  // The normalising constant Z(φ) and the probability of an order.
  double normaliser() const {
    const std::size_t n = centre_.size();
    double z = 1;
    switch (metric_) {
      case Metric::kendall:
        for (std::size_t j = 0; j < n; j++) {
          double t = 0, pw = 1;
          for (std::size_t i = 0; i <= j; i++) t += pw, pw *= phi_;
          z *= t;
        }
        return z;
      case Metric::cayley:
        for (std::size_t i = 1; i < n; i++) z *= 1 + static_cast<double>(i) * phi_;
        return z;
      case Metric::hamming: return hamming_z_;
    }
    return z;
  }
  double probability(const variation::Permutation& p) const { return std::pow(phi_, distance_to_centre(p)) / normaliser(); }
  // The expected distance to the centre.
  double expected_distance() const {
    const std::size_t n = centre_.size();
    double e = 0;
    switch (metric_) {
      case Metric::kendall:
        for (std::size_t j = 1; j < n; j++) e += expected_insertion(j, phi_);
        return e;
      case Metric::cayley: {
        if (phi_ == 0) return 0;
        const double theta = 1 / phi_;
        double cycles = 0;
        for (std::size_t i = 1; i <= n; i++) cycles += theta / (theta + static_cast<double>(i) - 1);
        return static_cast<double>(n) - cycles;
      }
      case Metric::hamming: {
        double total = 0;
        for (std::size_t k = 0; k <= n; k++) e += static_cast<double>(n - k) * fixed_weights_[k], total += fixed_weights_[k];
        return e / total;
      }
    }
    return e;
  }

  // The expected number of inversions the insertion of item j makes: Σ i φ^i / Σ φ^i over i = 0…j.
  static double expected_insertion(std::size_t j, double phi) {
    double num = 0, den = 0, pw = 1;
    for (std::size_t i = 0; i <= j; i++) num += static_cast<double>(i) * pw, den += pw, pw *= phi;
    return num / den;
  }

 private:
  variation::Permutation kendall(calx::random::Stream& s) const {
    const std::size_t n = centre_.size();
    std::vector<std::uint32_t> order;  // the items placed so far, in order
    for (std::size_t j = 0; j < n; j++) {
      // i items placed after item j, with probability ∝ φ^i
      std::vector<double> w(j + 1);
      double pw = 1;
      for (std::size_t i = 0; i <= j; i++) w[i] = pw, pw *= phi_;
      if (phi_ == 0) std::fill(w.begin() + 1, w.end(), 0.0);
      const std::size_t i = spaces::detail::pick(w, s);
      order.insert(order.end() - static_cast<std::ptrdiff_t>(i), static_cast<std::uint32_t>(j));
    }
    return order;
  }
  variation::Permutation cayley(calx::random::Stream& s) const {
    const std::size_t n = centre_.size();
    variation::Permutation next(n);  // the map i → next(i), built cycle by cycle
    for (std::size_t i = 0; i < n; i++) {
      const bool open = phi_ == 0 || i == 0 || s.uniform() < (1 / phi_) / (1 / phi_ + static_cast<double>(i));
      if (open) next[i] = static_cast<std::uint32_t>(i);
      else {
        const std::size_t j = static_cast<std::size_t>(s.below(i));
        next[i] = next[j];
        next[j] = static_cast<std::uint32_t>(i);
      }
    }
    return next;
  }
  variation::Permutation hamming(calx::random::Stream& s) const {
    const std::size_t n = centre_.size();
    const std::size_t k = spaces::detail::pick(fixed_weights_, s);
    const auto fixed = spaces::detail::floyd(n, k, s);
    std::vector<char> is_fixed(n, 0);
    for (auto f : fixed) is_fixed[static_cast<std::size_t>(f)] = 1;
    std::vector<std::uint32_t> rest;
    for (std::size_t i = 0; i < n; i++)
      if (!is_fixed[i]) rest.push_back(static_cast<std::uint32_t>(i));
    variation::Permutation p(n);
    for (std::size_t i = 0; i < n; i++) p[i] = static_cast<std::uint32_t>(i);
    if (rest.empty()) return p;
    for (;;) {  // a uniform derangement of the rest, by rejection
      std::vector<std::uint32_t> img = rest;
      for (std::size_t i = img.size(); i > 1; i--) std::swap(img[i - 1], img[s.below(i)]);
      bool ok = true;
      for (std::size_t i = 0; i < rest.size() && ok; i++) ok = img[i] != rest[i];
      if (!ok) continue;
      for (std::size_t i = 0; i < rest.size(); i++) p[rest[i]] = img[i];
      return p;
    }
  }
  // Weights of k fixed points, C(n, k) D(n − k) φ^{n − k}, in logarithms scaled to a largest of one, and Z itself.
  void hamming_weights() {
    const std::size_t n = centre_.size();
    std::vector<double> logfact(n + 1, 0), ratio(n + 1, 0);  // ratio[m] = D(m)/m!
    for (std::size_t m = 1; m <= n; m++) logfact[m] = logfact[m - 1] + std::log(static_cast<double>(m));
    double term = 1, sum = 1;
    ratio[0] = 1;
    for (std::size_t m = 1; m <= n; m++) term *= -1.0 / static_cast<double>(m), sum += term, ratio[m] = sum;
    std::vector<double> lw(n + 1, -std::numeric_limits<double>::infinity());
    for (std::size_t k = 0; k <= n; k++) {
      const std::size_t m = n - k;
      if (ratio[m] <= 1e-300 || (m > 0 && phi_ == 0)) continue;  // D(1) = 0; and no moved item when φ = 0
      // C(n, k) D(m) = n!/(k! m!) · m! ratio[m] = n!/k! · ratio[m]
      lw[k] = logfact[n] - logfact[k] + std::log(ratio[m]) + (m > 0 ? static_cast<double>(m) * std::log(phi_) : 0);
    }
    const double top = *std::max_element(lw.begin(), lw.end());
    fixed_weights_.assign(n + 1, 0);
    hamming_z_ = 0;
    for (std::size_t k = 0; k <= n; k++) {
      fixed_weights_[k] = std::isfinite(lw[k]) ? std::exp(lw[k] - top) : 0;
      hamming_z_ += std::isfinite(lw[k]) ? std::exp(lw[k]) : 0;
    }
  }
  variation::Permutation centre_;
  double phi_;
  Metric metric_;
  std::vector<double> fixed_weights_;
  double hamming_z_ = 1;
};

// A Mallows law under Kendall's metric fitted to orders: the Borda centre, and φ solving E_φ[d] = the mean distance.
inline Mallows fit_mallows(const std::vector<variation::Permutation>& orders) {
  if (orders.empty()) throw Refusal("a Mallows law is fitted to at least one order");
  const std::size_t n = orders[0].size();
  std::vector<double> mean(n, 0);
  for (const auto& p : orders) {
    if (p.size() != n || !variation::is_permutation(p)) throw Refusal("a Mallows law is fitted to permutations of one size");
    for (std::size_t i = 0; i < n; i++) mean[p[i]] += static_cast<double>(i);
  }
  variation::Permutation centre(n);
  std::iota(centre.begin(), centre.end(), 0u);
  std::stable_sort(centre.begin(), centre.end(), [&](std::uint32_t a, std::uint32_t b) { return mean[a] < mean[b]; });
  double d = 0;
  for (const auto& p : orders) d += static_cast<double>(variation::kendall_distance(p, centre));
  d /= static_cast<double>(orders.size());
  auto expected = [n](double phi) {
    double e = 0;
    for (std::size_t j = 1; j < n; j++) e += Mallows::expected_insertion(j, phi);
    return e;
  };
  double lo = 0, hi = 1;
  if (d >= expected(1)) return Mallows(centre, 1);
  for (int it = 0; it < 200; it++) {
    const double mid = (lo + hi) / 2;
    (expected(mid) < d ? lo : hi) = mid;
  }
  return Mallows(centre, (lo + hi) / 2);
}

}  // namespace heuresis::sequential
