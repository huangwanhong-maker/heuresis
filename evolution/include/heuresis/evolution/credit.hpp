// Adaptive operator selection: which operator makes the next child, learned from what the operators' children did (decision 0006).
//
// Each operator k keeps a quality q_k, an estimate of its reward, and a count n_k. A schedule turns them into a choice:
// - fixed: the operators' declared weights, never changed;
// - probability matching (Goldberg 1990; Thierens 2005, GECCO, 1539–1546): p_k = p_min + (1 − K p_min) q_k / Σ q;
// - adaptive pursuit (Thierens 2005): the best operator's probability pursues p_max = 1 − (K − 1) p_min, the others p_min, at rate β;
// - UCB1 (Auer, Cesa-Bianchi and Fischer 2002, Mach. Learn. 47, 235–256): the operator maximising q_k + c √(2 ln N / n_k), each tried
//   once first (Fialho, Da Costa, Schoenauer and Sebag 2010 for its use in operator selection);
// - Thompson sampling (Thompson 1933; Chapelle and Li 2011): rewards in [0, 1] read as Bernoulli successes, a Beta(1 + s_k, 1 + f_k)
//   drawn for each and the largest taken.
// Qualities are updated as exponential averages, q_k ← q_k + α (r − q_k), so that an operator good early and poor later is noticed.
#pragma once

#include <calx/random/stream.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <heuresis/evolution/strategies.hpp>

namespace heuresis::evolution {

struct Credits {
  std::vector<double> weight;       // declared, for fixed
  std::vector<double> quality;      // q_k
  std::vector<double> probability;  // p_k (probability matching, adaptive pursuit; fixed: the normalised weights)
  std::vector<double> successes, failures;  // for Thompson: accumulated reward and its complement
  std::vector<std::size_t> count;   // children made by each
  std::size_t total = 0;

  explicit Credits(std::vector<double> w = {}) { reset(std::move(w)); }

  void reset(std::vector<double> w) {
    weight = std::move(w);
    const std::size_t K = weight.size();
    quality.assign(K, 1.0);  // optimistic start
    successes.assign(K, 0.0), failures.assign(K, 0.0), count.assign(K, 0);
    total = 0;
    probability = normalised(weight);
  }

  static std::vector<double> normalised(const std::vector<double>& w) {
    double s = 0;
    for (double x : w) s += std::max(0.0, x);
    std::vector<double> p(w.size(), w.empty() ? 0.0 : 1.0 / static_cast<double>(w.size()));
    if (s > 0)
      for (std::size_t k = 0; k < w.size(); k++) p[k] = std::max(0.0, w[k]) / s;
    return p;
  }

  std::size_t size() const { return weight.size(); }

  std::size_t choose(const Strategy& st, calx::random::Stream& s) const {
    const std::size_t K = size();
    if (K == 0) throw std::invalid_argument("there is no operator to choose");
    if (st.schedule == Schedule::ucb) {
      for (std::size_t k = 0; k < K; k++)
        if (count[k] == 0 && weight[k] > 0) return k;
      std::size_t best = 0;
      double bv = -std::numeric_limits<double>::infinity();
      for (std::size_t k = 0; k < K; k++) {
        if (!(weight[k] > 0)) continue;
        const double v = quality[k] + st.exploration * std::sqrt(2 * std::log(static_cast<double>(std::max<std::size_t>(total, 1))) / static_cast<double>(count[k]));
        if (v > bv) bv = v, best = k;
      }
      return best;
    }
    if (st.schedule == Schedule::thompson) {
      std::size_t best = 0;
      double bv = -1;
      for (std::size_t k = 0; k < K; k++) {
        if (!(weight[k] > 0)) continue;
        const double v = beta(1 + successes[k], 1 + failures[k], s);
        if (v > bv) bv = v, best = k;
      }
      return best;
    }
    // Draw by the probabilities.
    const double u = s.uniform();
    double c = 0;
    for (std::size_t k = 0; k < K; k++) {
      c += probability[k];
      if (u < c) return k;
    }
    for (std::size_t k = K; k-- > 0;)
      if (probability[k] > 0) return k;
    return K - 1;
  }

  // The reward r ∈ [0, 1] of one child of operator k.
  void reward(std::size_t k, double r, const Strategy& st) {
    r = std::clamp(r, 0.0, 1.0);
    count[k]++, total++;
    successes[k] += r, failures[k] += 1 - r;
    quality[k] += st.adaptation * (r - quality[k]);
    if (st.schedule == Schedule::ucb && count[k] == 1) quality[k] = r;  // UCB's estimate is the mean, from its first pull
    update(st);
  }

  void update(const Strategy& st) {
    const std::size_t K = size();
    if (K == 0) return;
    if (st.schedule == Schedule::fixed || st.schedule == Schedule::ucb || st.schedule == Schedule::thompson) {
      probability = normalised(weight);
      return;
    }
    std::size_t live = 0;
    for (double w : weight) live += w > 0;
    const double pmin = std::min(st.p_min, live ? 1.0 / static_cast<double>(live) : 0.0);
    if (st.schedule == Schedule::probability_matching) {
      double s = 0;
      for (std::size_t k = 0; k < K; k++)
        if (weight[k] > 0) s += quality[k];
      for (std::size_t k = 0; k < K; k++)
        probability[k] = weight[k] > 0 ? pmin + (1 - static_cast<double>(live) * pmin) * (s > 0 ? quality[k] / s : 1.0 / static_cast<double>(live)) : 0.0;
      return;
    }
    // Adaptive pursuit.
    std::size_t best = 0;
    for (std::size_t k = 0; k < K; k++)
      if (weight[k] > 0 && (weight[best] <= 0 || quality[k] > quality[best])) best = k;
    const double pmax = 1 - static_cast<double>(live - 1) * pmin;
    for (std::size_t k = 0; k < K; k++) {
      if (!(weight[k] > 0)) { probability[k] = 0; continue; }
      const double target = k == best ? pmax : pmin;
      probability[k] += st.adaptation * (target - probability[k]);
    }
    double s = std::accumulate(probability.begin(), probability.end(), 0.0);
    if (s > 0)
      for (double& p : probability) p /= s;
  }

  // A Beta(a, b) variate from two gamma variates (Marsaglia and Tsang 2000, ACM TOMS 26, 363–372, for shape ≥ 1).
  static double beta(double a, double b, calx::random::Stream& s) {
    const double x = gamma(a, s), y = gamma(b, s);
    return x / (x + y);
  }
  static double gamma(double a, calx::random::Stream& s) {
    if (a < 1) return gamma(a + 1, s) * std::pow(s.uniform(), 1 / a);
    const double d = a - 1.0 / 3, c = 1 / std::sqrt(9 * d);
    for (;;) {
      double x, v;
      do { x = s.normal(); v = 1 + c * x; } while (v <= 0);
      v = v * v * v;
      const double u = s.uniform();
      if (u < 1 - 0.0331 * x * x * x * x || std::log(u) < 0.5 * x * x + d * (1 - v + std::log(v))) return d * v;
    }
  }
};

}  // namespace heuresis::evolution
