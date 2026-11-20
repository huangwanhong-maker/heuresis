// Pairwise judgements, from people or from language models, read as Bradley and Terry's model with an effect of the order of
// presentation (decision 0010; the survey, Part V, design §5–6).
//
// P(i preferred to j | i shown first) = θπᵢ/(θπᵢ + πⱼ), strengths πᵢ > 0 and θ > 0 the advantage of being shown first (θ = 1 when
// order does not matter; Bradley and Terry 1952, Biometrika 39, 324–345; the order effect as the home-field advantage of Agresti
// 1990, Categorical Data Analysis). The maximum-likelihood strengths are found by Hunter's minorise–maximise iteration, each step
// increasing the likelihood (Hunter 2004, "MM algorithms for generalized Bradley–Terry models", Ann. Statist. 32, 384–406):
//   πᵢ ← Wᵢ / [ Σ_{i shown first} θ/(θπᵢ + πⱼ) + Σ_{i shown second} 1/(θπⱼ + πᵢ) ],
//   θ ← H / Σ_{all} πᶠ/(θπᶠ + πˢ),
// Wᵢ the judgements won by i, H those won by the item shown first. The maximum exists exactly when every partition of the items
// into two groups has a judgement won in each direction across it (Ford 1957, Amer. Math. Monthly 64, 28–33): the graph of wins is
// strongly connected. Otherwise it is refused unless a prior is declared: a number of pseudo-judgements of each pair, half won each
// way and half shown each way, which shrinks the strengths toward one another.
//
// Strengths are normalised to a geometric mean of one; log-strengths are their centred logarithms. Judgements enter selection only
// as descriptors (tiers.hpp): what a person or a model prefers is recorded as such, never summed into a fitness.

#pragma once

#include <heuresis/assessment/record.hpp>

#include <cmath>
#include <vector>

namespace heuresis::assessment {

struct Judgement {
  std::size_t first = 0, second = 0;   // the items, in the order shown
  bool first_preferred = true;
};

struct BradleyTerry {
  std::vector<double> strength;        // geometric mean one
  std::vector<double> log_strength;    // centred
  double order_effect = 1;             // θ
  double log_likelihood = 0;
  std::size_t iterations = 0;
  bool converged = false;
  bool likelihood_rose = true;         // every step increased the likelihood (to rounding)
};

namespace detail {
inline bool strongly_connected(std::size_t n, const std::vector<std::vector<char>>& beat) {
  for (int pass = 0; pass < 2; pass++) {
    std::vector<char> seen(n, 0);
    std::vector<std::size_t> stack{0};
    seen[0] = 1;
    while (!stack.empty()) {
      const std::size_t i = stack.back();
      stack.pop_back();
      for (std::size_t j = 0; j < n; j++)
        if (!seen[j] && (pass == 0 ? beat[i][j] : beat[j][i])) seen[j] = 1, stack.push_back(j);
    }
    for (char c : seen)
      if (!c) return false;
  }
  return true;
}
}  // namespace detail

inline BradleyTerry bradley_terry(std::size_t items, const std::vector<Judgement>& judgements, bool order_effect = true, double prior = 0,
                                  std::size_t max_iterations = 20000, double tolerance = 1e-12) {
  if (items < 2) throw Refusal("Bradley–Terry needs at least two items");
  if (!(prior >= 0)) throw Refusal("a prior of pseudo-judgements must be at least zero");
  struct Weighted {
    std::size_t f, s;
    double first_wins, second_wins;
  };
  std::vector<Weighted> games;
  std::vector<std::vector<char>> beat(items, std::vector<char>(items, 0));
  for (const auto& j : judgements) {
    if (j.first >= items || j.second >= items || j.first == j.second) throw Refusal("a judgement between an item and itself or outside the items");
    games.push_back({j.first, j.second, j.first_preferred ? 1.0 : 0.0, j.first_preferred ? 0.0 : 1.0});
    if (j.first_preferred) beat[j.first][j.second] = 1;
    else beat[j.second][j.first] = 1;
  }
  if (prior > 0) {
    for (std::size_t a = 0; a < items; a++)
      for (std::size_t b = a + 1; b < items; b++) {
        games.push_back({a, b, prior / 4, prior / 4});
        games.push_back({b, a, prior / 4, prior / 4});
      }
  } else if (!detail::strongly_connected(items, beat)) {
    throw Refusal("the judgements leave a group of items never preferred to the rest (Ford's condition fails): declare a prior");
  }
  std::vector<double> wins(items, 0);
  double first_total = 0, total = 0;
  for (const auto& g : games) wins[g.f] += g.first_wins, wins[g.s] += g.second_wins, first_total += g.first_wins, total += g.first_wins + g.second_wins;
  BradleyTerry r;
  std::vector<double> pi(items, 1.0);
  double theta = 1;
  auto loglik = [&]() {
    double l = 0;
    for (const auto& g : games) {
      const double a = theta * pi[g.f], b = pi[g.s];
      l += g.first_wins * std::log(a / (a + b)) + g.second_wins * std::log(b / (a + b));
    }
    return l;
  };
  double last = loglik();
  for (r.iterations = 0; r.iterations < max_iterations; r.iterations++) {
    std::vector<double> denom(items, 0);
    for (const auto& g : games) {
      const double w = g.first_wins + g.second_wins;
      const double d = theta * pi[g.f] + pi[g.s];
      denom[g.f] += w * theta / d;
      denom[g.s] += w / d;
    }
    for (std::size_t i = 0; i < items; i++) pi[i] = wins[i] / denom[i];
    double logmean = 0;
    for (double p : pi) logmean += std::log(p);
    logmean /= static_cast<double>(items);
    for (double& p : pi) p /= std::exp(logmean);
    if (order_effect) {
      double d = 0;
      for (const auto& g : games) d += (g.first_wins + g.second_wins) * pi[g.f] / (theta * pi[g.f] + pi[g.s]);
      theta = first_total / d;
    }
    const double now = loglik();
    if (now < last - 1e-9 * (1 + std::abs(last))) r.likelihood_rose = false;
    if (std::abs(now - last) <= tolerance * (1 + std::abs(now))) {
      last = now;
      r.converged = true;
      break;
    }
    last = now;
  }
  (void)total;
  r.strength = pi;
  for (double p : pi) r.log_strength.push_back(std::log(p));
  r.order_effect = theta;
  r.log_likelihood = last;
  return r;
}

}  // namespace heuresis::assessment
