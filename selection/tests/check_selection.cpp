// Checks of heuresis selection: tournament frequencies against Goldberg and Deb's closed form; ranking
// probabilities; Baker's universal sampling within one of every expected count, exactly; survivors;
// lexicase's specialists; invariance of order-based selection under monotone transformations, exactly.

#include <heuresis/search/draws.hpp>
#include <heuresis/selection/selection.hpp>

#include <calx/verification/check.hpp>

#include <cmath>

using namespace heuresis::selection;
using heuresis::search::Draws;
using heuresis::search::scalar;
using heuresis::search::tag;

int main() {
  calx::verification::Suite s;
  const Draws draws(11);

  std::vector<Evaluation> pop;
  for (double v : {5.0, 1.0, 9.0, 3.0, 7.0, 2.0, 8.0, 4.0, 6.0, 0.0}) pop.push_back(scalar(v));
  const std::size_t n = pop.size();

  // Tournaments: the winner of rank i with probability ((N − i + 1)^k − (N − i)^k)/N^k.
  {
    const std::size_t k = 3;
    const int trials = 400000;
    std::vector<double> wins(n, 0);
    for (int t = 0; t < trials; t++) {
      auto r = draws(tag("check.tournament"), 0, t);
      wins[tournament(pop, k, r)] += 1;
    }
    const auto ord = order(n, by_evaluation(pop));
    double dev = 0, total = 0;
    for (std::size_t i = 0; i < n; i++) {
      const double p = tournament_probability(n, k, i + 1);
      total += p;
      dev = std::max(dev, std::fabs(wins[ord[i]] / trials - p));
    }
    s.check("tournament: the closed form sums to 1", total - 1, 1e-14);
    s.check("tournament of 3: frequencies by rank", dev, 0.003);
  }

  // Order-based selection is invariant under strictly increasing transformations, with the same stream.
  {
    std::vector<Evaluation> transformed;
    for (const auto& e : pop) transformed.push_back(scalar(std::exp(3 * e.value()) - 40));
    bool same = true;
    for (int t = 0; t < 2000; t++) {
      auto a = draws(tag("check.invariance"), 0, t), b = a;
      same = same && tournament(pop, 4, a) == tournament(transformed, 4, b);
    }
    s.require("tournaments are invariant under x ↦ e^{3x} − 40", same && truncation(pop, 4) == truncation(transformed, 4));
  }

  // Ranking.
  {
    const auto lin = linear_ranking(n, 1.8);
    double t = 0;
    for (auto p : lin) t += p;
    s.check("linear ranking sums to 1", t - 1, 1e-14);
    s.check("linear ranking: P(best) = s/N", lin.front() - 1.8 / n, 1e-15);
    s.check("linear ranking: P(worst) = (2 − s)/N", lin.back() - 0.2 / n, 1e-15);
    const auto ex = exponential_ranking(n, 0.7);
    s.check("exponential ranking: successive ratios c", ex[3] / ex[2] - 0.7, 1e-14);
  }

  // Universal sampling: each individual gets ⌊e⌋ or ⌈e⌉ copies, every spin.
  {
    const std::vector<double> w{0.5, 2.25, 1, 0.1, 3.15, 1};
    double total = 0;
    for (double v : w) total += v;
    bool within = true;
    std::vector<double> freq(w.size(), 0);
    const int spins = 20000;
    for (int t = 0; t < spins; t++) {
      auto r = draws(tag("check.sus"), 0, t);
      const std::size_t count = 7;
      std::vector<int> copies(w.size(), 0);
      for (auto i : universal_sampling(w, count, r)) copies[i]++;
      for (std::size_t i = 0; i < w.size(); i++) {
        const double e = w[i] / total * count;
        within = within && copies[i] >= std::floor(e) && copies[i] <= std::ceil(e);
        freq[i] += copies[i];
      }
    }
    s.require("universal sampling: every count within one of its expectation", within);
    double dev = 0;
    for (std::size_t i = 0; i < w.size(); i++) dev = std::max(dev, std::fabs(freq[i] / (7.0 * spins) - w[i] / total));
    s.check("universal sampling: unbiased", dev, 0.003);
    double rw = 0;
    for (int t = 0; t < 200000; t++) { auto r = draws(tag("check.roulette"), 0, t); rw += roulette(w, r) == 4; }
    s.check("roulette: frequency proportional to weight", rw / 200000 - w[4] / total, 0.004);
  }

  // Survivors.
  {
    const std::vector<Evaluation> parents{scalar(4), scalar(1), scalar(6)}, offspring{scalar(5), scalar(0.5), scalar(9), scalar(2)};
    const auto c = comma(offspring, 2), p = plus(parents, offspring, 3), g = generational(parents, offspring, 1);
    s.require("comma keeps the best offspring only", c == std::vector<std::size_t>{1, 3});
    s.require("plus keeps the best of parents and offspring", p == std::vector<std::size_t>{4, 1, 6});
    s.require("generational with one elite: the best parent replaces the worst offspring", g == std::vector<std::size_t>{4, 1, 6, 3});
    s.require("the worst of a population", worst(parents) == 2);
  }

  // Lexicase: two specialists win half the time each; the generalist never.
  {
    const std::vector<std::vector<double>> errors{{0, 9}, {9, 0}, {5, 5}};
    std::vector<double> wins(3, 0);
    const int trials = 40000;
    for (int t = 0; t < trials; t++) {
      auto r = draws(tag("check.lexicase"), 0, t);
      wins[lexicase(errors, r)] += 1;
    }
    s.check("lexicase: a specialist is chosen with probability 1/2", wins[0] / trials - 0.5, 0.01);
    s.require("lexicase: the generalist who is best on no case is never chosen", wins[2] == 0);
    const std::vector<std::vector<double>> close{{1.0, 3}, {1.1, 3}, {4.0, 3}};
    const auto eps = case_epsilons(close);
    s.check("ε-lexicase: ε is the median absolute deviation", eps[0] - 0.1, 1e-12);
    double w01 = 0;
    for (int t = 0; t < 20000; t++) { auto r = draws(tag("check.elexicase"), 0, t); w01 += lexicase(close, r, eps) <= 1; }
    s.require("ε-lexicase keeps both individuals within ε of the best", w01 == 20000);
  }

  return s.done();
}
