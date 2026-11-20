// Noisy evaluators (decision 0010; the survey, Part V, design §2–4): how many replicates, where they are worth spending, races
// that select with a stated guarantee, and common random numbers kept only where they help.
//
// - Replicates for a difference: n = ⌈2 (z_{1−α/2} + z_{1−β})² σ²/δ²⌉ per arm tells a difference δ between two means at two-sided
//   level α with power 1 − β, for a known standard deviation σ (the normal approximation; any text on sample size, e.g. Chow, Shao
//   and Wang 2008, Sample Size Calculations in Clinical Research, §3.2). Normal quantiles are tyche's.
// - In doubt: two candidates whose means' intervals overlap; replicates are spent there and nowhere else.
// - Races (Maron and Moore 1994, "Hoeffding races", NIPS 6; Mnih, Szepesvári and Audibert 2008, "Empirical Bernstein stopping",
//   ICML): every surviving candidate is sampled once a round; a candidate is eliminated when its lower bound lies above the least
//   upper bound. Values lie in a declared range [a, b] of width R. After n samples a candidate's bound is
//     Hoeffding: R √(log(2/δ′)/(2n));
//     empirical Bernstein: √(2 V log(3/δ′)/n) + 3R log(3/δ′)/n, V the empirical variance (Audibert, Munos and Szepesvári 2009,
//     Theoret. Comput. Sci. 410, 1876–1902, Thm 1),
//   with δ′ = δ/(kN) spread over k candidates and at most N rounds, so that with probability at least 1 − δ every bound holds at
//   once and the least mean is never eliminated: a race that ends with one survivor has selected the best with that probability.
// - Common random numbers: the variance of a difference is compared with paired streams and with independent ones, and the
//   pairing kept only when it is lower.

#pragma once

#include <heuresis/assessment/record.hpp>

#include <tyche/probability/distributions.hpp>

#include <cmath>
#include <functional>
#include <limits>
#include <string>
#include <vector>

namespace heuresis::assessment {

struct Summary {
  std::size_t n = 0;
  double mean = 0, sd = 0, se = 0;
};

inline Summary summarise(std::span<const double> x) {
  Summary s;
  s.n = x.size();
  if (s.n == 0) return s;
  for (double v : x) s.mean += v;
  s.mean /= static_cast<double>(s.n);
  if (s.n > 1) {
    double ss = 0;
    for (double v : x) ss += (v - s.mean) * (v - s.mean);
    s.sd = std::sqrt(ss / static_cast<double>(s.n - 1));
    s.se = s.sd / std::sqrt(static_cast<double>(s.n));
  }
  return s;
}

inline std::size_t replicates_for(double sd, double smallest_difference, double alpha = 0.05, double power = 0.8) {
  if (!(sd >= 0) || !(smallest_difference > 0)) throw Refusal("replicates for a difference need a standard deviation and a positive difference");
  if (!(alpha > 0 && alpha < 1) || !(power > 0 && power < 1)) throw Refusal("replicates for a difference need a level and a power in (0, 1)");
  const double z = tyche::probability::Normal::standard_quantile(1 - alpha / 2) + tyche::probability::Normal::standard_quantile(power);
  return static_cast<std::size_t>(std::ceil(2 * z * z * sd * sd / (smallest_difference * smallest_difference)));
}

inline bool in_doubt(const Summary& a, const Summary& b, double z = 1.96) {
  return std::abs(a.mean - b.mean) <= z * (a.se + b.se);
}

enum class Bound { hoeffding, bernstein };

struct Race {
  std::size_t best = 0;                    // the survivor with the least mean
  std::vector<std::size_t> survivors;
  std::vector<std::size_t> samples;        // per candidate
  std::vector<double> means;
  std::size_t total = 0;
  bool decided = false;                    // one survivor remained
  std::string guarantee;
};

// A race to the least mean among k candidates; draw(i, r) is candidate i's r-th sample, in [lo, hi].
inline Race race(std::size_t k, const std::function<double(std::size_t, std::size_t)>& draw, double lo, double hi, double delta, std::size_t rounds,
                 Bound bound = Bound::hoeffding) {
  if (k == 0 || rounds == 0) throw Refusal("a race needs candidates and rounds");
  if (!(hi > lo)) throw Refusal("a race needs a range with hi > lo");
  if (!(delta > 0 && delta < 1)) throw Refusal("a race needs δ in (0, 1)");
  const double R = hi - lo;
  const double d1 = delta / (static_cast<double>(k) * static_cast<double>(rounds));
  Race r;
  r.samples.assign(k, 0);
  r.means.assign(k, 0);
  std::vector<double> sum(k, 0), sum2(k, 0);
  std::vector<char> alive(k, 1);
  std::size_t living = k;
  for (std::size_t round = 0; round < rounds && living > 1; round++) {
    for (std::size_t i = 0; i < k; i++) {
      if (!alive[i]) continue;
      const double v = draw(i, round);
      if (!(v >= lo && v <= hi)) throw Refusal("a race's sample fell outside its declared range");
      sum[i] += v, sum2[i] += v * v, r.samples[i]++, r.total++;
    }
    const double n = static_cast<double>(round + 1);
    std::vector<double> half(k, 0);
    double least_upper = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < k; i++) {
      if (!alive[i]) continue;
      const double mean = sum[i] / n;
      const double var = std::max(0.0, sum2[i] / n - mean * mean);
      half[i] = bound == Bound::hoeffding ? R * std::sqrt(std::log(2 / d1) / (2 * n)) : std::sqrt(2 * var * std::log(3 / d1) / n) + 3 * R * std::log(3 / d1) / n;
      least_upper = std::min(least_upper, mean + half[i]);
    }
    for (std::size_t i = 0; i < k; i++)
      if (alive[i] && sum[i] / n - half[i] > least_upper) alive[i] = 0, living--;
  }
  for (std::size_t i = 0; i < k; i++) {
    r.means[i] = r.samples[i] ? sum[i] / static_cast<double>(r.samples[i]) : 0;
    if (alive[i]) r.survivors.push_back(i);
  }
  r.best = r.survivors[0];
  for (auto i : r.survivors)
    if (r.means[i] < r.means[r.best]) r.best = i;
  r.decided = r.survivors.size() == 1;
  r.guarantee = std::string(bound == Bound::hoeffding ? "Hoeffding" : "empirical Bernstein") + " bounds at δ = " + std::to_string(delta) +
                (r.decided ? ": the least mean was selected with probability at least 1 − δ" : ": undecided after the rounds allowed; the least sample mean among the survivors is not guaranteed");
  return r;
}

struct Pairing {
  double independent = 0, common = 0;   // the variance of a difference without and with common random numbers
  bool keep() const { return common < independent; }
};

// a(stream) and b(stream): two evaluators drawing from a stream identified by an integer.
inline Pairing common_numbers(const std::function<double(std::uint64_t)>& a, const std::function<double(std::uint64_t)>& b, std::size_t n) {
  if (n < 2) throw Refusal("common random numbers are judged on at least two pairs");
  std::vector<double> paired, apart;
  for (std::size_t i = 0; i < n; i++) {
    paired.push_back(a(i) - b(i));
    apart.push_back(a(i) - b(n + i));
  }
  const auto sp = summarise(paired), sa = summarise(apart);
  return {sa.sd * sa.sd, sp.sd * sp.sd};
}

}  // namespace heuresis::assessment
