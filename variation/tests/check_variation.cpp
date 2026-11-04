// Checks of heuresis variation: repairs against their closed forms; the laws of the bit mutations against the
// binomial and the power law; the equivariance of bit mutation and uniform crossover under translations of
// (ℤ/2)ⁿ, exactly; what each crossover preserves (complements, positions, order, sums), exactly; Rudolph's mean
// step; Kendall's and Cayley's distances against brute force.

#include <heuresis/search/draws.hpp>
#include <heuresis/variation/variation.hpp>

#include <calx/verification/check.hpp>

#include <cmath>
#include <map>
#include <set>

using namespace heuresis::variation;
using heuresis::search::Draws;
using heuresis::search::tag;

namespace {

double binomial_pmf(int n, int k, double p) { return std::exp(std::lgamma(n + 1.0) - std::lgamma(k + 1.0) - std::lgamma(n - k + 1.0) + k * std::log(p) + (n - k) * std::log1p(-p)); }

std::uint64_t kendall_brute(const Permutation& p, const Permutation& q) {
  const auto pq = inverse(q);
  std::uint64_t d = 0;
  for (std::size_t i = 0; i < p.size(); i++)
    for (std::size_t j = i + 1; j < p.size(); j++) d += pq[p[i]] > pq[p[j]];
  return d;
}

// The least number of transpositions from p to q, by breadth-first search on Sₙ.
std::size_t cayley_brute(const Permutation& p, const Permutation& q) {
  std::map<Permutation, std::size_t> dist{{p, 0}};
  std::vector<Permutation> frontier{p};
  while (!dist.count(q)) {
    std::vector<Permutation> next;
    for (const auto& x : frontier)
      for (std::size_t i = 0; i < x.size(); i++)
        for (std::size_t j = i + 1; j < x.size(); j++) {
          Permutation y = x;
          std::swap(y[i], y[j]);
          if (dist.emplace(y, dist[x] + 1).second) next.push_back(y);
        }
    frontier = std::move(next);
  }
  return dist[q];
}

}  // namespace

int main() {
  calx::verification::Suite s;
  const Draws draws(2026);

  // Repairs.
  {
    const Box b = Box::cube(1, 0, 1);
    auto r = draws(tag("check.repair"), 0, 0);
    auto rep = [&](double x, Repair h) { Real v{x}; repair(v, b, h, r); return v[0]; };
    s.check("reflect 1.3 into [0, 1] is 0.7", rep(1.3, Repair::reflect) - 0.7, 1e-15);
    s.check("reflect −0.2 is 0.2", rep(-0.2, Repair::reflect) - 0.2, 1e-15);
    s.check("reflect 2.3 is 0.3", rep(2.3, Repair::reflect) - 0.3, 1e-15);
    s.check("wrap 1.3 is 0.3", rep(1.3, Repair::wrap) - 0.3, 1e-15);
    s.check("wrap −0.25 is 0.75", rep(-0.25, Repair::wrap) - 0.75, 1e-15);
    s.require("clip and resample land in the box", rep(5, Repair::clip) == 1 && rep(-5, Repair::resample) >= 0 && rep(-5, Repair::resample) <= 1);
  }

  // Standard bit mutation: the number of flips is Bin(n, p), each bit flipped with probability p.
  {
    const int n = 40, trials = 200000;
    const double p = 1.0 / n;
    std::vector<double> count(n + 1, 0), per_bit(n, 0);
    for (int t = 0; t < trials; t++) {
      auto r = draws(tag("check.sbm"), 0, t);
      Bits x(n, 0);
      count[standard_bit_mutation(x, p, r)] += 1;
      for (int i = 0; i < n; i++) per_bit[i] += x[i];
    }
    double dev = 0, bitdev = 0;
    for (int k = 0; k <= 6; k++) dev = std::max(dev, std::fabs(count[k] / trials - binomial_pmf(n, k, p)));
    for (int i = 0; i < n; i++) bitdev = std::max(bitdev, std::fabs(per_bit[i] / trials - p));
    s.check("standard bit mutation: flips follow Bin(n, p)", dev, 0.004);
    s.check("standard bit mutation: each bit flips with probability p", bitdev, 0.002);

    std::vector<double> pos(n + 1, 0);
    for (int t = 0; t < trials; t++) {
      auto r = draws(tag("check.sbm+"), 0, t);
      Bits x(n, 0);
      pos[standard_bit_mutation_positive(x, p, r)] += 1;
    }
    const double z = 1 - std::pow(1 - p, n);
    double devp = std::fabs(pos[0]);
    for (int k = 1; k <= 6; k++) devp = std::max(devp, std::fabs(pos[k] / trials - binomial_pmf(n, k, p) / z));
    s.check("the positive variant: Bin(n, p) conditioned on at least one flip", devp, 0.004);
  }

  // Exactly k flips, uniform over positions.
  {
    const int n = 12, k = 5, trials = 120000;
    std::vector<double> freq(n, 0);
    bool exact = true;
    for (int t = 0; t < trials; t++) {
      auto r = draws(tag("check.flipk"), 0, t);
      Bits x(n, 0);
      flip_exactly(x, (t % 2) ? k : n - k, r);
      exact = exact && ones(x) == static_cast<std::size_t>((t % 2) ? k : n - k);
      if (t % 2) for (int i = 0; i < n; i++) freq[i] += x[i];
    }
    double dev = 0;
    for (int i = 0; i < n; i++) dev = std::max(dev, std::fabs(freq[i] / (trials / 2) - static_cast<double>(k) / n));
    s.require("flip_exactly flips exactly k bits (both of Floyd's branches)", exact);
    s.check("flip_exactly: each position equally likely", dev, 0.01);
  }

  // The power law of fmut_β.
  {
    const PowerLaw law(100, 1.5);
    double total = 0;
    for (std::size_t a = 1; a <= 50; a++) total += law.probability(a);
    s.check("power law: the probabilities sum to 1", total - 1, 1e-12);
    s.check("power law: P(1)/P(2) = 2^β", law.probability(1) / law.probability(2) - std::pow(2, 1.5), 1e-12);
    double freq1 = 0;
    for (int t = 0; t < 100000; t++) { auto r = draws(tag("check.law"), 0, t); freq1 += law.draw(r) == 1; }
    s.check("power law: the frequency of α = 1", freq1 / 100000 - law.probability(1), 0.005);
  }

  // Equivariance under translations x ↦ x ⊕ m, exactly, with the same stream.
  {
    bool mut = true, cross = true;
    for (int t = 0; t < 200; t++) {
      auto r0 = draws(tag("check.eq.x"), 0, t);
      const Bits x = random_bits(33, r0), y = random_bits(33, r0), m = random_bits(33, r0);
      Bits xm = x, ym = y;
      for (int i = 0; i < 33; i++) { xm[i] ^= m[i]; ym[i] ^= m[i]; }
      auto r1 = draws(tag("check.eq.m"), 0, t), r2 = r1;
      Bits a = x, b = xm;
      standard_bit_mutation(a, 0.1, r1);
      standard_bit_mutation(b, 0.1, r2);
      for (int i = 0; i < 33; i++) mut = mut && (a[i] ^ m[i]) == b[i];
      auto r3 = draws(tag("check.eq.c"), 0, t), r4 = r3;
      auto c = uniform_crossover(x, y, r3), d = uniform_crossover(xm, ym, r4);
      for (int i = 0; i < 33; i++) cross = cross && (c.first[i] ^ m[i]) == d.first[i] && (c.second[i] ^ m[i]) == d.second[i];
    }
    s.require("standard bit mutation commutes with x ↦ x ⊕ m", mut);
    s.require("uniform crossover commutes with x ↦ x ⊕ m", cross);
  }

  // k-point crossover: complementary children, at most k switches.
  {
    bool ok = true;
    for (int t = 0; t < 500; t++) {
      auto r = draws(tag("check.kpoint"), 0, t);
      const std::size_t k = 1 + t % 4;
      std::vector<int> a(17, 0), b(17, 1);
      auto c = k_point_crossover(a, b, k, r);
      std::size_t switches = 0;
      for (std::size_t i = 0; i < 17; i++) {
        ok = ok && c.first[i] + c.second[i] == 1;
        if (i > 0) switches += c.first[i] != c.first[i - 1];
      }
      ok = ok && switches == k;
    }
    s.require("k-point crossover: complementary children with exactly k switches", ok);
  }

  // Real crossovers and mutations.
  {
    const Box wide = Box::cube(5, -1e9, 1e9), unit = Box::cube(5, 0, 1);
    double sum_err = 0;
    bool inside = true, blend = true, line = true;
    for (int t = 0; t < 1000; t++) {
      auto r = draws(tag("check.real"), 0, t);
      Real x = uniform_point(unit, r), y = uniform_point(unit, r);
      auto c = simulated_binary_crossover(x, y, wide, 15, r);
      for (int i = 0; i < 5; i++) sum_err = std::max(sum_err, std::fabs(c.first[i] + c.second[i] - x[i] - y[i]));
      auto d = simulated_binary_crossover(x, y, unit, 2, r);
      inside = inside && unit.contains(d.first) && unit.contains(d.second) && unit.contains(polynomial_mutation(x, unit, 20, r, 1.0));
      auto e = blend_crossover(x, y, 0.5, r);
      for (int i = 0; i < 5; i++) {
        const double lo = std::min(x[i], y[i]), hi = std::max(x[i], y[i]), w = hi - lo;
        blend = blend && e.first[i] >= lo - 0.5 * w && e.first[i] <= hi + 0.5 * w;
      }
      Real z = iso_line(x, y, 0, 0.3, r);
      // collinear: z − x parallel to y − x
      double tz = 0, den = 0;
      for (int i = 0; i < 5; i++) { tz += (z[i] - x[i]) * (y[i] - x[i]); den += (y[i] - x[i]) * (y[i] - x[i]); }
      tz /= den;
      for (int i = 0; i < 5; i++) line = line && std::fabs(z[i] - x[i] - tz * (y[i] - x[i])) < 1e-12;
    }
    s.check("SBX away from the bounds keeps the parents' sum", sum_err, 1e-12);
    s.require("bounded SBX and polynomial mutation stay in the box", inside);
    s.require("BLX-α stays in the widened interval", blend);
    s.require("iso+line without isotropic noise stays on the parents' line", line);

    const std::vector<Real> parents{{1, 2}, {3, 6}, {5, 10}};
    const std::vector<double> w{1, 2, 1};
    const Real m = intermediate_recombination(parents, w);
    s.check("weighted intermediate recombination", std::fabs(m[0] - 3) + std::fabs(m[1] - 6), 1e-15);

    double mean = 0, var = 0;
    const int N = 200000;
    for (int t = 0; t < N; t++) {
      auto r = draws(tag("check.gauss"), 0, t);
      const double v = gaussian_mutation(Real{0.0}, 0.5, r)[0];
      mean += v;
      var += v * v;
    }
    s.check("Gaussian mutation: mean 0", mean / N, 0.005);
    s.check("Gaussian mutation: variance σ²", var / N - 0.25, 0.005);
  }

  // Integers: bounds kept; Rudolph's mean step.
  {
    const IntegerBox b = IntegerBox::cube(4, -1000, 1000);
    const double sigma = 6;
    double step = 0;
    bool in = true;
    const int N = 50000;
    for (int t = 0; t < N; t++) {
      auto r = draws(tag("check.int"), 0, t);
      Integers x{0, 0, 0, 0};
      Integers y = geometric_mutation(x, b, sigma, r);
      for (auto v : y) step += std::fabs(static_cast<double>(v));
      in = in && b.contains(reset_mutation(x, IntegerBox::cube(4, -2, 2), 0.5, r)) && IntegerBox::cube(4, -1, 1).contains(creep_mutation(x, IntegerBox::cube(4, -1, 1), 3, 1, r));
    }
    s.check("Rudolph's mutation: mean ℓ¹ step σ", step / N / sigma - 1, 0.02);
    s.require("reset and creep keep their bounds", in);
  }

  // Permutations: every operator gives a permutation; what each crossover keeps.
  {
    bool perms = true, cx_positions = true, pmx_segment = true, ox_order = true;
    for (int t = 0; t < 2000; t++) {
      auto r = draws(tag("check.perm"), 0, t);
      const std::size_t n = 2 + t % 11;
      Permutation p = random_permutation(n, r), q = random_permutation(n, r);
      for (auto& m : {swap_mutation(p, r), inversion_mutation(p, r), insertion_mutation(p, r), scramble_mutation(p, r)}) perms = perms && is_permutation(m);
      auto pm = partially_mapped_crossover(p, q, r), ox = order_crossover(p, q, r), cx = cycle_crossover(p, q, r);
      perms = perms && is_permutation(pm.first) && is_permutation(pm.second) && is_permutation(ox.first) && is_permutation(ox.second) && is_permutation(cx.first) && is_permutation(cx.second);
      for (std::size_t i = 0; i < n; i++) cx_positions = cx_positions && (cx.first[i] == p[i] || cx.first[i] == q[i]) && (cx.second[i] == p[i] || cx.second[i] == q[i]);
      // PMX: the first child agrees with p on a contiguous segment, and elsewhere takes q's value where q's value is free.
      std::size_t agree = 0;
      for (std::size_t i = 0; i < n; i++) agree += pm.first[i] == p[i] || pm.first[i] == q[i];
      pmx_segment = pmx_segment && agree >= 2;
      // OX: the values of the first child outside its kept segment appear in q's relative order. Check via the
      // relative order of all of the child's values not equal to p at their position.
      std::vector<std::uint32_t> rest;
      for (std::size_t i = 0; i < n; i++) if (ox.first[i] != p[i]) rest.push_back(ox.first[i]);
      const Permutation qi = inverse(q);
      // rest, read cyclically, must be increasing in q-position up to one rotation
      std::size_t descents = 0;
      for (std::size_t i = 0; i + 1 < rest.size(); i++) descents += qi[rest[i]] > qi[rest[i + 1]];
      ox_order = ox_order && descents <= 1;
    }
    s.require("every permutation operator gives a permutation", perms);
    s.require("cycle crossover keeps every position from a parent", cx_positions);
    s.require("PMX takes each of at least two positions from a parent", pmx_segment);
    s.require("order crossover keeps the other parent's relative order (cyclically)", ox_order);
  }

  // Distances on permutations against brute force.
  {
    bool kendall = true, cayley = true;
    for (int t = 0; t < 300; t++) {
      auto r = draws(tag("check.dist"), 0, t);
      const std::size_t n = 1 + t % 6;
      Permutation p = random_permutation(n, r), q = random_permutation(n, r);
      kendall = kendall && kendall_distance(p, q) == kendall_brute(p, q);
      cayley = cayley && cayley_distance(p, q) == cayley_brute(p, q);
    }
    s.require("Kendall's distance is the number of discordant pairs", kendall);
    s.require("Cayley's distance is the least number of transpositions (breadth-first on S₆)", cayley);
    s.require("Hamming distance", hamming(std::vector<int>{1, 0, 1, 1}, std::vector<int>{1, 1, 1, 0}) == 2);
  }

  return s.done();
}
