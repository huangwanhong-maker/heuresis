// What a representation does before a search commits to it, and tools for checking samplers.
//
// - Representation measures through any decoder (Raidl and Gottlieb 2005, "Empirical analysis of locality, heritability and
//   heuristic bias in evolutionary algorithms", Evolutionary Computation 13(4), 441–475): the mean phenotypic distance a mutation
//   makes (mutation innovation); the share of mutations that leave the phenotype unchanged (redundancy); the mean phenotypic
//   distance from a child to its nearer parent (crossover innovation); and, as a metric form of their crossover loss, the
//   child's excess over the phenotypic segment, d(p₁, c) + d(c, p₂) − d(p₁, p₂), zero when crossover is geometric in the
//   phenotype.
// - Fitness–distance correlation (Jones and Forrest 1995, ICGA, 184–192): the correlation between fitness and the distance to
//   a known optimum over samples.
// - Uniformity of a finite space's sampler: Pearson's χ² over its elements, with z = (χ² − df)/√(2 df), the statistic's
//   standardised excess, large when the sampler is not uniform. Restricted to the values of one size, it checks a Boltzmann
//   sampler's uniformity given size.

#pragma once

#include <heuresis/spaces/space.hpp>

#include <cmath>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace heuresis::spaces {

struct RepresentationMeasures {
  std::size_t samples = 0;
  double mutation_innovation = 0;   // mean phenotypic distance of a mutant from its parent
  double unchanged = 0;             // share of mutations leaving the phenotype unchanged
  double crossover_innovation = 0;  // mean phenotypic distance of a child from its nearer parent
  double crossover_excess = 0;      // mean d(p₁, c) + d(c, p₂) − d(p₁, p₂) in the phenotype
};

template <class P>
RepresentationMeasures representation(const Space& genotype, const std::function<P(const Value&)>& decode,
                                      const std::function<double(const P&, const P&)>& phenotype_distance, calx::random::Stream& s,
                                      double scale = 0.1, std::size_t samples = 500) {
  RepresentationMeasures m;
  m.samples = samples;
  std::size_t same = 0;
  for (std::size_t i = 0; i < samples; i++) {
    const Value x = genotype.sample(s);
    const P px = decode(x);
    const P py = decode(genotype.mutate(x, scale, s));
    const double d = phenotype_distance(px, py);
    m.mutation_innovation += d;
    same += d == 0;
    if (genotype.can_recombine()) {
      const Value y = genotype.sample(s);
      const P q = decode(y);
      const P c = decode(genotype.recombine(x, y, s));
      const double a = phenotype_distance(px, c), b = phenotype_distance(c, q);
      m.crossover_innovation += std::min(a, b);
      m.crossover_excess += a + b - phenotype_distance(px, q);
    }
  }
  const double n = static_cast<double>(samples);
  m.mutation_innovation /= n;
  m.unchanged = static_cast<double>(same) / n;
  m.crossover_innovation /= n;
  m.crossover_excess /= n;
  return m;
}

// The correlation between fitness and distance to a stated optimum, over samples of the space.
inline double fitness_distance_correlation(const Space& sp, const std::function<double(const Value&)>& fitness, const Value& optimum, calx::random::Stream& s,
                                           std::size_t samples = 1000) {
  std::vector<double> f(samples), d(samples);
  for (std::size_t i = 0; i < samples; i++) {
    const Value x = sp.sample(s);
    f[i] = fitness(x), d[i] = sp.distance(x, optimum);
  }
  double mf = 0, md = 0;
  for (std::size_t i = 0; i < samples; i++) mf += f[i], md += d[i];
  mf /= static_cast<double>(samples), md /= static_cast<double>(samples);
  double sfd = 0, sff = 0, sdd = 0;
  for (std::size_t i = 0; i < samples; i++) sfd += (f[i] - mf) * (d[i] - md), sff += (f[i] - mf) * (f[i] - mf), sdd += (d[i] - md) * (d[i] - md);
  return sfd / std::sqrt(sff * sdd);
}

struct Uniformity {
  std::size_t cells = 0, draws = 0;
  double chi2 = 0, z = 0;   // z = (χ² − df)/√(2 df)
  std::size_t outside = 0;  // draws that were not among the listed values
};

// χ² of draws over the listed values; `keep` restricts both the values and the draws (to values of one size, say), and the
// draws made by `draw` (by default the space's sampler).
inline Uniformity uniformity(const Space& sp, calx::random::Stream& s, std::size_t draws, const std::function<bool(const Value&)>& keep = {},
                             const std::function<Value(calx::random::Stream&)>& draw = {}, std::size_t limit = 100000) {
  Uniformity u;
  std::map<std::string, std::size_t> count;
  for (const auto& v : sp.elements(limit))
    if (!keep || keep(v)) count[key(v)] = 0;
  u.cells = count.size();
  for (std::size_t i = 0; i < draws;) {
    const Value v = draw ? draw(s) : sp.sample(s);
    if (keep && !keep(v)) continue;
    i++;
    auto it = count.find(key(v));
    if (it == count.end()) u.outside++;
    else it->second++;
  }
  u.draws = draws;
  const double e = static_cast<double>(draws) / static_cast<double>(u.cells);
  for (const auto& [k, c] : count) u.chi2 += (static_cast<double>(c) - e) * (static_cast<double>(c) - e) / e;
  const double df = static_cast<double>(u.cells) - 1;
  u.z = df > 0 ? (u.chi2 - df) / std::sqrt(2 * df) : 0;
  return u;
}

}  // namespace heuresis::spaces
