// The kernels the multiobjective algorithms compose, and their usual choice on real vectors.
//
// An algorithm of this module is generic in its genotype: it is given a sampler, a crossover (two parents, two
// children) and a mutation, as std::functions (decision 0003 §2.7). On a box of real vectors the usual choice is
// Deb's: bounded simulated binary crossover with probability p_c (η_c = 20 for NSGA-II, 30 for NSGA-III) and
// bounded polynomial mutation at rate 1/n (η_m = 20) (Deb et al. 2002; Deb and Jain 2014, §VI).

#pragma once

#include <heuresis/search/protocol.hpp>
#include <heuresis/variation/real.hpp>

#include <calx/random/stream.hpp>

#include <functional>
#include <utility>

namespace heuresis::multiobjective {

template <class G>
struct Kernels {
  std::function<G(calx::random::Stream&)> initialise;
  std::function<std::pair<G, G>(const G&, const G&, calx::random::Stream&)> crossover;
  std::function<G(const G&, calx::random::Stream&)> mutate;
};

inline Kernels<heuresis::variation::Real> real_kernels(const heuresis::variation::Box& box, double eta_c = 20, double eta_m = 20, double crossover_probability = 0.9, double mutation_rate = -1) {
  box.validate();
  Kernels<heuresis::variation::Real> k;
  k.initialise = [box](calx::random::Stream& s) { return heuresis::variation::uniform_point(box, s); };
  k.crossover = [box, eta_c, crossover_probability](const heuresis::variation::Real& a, const heuresis::variation::Real& b, calx::random::Stream& s) {
    if (s.uniform() >= crossover_probability) return std::make_pair(a, b);
    return heuresis::variation::simulated_binary_crossover(a, b, box, eta_c, s);
  };
  k.mutate = [box, eta_m, mutation_rate](const heuresis::variation::Real& x, calx::random::Stream& s) { return heuresis::variation::polynomial_mutation(x, box, eta_m, s, mutation_rate); };
  return k;
}

}  // namespace heuresis::multiobjective
