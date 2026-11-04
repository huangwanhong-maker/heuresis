// The kernels the genetic algorithms compose, as run-time values.
//
// An algorithm of this module takes its sampling, mutation and crossover as std::function values (decision
// 0003 §2.7), so that a work can choose them from a registry; these factories wrap the kernels of variation/
// in that form. Each kernel draws only from the stream it is given.

#pragma once

#include <heuresis/variation/variation.hpp>

#include <calx/random/stream.hpp>

#include <functional>
#include <memory>
#include <utility>

namespace heuresis::genetic {

using heuresis::variation::Bits;
using heuresis::variation::Real;

template <class G>
using Sampler = std::function<G(calx::random::Stream&)>;
template <class G>
using Mutation = std::function<G(const G&, calx::random::Stream&)>;
template <class G>
using Crossover = std::function<std::pair<G, G>(const G&, const G&, calx::random::Stream&)>;

inline Sampler<Bits> random_bits(std::size_t n) {
  return [n](calx::random::Stream& s) { return heuresis::variation::random_bits(n, s); };
}

// Standard bit mutation at rate p (1/n when p < 0).
inline Mutation<Bits> bit_flip(double p = -1) {
  return [p](const Bits& x, calx::random::Stream& s) {
    Bits y = x;
    heuresis::variation::standard_bit_mutation(y, p < 0 ? 1.0 / static_cast<double>(x.size()) : p, s);
    return y;
  };
}

// Standard bit mutation conditioned on at least one flip.
inline Mutation<Bits> bit_flip_positive(double p = -1) {
  return [p](const Bits& x, calx::random::Stream& s) {
    Bits y = x;
    heuresis::variation::standard_bit_mutation_positive(y, p < 0 ? 1.0 / static_cast<double>(x.size()) : p, s);
    return y;
  };
}

// One uniformly chosen bit flipped: the step of randomised local search.
inline Mutation<Bits> flip_one() {
  return [](const Bits& x, calx::random::Stream& s) {
    Bits y = x;
    heuresis::variation::flip_exactly(y, 1, s);
    return y;
  };
}

// fmut_β on strings of length n.
inline Mutation<Bits> heavy_tailed(std::size_t n, double beta = 1.5) {
  auto law = std::make_shared<heuresis::variation::PowerLaw>(n, beta);
  return [law](const Bits& x, calx::random::Stream& s) {
    Bits y = x;
    heuresis::variation::heavy_tailed_mutation(y, *law, s);
    return y;
  };
}

template <class T>
Crossover<std::vector<T>> uniform_crossover(double rate = 0.5) {
  return [rate](const std::vector<T>& a, const std::vector<T>& b, calx::random::Stream& s) { return heuresis::variation::uniform_crossover(a, b, s, rate); };
}

template <class T>
Crossover<std::vector<T>> k_point_crossover(std::size_t k) {
  return [k](const std::vector<T>& a, const std::vector<T>& b, calx::random::Stream& s) { return heuresis::variation::k_point_crossover(a, b, k, s); };
}

inline Sampler<Real> uniform_point(heuresis::variation::Box box) {
  return [box = std::move(box)](calx::random::Stream& s) { return heuresis::variation::uniform_point(box, s); };
}

inline Mutation<Real> polynomial(heuresis::variation::Box box, double eta = 20, double rate = -1) {
  return [box = std::move(box), eta, rate](const Real& x, calx::random::Stream& s) { return heuresis::variation::polynomial_mutation(x, box, eta, s, rate); };
}

inline Crossover<Real> simulated_binary(heuresis::variation::Box box, double eta = 15) {
  return [box = std::move(box), eta](const Real& a, const Real& b, calx::random::Stream& s) { return heuresis::variation::simulated_binary_crossover(a, b, box, eta, s); };
}

}  // namespace heuresis::genetic
