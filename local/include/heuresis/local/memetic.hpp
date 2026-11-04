// Memetic refinement of real vectors by calx's local minimisers.
//
// A memetic algorithm improves the individuals of a global search by local search (Moscato 1989, "On evolution,
// search, optimization, genetic algorithms and martial arts: towards memetic algorithms", Caltech Concurrent
// Computation Program report 826; Hinton and Nowlan 1987, "How learning can guide evolution", Complex Systems 1,
// 495–502). Two ways of using what the local search found:
// - Baldwinian: the individual keeps its genotype and is scored by the value its refinement reaches, so the
//   global search sees a smoothed landscape. This fits the protocol as an objective: `baldwinian(f)` is an
//   objective whose cost is the number of evaluations of f the refinement spent.
// - Lamarckian: the refined point replaces the individual. Since an evaluation does not return a genotype, the
//   caller refines with `refine` and injects the refined point (search/protocol.hpp's inject), its origin the
//   operation `local.memetic.refine` with the original's ticket as its parent.
// The refinement is calx's Nelder–Mead (without derivatives) or BFGS with central differences; they own their
// loop and call f directly, which is why they serve inside an evaluation rather than as ask–tell algorithms.

#pragma once

#include <heuresis/search/protocol.hpp>
#include <heuresis/search/draws.hpp>

#include <calx/optimisation/nelder_mead.hpp>
#include <calx/optimisation/quasi_newton.hpp>

#include <functional>
#include <span>
#include <vector>

namespace heuresis::local {

enum class Refiner : std::uint8_t { nelder_mead, bfgs };

struct Refined {
  std::vector<double> x;
  double value = 0;
  std::size_t evaluations = 0;  // of f, every call counted (BFGS's differences included)
};

struct RefineOptions {
  Refiner method = Refiner::nelder_mead;
  int max_evaluations = 0;  // Nelder–Mead's budget, 0 for calx's default
  double initial_step = 0;  // Nelder–Mead's initial step, 0 for calx's default
};

inline Refined refine(const std::function<double(std::span<const double>)>& f, std::vector<double> x, const RefineOptions& o = RefineOptions{}) {
  std::size_t calls = 0;
  auto counted = [&](std::span<const double> y) {
    calls++;
    return f(y);
  };
  Refined r;
  if (o.method == Refiner::nelder_mead) {
    calx::optimisation::NelderMeadOptions nm;
    nm.max_evaluations = o.max_evaluations;
    if (o.initial_step > 0) nm.initial_step = {o.initial_step};
    const auto res = calx::optimisation::nelder_mead(counted, std::move(x), nm);
    r.x = res.x;
    r.value = res.value;
  } else {
    const auto res = calx::optimisation::bfgs(counted, std::move(x));
    r.x = res.x;
    r.value = res.value;
  }
  r.evaluations = calls;
  return r;
}

// The Baldwinian objective: the value refinement reaches from x, at the cost of its evaluations.
inline std::function<heuresis::search::Evaluation(const std::vector<double>&)> baldwinian(std::function<double(std::span<const double>)> f, RefineOptions o = RefineOptions{}) {
  return [f = std::move(f), o](const std::vector<double>& x) {
    const Refined r = refine(f, x, o);
    return heuresis::search::scalar(r.value, static_cast<double>(r.evaluations));
  };
}

inline constexpr std::uint32_t refine_tag = heuresis::search::tag("local.memetic.refine");

// The Lamarckian step: the refined individual, ready to inject; its ticket is the caller's to give.
inline heuresis::search::Individual<std::vector<double>> lamarckian(const std::function<double(std::span<const double>)>& f, const heuresis::search::Individual<std::vector<double>>& x, heuresis::search::Ticket ticket, const RefineOptions& o = RefineOptions{}) {
  const Refined r = refine(f, x.genotype, o);
  return {ticket, r.x, heuresis::search::scalar(r.value, static_cast<double>(r.evaluations)), {refine_tag, {x.ticket}}};
}

}  // namespace heuresis::local
