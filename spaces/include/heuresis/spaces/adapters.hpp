// A space's operators as heuresis's evolutionary loop takes them (decision 0006): samples as the maker, mutations at several
// scales (so that the loop's schedule credits the scale that pays), recombination as the crossover where the space has one, the
// distance divided by the space's unit (so that radii of crowding, clearing and novelty are stated in units), and the exact key.

#pragma once

#include <heuresis/spaces/space.hpp>

#include <heuresis/evolution/evolution.hpp>

#include <cstdio>
#include <string>
#include <vector>

namespace heuresis::spaces {

struct OperatorOptions {
  std::vector<double> scales = {0.01, 0.1, 1};   // one mutation operator per scale
  double size = 1;                               // passed to the sampler
  bool crossover = true;
};

inline evolution::Operators<Value> operators(SpacePtr sp, const OperatorOptions& o = {}) {
  if (!sp) throw Refusal("operators need a space");
  if (o.scales.empty()) throw Refusal("operators need at least one mutation scale");
  evolution::Operators<Value> ops;
  using Ctx = evolution::Context<Value>;
  using M = evolution::Member<Value>;
  const double size = o.size;
  ops.makers.push_back({"sample", [sp, size](std::size_t k, const Ctx&, calx::random::Stream& s) {
                          std::vector<Value> out;
                          for (std::size_t i = 0; i < k; i++) out.push_back(sp->sample(s, size));
                          return out;
                        }, 1.0});
  for (double scale : o.scales) {
    char name[48];
    std::snprintf(name, sizeof name, "mutate at %g", scale);
    ops.mutations.push_back({name, [sp, scale](const M& p, const Ctx&, calx::random::Stream& s) -> std::optional<Value> {
                               return sp->mutate(p.genotype, scale, s);
                             }, 1.0});
  }
  if (o.crossover && sp->can_recombine())
    ops.crossovers.push_back({"recombine", [sp](const M& a, const M& b, const Ctx&, calx::random::Stream& s) -> std::optional<Value> {
                                return sp->recombine(a.genotype, b.genotype, s);
                              }, 1.0});
  const double u = sp->unit() > 0 ? sp->unit() : 1;
  ops.distance = [sp, u](const Value& a, const Value& b) { return sp->distance(a, b) / u; };
  ops.key = [](const Value& v) { return key(v); };
  return ops;
}

}  // namespace heuresis::spaces
