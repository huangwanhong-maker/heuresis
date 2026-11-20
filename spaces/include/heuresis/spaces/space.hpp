// The contract of a typed space, after the infrastructure's survey Evolutionary heterogeneous resource allocation (Part III,
// design §1): what every space supplies, what it may supply, and what it declares about itself.
//
// Variation defined by a metric (Moraglio 2007, Towards a Geometric Unification of Evolutionary Algorithms, PhD thesis,
// University of Essex): a mutation draws from a ball of the space's metric about the parent, a geometric crossover from the
// segment between the parents, the points c with d(x, c) + d(c, y) = d(x, y). On a product of spaces, crossing each part
// geometrically is geometric for the weighted sum of the parts' distances, even when the parts are coupled (ibid., Thm 9.2.2
// and Cor. 9.2.3). A space declares a natural unit of distance (the mean or root-mean-square distance between two of its
// samples, or its diameter, said which), so that a product can scale its parts and no wide part dominates.
//
// Required: sample, mutate within a ball, distance, validity. Optional, each declared present or absent so that an algorithm
// that needs one refuses a space without it by name: recombine (with whether it is geometric, pure and respectful), repair
// (idempotent, the identity on valid values), a chart for curved and continuous spaces (log, retract, the trust radius within
// which they invert each other), a standardised embedding in ℝᵏ (bijective, or a surjection with a decoder), a floor on the
// probability that a mutation leaves a discrete value, a diameter, and for finite spaces their elements.
//
// scale ∈ (0, 1] is a mutation's standardised step: 1 moves about as far as the space's unit, small values stay close. A space
// states the ball's radius for a scale where it can bound it.

#pragma once

#include <heuresis/spaces/value.hpp>

#include <calx/random/stream.hpp>

#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace heuresis::spaces {

struct Refusal : std::invalid_argument {
  explicit Refusal(const std::string& why) : std::invalid_argument("heuresis::spaces: " + why) {}
};

enum class Unit { mean, rms, diameter };   // what a space's unit of distance is

struct Properties {
  bool geometric = false;      // every child of recombine lies on the segment between its parents
  bool pure = false;           // recombine(x, x) = x
  bool respectful = false;     // what both parents share, the child keeps (Radcliffe 1994)
  bool exact_metric = true;    // distance is exactly a metric (false where it is a declared approximation)
  Unit unit = Unit::mean;
  bool unit_exact = true;      // the unit is a closed form, not an estimate
};

class Space {
 public:
  virtual ~Space() = default;
  virtual std::string describe() const = 0;

  // ---- required ----
  virtual Value sample(calx::random::Stream& s, double size = 1) const = 0;
  virtual Value mutate(const Value& x, double scale, calx::random::Stream& s) const = 0;
  virtual double distance(const Value& x, const Value& y) const = 0;
  virtual double unit() const = 0;
  virtual std::optional<std::string> invalid(const Value& x) const = 0;
  bool valid(const Value& x) const { return !invalid(x).has_value(); }

  // ---- optional ----
  virtual bool can_recombine() const { return false; }
  virtual Value recombine(const Value&, const Value&, calx::random::Stream&) const { throw Refusal(describe() + " has no recombination"); }
  virtual Properties properties() const { return {}; }
  virtual bool can_repair() const { return false; }
  virtual Value repair(const Value&) const { throw Refusal(describe() + " has no repair"); }
  // The radius of a mutation's ball at a scale, where the space can bound it.
  virtual std::optional<double> mutation_radius(double) const { return std::nullopt; }
  // The least probability that a mutation changes the value (a margin against fixation, Hamano et al. 2022), if guaranteed.
  virtual double change_floor() const { return 0; }
  // A bound on the distance between any two values: the diameter itself where the space knows it, a bound where its
  // header says so, infinity where none is stated.
  virtual double diameter() const { return std::numeric_limits<double>::infinity(); }
  // A chart: the tangent dimension, log_x(y), retract_x(v), and the radius within which they invert each other.
  virtual bool has_chart() const { return false; }
  virtual std::size_t chart_dimension() const { return 0; }
  virtual std::vector<double> log(const Value&, const Value&) const { throw Refusal(describe() + " has no chart"); }
  virtual Value retract(const Value&, std::span<const double>) const { throw Refusal(describe() + " has no chart"); }
  virtual double trust_radius(const Value&) const { return 0; }
  // A standardised embedding in ℝᵏ: unembed after embed is the identity always; bijective means also that every point of
  // ℝᵏ is the embedding of a value (embed after unembed the identity), else unembed is a decoder onto the space.
  virtual std::optional<std::size_t> embedding_dimension() const { return std::nullopt; }
  virtual bool embedding_bijective() const { return false; }
  virtual std::vector<double> embed(const Value&) const { throw Refusal(describe() + " has no embedding"); }
  virtual Value unembed(std::span<const double>) const { throw Refusal(describe() + " has no embedding"); }
  // Finite spaces: how many values, and all of them (up to a limit the caller states).
  virtual std::optional<std::size_t> cardinality() const { return std::nullopt; }
  virtual std::vector<Value> elements(std::size_t limit) const {
    (void)limit;
    throw Refusal(describe() + " is not finite, or is too large to list");
  }
};

using SpacePtr = std::shared_ptr<const Space>;

namespace detail {

inline void kind_is(const Value& x, Value::Kind k, const char* who) {
  if (x.kind != k) throw Refusal(std::string(who) + ": a " + name_of(x.kind) + " value where a " + name_of(k) + " was expected");
}

inline double clamp_scale(double s) { return s <= 0 ? 1e-6 : s > 1 ? 1 : s; }

}  // namespace detail

}  // namespace heuresis::spaces
