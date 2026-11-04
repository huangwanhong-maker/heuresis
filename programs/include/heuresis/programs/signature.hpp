// Programs as terms: a signature of sorted symbols, and the sorts its terms can take.
//
// A program is a term of a many-sorted signature Σ: function symbols f : s₁ × … × s_k → s and constants and
// variables of given sorts, so that the programs of sort s are the carrier T_Σ(X)_s of the term algebra over the
// variables X (Goguen, Thatcher, Wagner and Wright 1977, "Initial algebra semantics and continuous algebras",
// J. ACM 24, 68–95). Genetic programming searches T_Σ(X)_s by variation of terms (Koza 1992, Genetic Programming,
// MIT Press); strongly typed genetic programming keeps every term well-sorted (Montana 1995, "Strongly typed
// genetic programming", Evol. Comput. 3, 199–230).
//
// A sort here is a pair (kind, dimension). The kinds are finitely many (quantities, truth values, …); a quantity's
// dimension is an element of the free abelian group ℤ⁷ over the SI base units (physis::units::Dimension). The
// arithmetic symbols are polymorphic in the dimension, and the dimension of a term is computed by a homomorphism
// from its subterms' dimensions:
//
//   fixed      f : s₁ × … × s_k → s, sorts given (sin, exp, log on dimensionless quantities; comparisons);
//   same       f : (κ, D)ᵏ → (κ, D) for every D (addition, subtraction, minimum);
//   product    × : (D₁, D₂) ↦ D₁ + D₂ (in additive notation for ℤ⁷);
//   quotient   ÷ : (D₁, D₂) ↦ D₁ − D₂;
//   power      x^{p/q} : D ↦ (p/q) D, defined when q divides every exponent of D (√ needs even exponents).
//
// So a term is dimensionally consistent exactly when this partial homomorphism is defined on it, and its value
// is then covariant under a change of units: scaling the base units by λ ∈ (ℝ_{>0})⁷ scales a variable of
// dimension D by λ^D and the term's value by λ^{dim t} (the scaling at the root of Buckingham's Π theorem;
// Barenblatt 1996, Scaling, Self-Similarity, and Intermediate Asymptotics, ch. 1).
//
// To generate a term of a given sort and depth, the generator must know which sorts are reachable: G₀ is the set of
// the terminals' sorts and G_d = G_{d−1} ∪ {sort(f(t₁, …, t_k)) : sort(tᵢ) ∈ G_{d−1}}, the sorts of terms of depth
// ≤ d (a terminal has depth 0). Exponents are bounded by |eᵢ| ≤ `bound`, so every G_d is finite and the sequence
// reaches its fixpoint; generation and variation never leave the bounded sorts. The sets are computed when first
// asked for and cached, so a signature is not to be shared between threads that generate.

#pragma once

#include <physis/units/quantity.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace heuresis::programs {

using physis::units::Dimension;

inline constexpr std::uint32_t quantity = 0;  // the kind of real quantities; other kinds are the caller's

struct Sort {
  std::uint32_t kind = quantity;
  Dimension dimension{};
  bool operator==(const Sort&) const = default;
};

inline bool operator<(const Sort& a, const Sort& b) {
  if (a.kind != b.kind) return a.kind < b.kind;
  return a.dimension.e < b.dimension.e;
}

struct SortHash {
  std::size_t operator()(const Sort& s) const {
    std::size_t h = s.kind * 0x9E3779B97F4A7C15ull;
    for (int v : s.dimension.e) h = (h ^ static_cast<std::size_t>(v + 64)) * 0x100000001B3ull;
    return h;
  }
};

inline Sort dimensionless() { return {}; }
inline Sort of(Dimension d) { return {quantity, d}; }

enum class Rule : std::uint8_t { fixed, same, product, quotient, power };

struct Primitive {
  std::string name;
  std::size_t arity = 2;
  Rule rule = Rule::fixed;
  std::vector<Sort> arguments;  // fixed: the argument sorts
  Sort result;                  // fixed: the result sort; same: its kind (the dimension is free)
  int numerator = 1, denominator = 1;  // power: the exponent p/q
  std::function<double(std::span<const double>)> apply;
};

struct Variable {
  std::string name;
  Sort sort;
};

// A constant: fixed (its value), or ephemeral, drawn uniformly in [lower, upper] where it is created (Koza 1992,
// §10.2, the ephemeral random constant).
struct Constant {
  std::string name;
  Sort sort;
  double value = 0;
  bool ephemeral = false;
  double lower = -1, upper = 1;
};

class Signature {
 public:
  explicit Signature(int bound = 3) : bound_(bound) {}

  std::size_t add(Primitive p) {
    if (p.rule == Rule::fixed && p.arguments.size() != p.arity) throw std::invalid_argument("heuresis::programs::Signature: a fixed primitive needs one sort per argument");
    if ((p.rule == Rule::product || p.rule == Rule::quotient) && p.arity != 2) throw std::invalid_argument("heuresis::programs::Signature: products and quotients are binary");
    if (p.rule == Rule::power && (p.arity != 1 || p.denominator == 0 || p.numerator == 0)) throw std::invalid_argument("heuresis::programs::Signature: a power is unary with a nonzero rational exponent");
    if (!p.apply) throw std::invalid_argument("heuresis::programs::Signature: a primitive without an implementation");
    if (!variables_.empty() || !constants_.empty()) throw std::logic_error("heuresis::programs::Signature: add the primitives before the terminals");
    primitives_.push_back(std::move(p));
    levels_.clear();
    return primitives_.size() - 1;
  }
  std::size_t add(Variable v) {
    if (!constants_.empty()) throw std::logic_error("heuresis::programs::Signature: add the variables before the constants");
    variables_.push_back(std::move(v));
    levels_.clear();
    return primitives_.size() + variables_.size() - 1;
  }
  std::size_t add(Constant c) {
    constants_.push_back(std::move(c));
    levels_.clear();
    return symbols() - 1;
  }

  // Symbols are numbered: primitives, then variables, then constants.
  std::size_t symbols() const { return primitives_.size() + variables_.size() + constants_.size(); }
  bool is_primitive(std::size_t s) const { return s < primitives_.size(); }
  bool is_variable(std::size_t s) const { return s >= primitives_.size() && s < primitives_.size() + variables_.size(); }
  bool is_constant(std::size_t s) const { return s >= primitives_.size() + variables_.size() && s < symbols(); }
  std::size_t arity(std::size_t s) const { return is_primitive(s) ? primitives_[s].arity : 0; }
  const Primitive& primitive(std::size_t s) const { return primitives_[s]; }
  const Variable& variable(std::size_t s) const { return variables_[s - primitives_.size()]; }
  const Constant& constant(std::size_t s) const { return constants_[s - primitives_.size() - variables_.size()]; }
  std::size_t variable_index(std::size_t s) const { return s - primitives_.size(); }
  std::size_t primitive_count() const { return primitives_.size(); }
  std::size_t variable_count() const { return variables_.size(); }
  int bound() const { return bound_; }
  std::string name(std::size_t s) const { return is_primitive(s) ? primitive(s).name : is_variable(s) ? variable(s).name : constant(s).name; }

  // The sort of a terminal symbol.
  Sort terminal_sort(std::size_t s) const { return is_variable(s) ? variable(s).sort : constant(s).sort; }

  bool bounded(const Sort& s) const {
    for (int v : s.dimension.e)
      if (v > bound_ || v < -bound_) return false;
    return true;
  }

  // The sort of f(t₁, …, t_k) from the sorts of its arguments: the homomorphism, where it is defined.
  std::optional<Sort> synthesise(std::size_t f, std::span<const Sort> args) const {
    const Primitive& p = primitives_[f];
    if (args.size() != p.arity) return std::nullopt;
    std::optional<Sort> out;
    switch (p.rule) {
      case Rule::fixed:
        for (std::size_t i = 0; i < args.size(); i++)
          if (!(args[i] == p.arguments[i])) return std::nullopt;
        out = p.result;
        break;
      case Rule::same:
        for (const auto& a : args)
          if (!(a == args[0]) || a.kind != p.result.kind) return std::nullopt;
        out = args.empty() ? p.result : args[0];
        break;
      case Rule::product:
        if (args[0].kind != quantity || args[1].kind != quantity) return std::nullopt;
        out = of(args[0].dimension * args[1].dimension);
        break;
      case Rule::quotient:
        if (args[0].kind != quantity || args[1].kind != quantity) return std::nullopt;
        out = of(args[0].dimension / args[1].dimension);
        break;
      case Rule::power: {
        if (args[0].kind != quantity) return std::nullopt;
        Dimension d;
        for (int i = 0; i < 7; i++) {
          const int v = args[0].dimension.e[i] * p.numerator;
          if (v % p.denominator) return std::nullopt;
          d.e[i] = v / p.denominator;
        }
        out = of(d);
        break;
      }
    }
    if (out && !bounded(*out)) return std::nullopt;
    return out;
  }

  // The sorts of terms of depth ≤ d.
  const std::vector<Sort>& generable(std::size_t d) const {
    extend(d);
    return levels_[std::min(d, levels_.size() - 1)].list;
  }
  bool generable(const Sort& s, std::size_t d) const {
    extend(d);
    return levels_[std::min(d, levels_.size() - 1)].set.count(s) != 0;
  }

  // The tuples of argument sorts, each generable at depth ≤ d, for which f yields `target`.
  std::vector<std::vector<Sort>> argument_options(std::size_t f, const Sort& target, std::size_t d) const {
    const Primitive& p = primitives_[f];
    std::vector<std::vector<Sort>> out;
    auto ok = [&](const Sort& s) { return generable(s, d); };
    switch (p.rule) {
      case Rule::fixed:
        if (p.result == target && std::all_of(p.arguments.begin(), p.arguments.end(), ok)) out.push_back(p.arguments);
        break;
      case Rule::same:
        if (target.kind == p.result.kind && ok(target)) out.push_back(std::vector<Sort>(p.arity, target));
        break;
      case Rule::product:
      case Rule::quotient:
        if (target.kind != quantity) break;
        for (const auto& a : generable(d)) {
          if (a.kind != quantity) continue;
          const Sort b = of(p.rule == Rule::product ? target.dimension / a.dimension : a.dimension / target.dimension);
          if (bounded(b) && ok(b)) out.push_back({a, b});
        }
        break;
      case Rule::power: {
        if (target.kind != quantity) break;
        Dimension d0;
        bool integral = true;
        for (int i = 0; i < 7; i++) {
          const int v = target.dimension.e[i] * p.denominator;
          if (v % p.numerator) { integral = false; break; }
          d0.e[i] = v / p.numerator;
        }
        if (integral && ok(of(d0))) out.push_back({of(d0)});
        break;
      }
    }
    return out;
  }

  // The terminal symbols of a sort.
  std::vector<std::size_t> terminals(const Sort& s) const {
    std::vector<std::size_t> out;
    for (std::size_t t = primitives_.size(); t < symbols(); t++)
      if (terminal_sort(t) == s) out.push_back(t);
    return out;
  }

 private:
  struct Level {
    std::unordered_set<Sort, SortHash> set;
    std::vector<Sort> list;  // in the order found
  };

  void insert(Level& l, const Sort& s) const {
    if (l.set.insert(s).second) l.list.push_back(s);
  }

  void extend(std::size_t d) const {
    if (levels_.empty()) {
      Level l0;
      for (std::size_t t = primitives_.size(); t < symbols(); t++)
        if (bounded(terminal_sort(t))) insert(l0, terminal_sort(t));
      levels_.push_back(std::move(l0));
      saturated_ = false;
    }
    while (levels_.size() <= d && !saturated_) {
      const Level& prev = levels_.back();
      Level next = prev;
      for (std::size_t f = 0; f < primitives_.size(); f++) {
        const Primitive& p = primitives_[f];
        if (p.rule == Rule::fixed) {
          if (std::all_of(p.arguments.begin(), p.arguments.end(), [&](const Sort& s) { return prev.set.count(s) != 0; })) insert(next, p.result);
        } else if (p.rule == Rule::same) {
          for (const auto& s : prev.list)
            if (s.kind == p.result.kind) insert(next, s);
        } else if (p.rule == Rule::power) {
          for (const auto& s : prev.list) {
            const Sort a[2] = {s, s};  // one argument; the array is the binary case's size
            if (auto r = synthesise(f, std::span<const Sort>(a, 1))) insert(next, *r);
          }
        } else {
          for (const auto& s : prev.list)
            for (const auto& t : prev.list) {
              const Sort a[2] = {s, t};
              if (auto r = synthesise(f, a)) insert(next, *r);
            }
        }
      }
      saturated_ = next.list.size() == prev.list.size();
      levels_.push_back(std::move(next));
    }
  }

  int bound_;
  std::vector<Primitive> primitives_;
  std::vector<Variable> variables_;
  std::vector<Constant> constants_;
  mutable std::vector<Level> levels_;
  mutable bool saturated_ = false;
};

// The usual arithmetic, typed by dimension: +, −, ×, ÷ (unprotected: a division by zero gives a value that is not
// finite, which a regression reports as a failure), the square, the cube and √; and sin, cos, exp and log on
// dimensionless quantities.
namespace library {
inline Primitive add() { return {"+", 2, Rule::same, {}, of({}), 1, 1, [](std::span<const double> a) { return a[0] + a[1]; }}; }
inline Primitive subtract() { return {"-", 2, Rule::same, {}, of({}), 1, 1, [](std::span<const double> a) { return a[0] - a[1]; }}; }
inline Primitive multiply() { return {"*", 2, Rule::product, {}, of({}), 1, 1, [](std::span<const double> a) { return a[0] * a[1]; }}; }
inline Primitive divide() { return {"/", 2, Rule::quotient, {}, of({}), 1, 1, [](std::span<const double> a) { return a[0] / a[1]; }}; }
inline Primitive square() { return {"sq", 1, Rule::power, {}, of({}), 2, 1, [](std::span<const double> a) { return a[0] * a[0]; }}; }
inline Primitive cube() { return {"cube", 1, Rule::power, {}, of({}), 3, 1, [](std::span<const double> a) { return a[0] * a[0] * a[0]; }}; }
inline Primitive root() { return {"sqrt", 1, Rule::power, {}, of({}), 1, 2, [](std::span<const double> a) { return std::sqrt(a[0]); }}; }
inline Primitive unary(std::string name, double (*f)(double)) {
  return {std::move(name), 1, Rule::fixed, {dimensionless()}, dimensionless(), 1, 1, [f](std::span<const double> a) { return f(a[0]); }};
}
inline Primitive sine() { return unary("sin", [](double x) { return std::sin(x); }); }
inline Primitive cosine() { return unary("cos", [](double x) { return std::cos(x); }); }
inline Primitive exponential() { return unary("exp", [](double x) { return std::exp(x); }); }
inline Primitive logarithm() { return unary("log", [](double x) { return std::log(x); }); }
}  // namespace library

}  // namespace heuresis::programs
