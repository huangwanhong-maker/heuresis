// The values of typed spaces: one tree for every genotype, so that heuresis's evolutionary loop, which takes any copyable
// genotype, runs on all of them.
//
// A value is one of: a real vector; an integer vector; a category (its index); a permutation; a set (its indices, sorted);
// a sequence of values; a record of values (the parts of a product, in the product's order); a choice (its branch's index
// and its payload); a point of a manifold (its coordinates in the space's representation: a unit vector for a sphere, a
// unit quaternion for SO(3)). The space a value belongs to says how to read it; the value does not carry its space.
//
// key() writes a value exactly (reals in hexadecimal floating point), so that two values have one key exactly when they are
// equal: the identity the evolutionary loop uses to evaluate a genotype once.

#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace heuresis::spaces {

struct Value {
  enum class Kind { real, integer, category, permutation, set, sequence, record, choice, point };
  Kind kind = Kind::real;
  std::vector<double> reals;          // real, point
  std::vector<std::int64_t> ints;     // integer, category (one entry), permutation, set (sorted), choice (the branch)
  std::vector<Value> parts;           // sequence, record, choice (the payload)

  friend bool operator==(const Value& a, const Value& b) { return a.kind == b.kind && a.reals == b.reals && a.ints == b.ints && a.parts == b.parts; }
  friend bool operator!=(const Value& a, const Value& b) { return !(a == b); }
};

inline Value real_value(std::vector<double> x) { return {Value::Kind::real, std::move(x), {}, {}}; }
inline Value integer_value(std::vector<std::int64_t> x) { return {Value::Kind::integer, {}, std::move(x), {}}; }
inline Value category_value(std::int64_t i) { return {Value::Kind::category, {}, {i}, {}}; }
inline Value permutation_value(std::vector<std::int64_t> p) { return {Value::Kind::permutation, {}, std::move(p), {}}; }
inline Value set_value(std::vector<std::int64_t> s) { return {Value::Kind::set, {}, std::move(s), {}}; }
inline Value sequence_value(std::vector<Value> xs) { return {Value::Kind::sequence, {}, {}, std::move(xs)}; }
inline Value record_value(std::vector<Value> xs) { return {Value::Kind::record, {}, {}, std::move(xs)}; }
inline Value choice_value(std::int64_t branch, Value payload) { return {Value::Kind::choice, {}, {branch}, {std::move(payload)}}; }
inline Value point_value(std::vector<double> x) { return {Value::Kind::point, std::move(x), {}, {}}; }

inline const char* name_of(Value::Kind k) {
  switch (k) {
    case Value::Kind::real: return "real";
    case Value::Kind::integer: return "integer";
    case Value::Kind::category: return "category";
    case Value::Kind::permutation: return "permutation";
    case Value::Kind::set: return "set";
    case Value::Kind::sequence: return "sequence";
    case Value::Kind::record: return "record";
    case Value::Kind::choice: return "choice";
    case Value::Kind::point: return "point";
  }
  return "";
}

// The exact identity of a value.
inline std::string key(const Value& v) {
  std::string s = name_of(v.kind);
  s += '(';
  char buf[40];
  for (double x : v.reals) std::snprintf(buf, sizeof buf, "%a,", x), s += buf;
  s += ';';
  for (auto i : v.ints) s += std::to_string(i) + ",";
  s += ';';
  for (const auto& p : v.parts) s += key(p) + ",";
  s += ')';
  return s;
}

// A value as a person reads it (reals to six significant figures).
inline std::string words(const Value& v) {
  char buf[40];
  std::string s;
  switch (v.kind) {
    case Value::Kind::real:
    case Value::Kind::point:
      s = "(";
      for (std::size_t i = 0; i < v.reals.size(); i++) std::snprintf(buf, sizeof buf, "%s%.6g", i ? ", " : "", v.reals[i]), s += buf;
      return s + ")";
    case Value::Kind::integer:
    case Value::Kind::permutation:
      s = v.kind == Value::Kind::permutation ? "⟨" : "(";
      for (std::size_t i = 0; i < v.ints.size(); i++) s += (i ? ", " : "") + std::to_string(v.ints[i]);
      return s + (v.kind == Value::Kind::permutation ? "⟩" : ")");
    case Value::Kind::category: return "#" + std::to_string(v.ints.empty() ? -1 : v.ints[0]);
    case Value::Kind::set:
      s = "{";
      for (std::size_t i = 0; i < v.ints.size(); i++) s += (i ? ", " : "") + std::to_string(v.ints[i]);
      return s + "}";
    case Value::Kind::sequence:
    case Value::Kind::record:
      s = v.kind == Value::Kind::sequence ? "[" : "{";
      for (std::size_t i = 0; i < v.parts.size(); i++) s += (i ? ", " : "") + words(v.parts[i]);
      return s + (v.kind == Value::Kind::sequence ? "]" : "}");
    case Value::Kind::choice: return "branch " + std::to_string(v.ints.empty() ? -1 : v.ints[0]) + ": " + (v.parts.empty() ? std::string("?") : words(v.parts[0]));
  }
  return s;
}

}  // namespace heuresis::spaces
