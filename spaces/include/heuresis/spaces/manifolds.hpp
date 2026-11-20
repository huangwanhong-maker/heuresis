// Curved spaces, varied by steps in a tangent space and retraction (Absil, Mahony and Sepulchre 2008, Optimization Algorithms on
// Matrix Manifolds, Princeton, ch. 4): the sphere Sⁿ⁻¹ ⊂ ℝⁿ and the rotation group SO(3), each with its geodesic distance, the
// exponential map as retraction, and the logarithm inverting it within the injectivity radius π.
//
// - The sphere. Uniform sampling by normalised Gaussian vectors; the geodesic distance 2·atan2(|x − y|, |x + y|), accurate at
//   both ends; the unit π/2, exactly the mean geodesic distance between two uniform points, since y ↦ −y maps the angle θ to
//   π − θ and preserves the uniform law. Tangent coordinates in the orthonormal basis of x⊥ given by Householder's reflection
//   that maps x to ±e_n. Recombination along the minimal geodesic (spherical linear interpolation, Shoemake 1985,
//   SIGGRAPH '85, 245–254), geometric under the geodesic distance; between antipodes every great semicircle is minimal, and
//   one is drawn at random.
// - SO(3). Values are unit quaternions with the sign fixed (w > 0, or w = 0 and the first non-zero part positive), so that one
//   rotation has one value. physis's rotations give the group law, exp, log, the geodesic distance (the angle of a⁻¹b) and
//   slerp. Uniform (Haar) sampling by normalised Gaussian quaternions (Shoemake 1992, Graphics Gems III, 124–132); the unit
//   π/2 + 2/π, exactly the mean angle of a uniform rotation, whose angle has density (1 − cos θ)/π on [0, π]. Steps act on
//   the right, x·exp(v), in the body's frame.

#pragma once

#include <heuresis/spaces/base.hpp>

#include <physis/geometry/rotation.hpp>

#include <cmath>
#include <numbers>
#include <span>
#include <vector>

namespace heuresis::spaces {

class Sphere final : public Space {
 public:
  // The unit sphere in ℝⁿ (of dimension n − 1).
  explicit Sphere(std::size_t ambient) : n_(ambient) {
    if (n_ < 2) throw Refusal("a sphere needs an ambient dimension of at least two");
  }
  std::size_t ambient() const { return n_; }
  std::string describe() const override { return "the sphere S^" + std::to_string(n_ - 1); }

  Value sample(calx::random::Stream& s, double = 1) const override {
    for (;;) {
      std::vector<double> x(n_);
      double r = 0;
      for (double& v : x) {
        v = s.normal();
        r += v * v;
      }
      r = std::sqrt(r);
      if (r > 1e-12) {
        for (double& v : x) v /= r;
        return point_value(std::move(x));
      }
    }
  }
  Value mutate(const Value& x, double scale, calx::random::Stream& s) const override {
    detail::require(*this, x, "mutate");
    const double sigma = detail::clamp_scale(scale) * unit() / std::sqrt(static_cast<double>(n_ - 1));
    std::vector<double> v(n_ - 1);
    for (double& c : v) c = sigma * s.normal();
    return exp_at(x.reals, v);
  }
  double distance(const Value& x, const Value& y) const override {
    detail::require(*this, x, "distance");
    detail::require(*this, y, "distance");
    return angle(x.reals, y.reals);
  }
  double unit() const override { return std::numbers::pi / 2; }
  std::optional<std::string> invalid(const Value& x) const override {
    if (x.kind != Value::Kind::point) return std::string("a ") + name_of(x.kind) + " value, not a point";
    if (x.reals.size() != n_) return "a point of " + std::to_string(x.reals.size()) + " coordinates, not " + std::to_string(n_);
    double r = 0;
    for (double v : x.reals) {
      if (!std::isfinite(v)) return std::string("a coordinate not finite");
      r += v * v;
    }
    if (std::abs(r - 1) > 1e-9) return "a point of norm " + std::to_string(std::sqrt(r)) + ", not 1";
    return std::nullopt;
  }
  bool can_recombine() const override { return true; }
  Value recombine(const Value& x, const Value& y, calx::random::Stream& s) const override {
    detail::require(*this, x, "recombine");
    detail::require(*this, y, "recombine");
    if (x == y) return x;
    const double t = s.uniform();
    auto v = log_at(x.reals, y.reals);
    if (antipodal(x.reals, y.reals)) {  // every great semicircle is minimal: a direction drawn at random
      double r = 0;
      for (double& c : v) {
        c = s.normal();
        r += c * c;
      }
      r = std::sqrt(r);
      for (double& c : v) c *= std::numbers::pi / r;
    }
    for (double& c : v) c *= t;
    return exp_at(x.reals, v);
  }
  Properties properties() const override { return {true, true, false, true, Unit::mean, true}; }
  bool can_repair() const override { return true; }
  Value repair(const Value& x) const override {
    if (valid(x)) return x;
    if (x.kind != Value::Kind::point || x.reals.size() != n_) throw Refusal("repair on " + describe() + ": a value of another shape");
    return normalised(x.reals);
  }
  double change_floor() const override { return 1; }
  double diameter() const override { return std::numbers::pi; }
  bool has_chart() const override { return true; }
  std::size_t chart_dimension() const override { return n_ - 1; }
  std::vector<double> log(const Value& x, const Value& y) const override {
    detail::require(*this, x, "log");
    detail::require(*this, y, "log");
    return log_at(x.reals, y.reals);
  }
  Value retract(const Value& x, std::span<const double> v) const override {
    detail::require(*this, x, "retract");
    if (v.size() != n_ - 1) throw Refusal("retract on " + describe() + ": a tangent vector of another dimension");
    return exp_at(x.reals, std::vector<double>(v.begin(), v.end()));
  }
  double trust_radius(const Value&) const override { return std::numbers::pi; }
  std::optional<std::size_t> embedding_dimension() const override { return n_; }
  std::vector<double> embed(const Value& x) const override {
    detail::require(*this, x, "embed");
    return x.reals;
  }
  Value unembed(std::span<const double> z) const override {
    if (z.size() != n_) throw Refusal("unembed on " + describe() + ": coordinates of another dimension");
    return normalised(std::vector<double>(z.begin(), z.end()));
  }

 private:
  static double angle(const std::vector<double>& x, const std::vector<double>& y) {
    double a = 0, b = 0;
    for (std::size_t i = 0; i < x.size(); i++) a += (x[i] - y[i]) * (x[i] - y[i]), b += (x[i] + y[i]) * (x[i] + y[i]);
    return 2 * std::atan2(std::sqrt(a), std::sqrt(b));
  }
  static bool antipodal(const std::vector<double>& x, const std::vector<double>& y) {
    double b = 0;
    for (std::size_t i = 0; i < x.size(); i++) b += (x[i] + y[i]) * (x[i] + y[i]);
    return b < 1e-24;
  }
  Value normalised(std::vector<double> x) const {
    double r = 0;
    for (double& v : x) {
      if (!std::isfinite(v)) v = 0;
      r += v * v;
    }
    if (!(r > 0)) {
      x.assign(n_, 0);
      x[0] = 1;
      return point_value(std::move(x));
    }
    r = std::sqrt(r);
    for (double& v : x) v /= r;
    return point_value(std::move(x));
  }
  // The orthonormal basis of x⊥: the first n − 1 columns of Householder's reflection H with H x = ±e_n.
  std::vector<std::vector<double>> basis(const std::vector<double>& x) const {
    std::vector<double> u = x;
    const double sign = x[n_ - 1] >= 0 ? 1.0 : -1.0;
    u[n_ - 1] += sign;
    double uu = 0;
    for (double v : u) uu += v * v;
    std::vector<std::vector<double>> b(n_ - 1, std::vector<double>(n_, 0));
    for (std::size_t j = 0; j + 1 < n_; j++) {
      for (std::size_t i = 0; i < n_; i++) b[j][i] = (i == j ? 1.0 : 0.0) - 2 * u[i] * u[j] / uu;
    }
    return b;
  }
  Value exp_at(const std::vector<double>& x, const std::vector<double>& c) const {
    const auto b = basis(x);
    std::vector<double> v(n_, 0);
    for (std::size_t j = 0; j + 1 < n_; j++)
      for (std::size_t i = 0; i < n_; i++) v[i] += c[j] * b[j][i];
    double t = 0;
    for (double w : v) t += w * w;
    t = std::sqrt(t);
    if (t == 0) return point_value(x);
    std::vector<double> y(n_);
    for (std::size_t i = 0; i < n_; i++) y[i] = std::cos(t) * x[i] + std::sin(t) * v[i] / t;
    return normalised(std::move(y));
  }
  std::vector<double> log_at(const std::vector<double>& x, const std::vector<double>& y) const {
    double xy = 0;
    for (std::size_t i = 0; i < n_; i++) xy += x[i] * y[i];
    std::vector<double> w(n_);
    double r = 0;
    for (std::size_t i = 0; i < n_; i++) r += (w[i] = y[i] - xy * x[i]) * w[i];
    r = std::sqrt(r);
    const double theta = angle(x, y);
    std::vector<double> c(n_ - 1, 0);
    if (theta == 0) return c;
    const auto b = basis(x);
    if (!(r > 0)) {  // the antipode: any direction; the first basis vector, by convention
      c[0] = theta;
      return c;
    }
    for (std::size_t j = 0; j + 1 < n_; j++) {
      double d = 0;
      for (std::size_t i = 0; i < n_; i++) d += b[j][i] * w[i];
      c[j] = theta * d / r;
    }
    return c;
  }
  std::size_t n_;
};

class Rotations final : public Space {
 public:
  using Rotation = physis::geometry::Rotation;
  Rotations() = default;
  std::string describe() const override { return "the rotations SO(3)"; }

  static Value value_of(const Rotation& r) {
    double w = r.w(), x = r.x(), y = r.y(), z = r.z();
    const bool flip = w < 0 || (w == 0 && (x < 0 || (x == 0 && (y < 0 || (y == 0 && z < 0)))));
    if (flip) w = -w, x = -x, y = -y, z = -z;
    return point_value({w, x, y, z});
  }
  static Rotation rotation_of(const Value& v) { return Rotation::from_quaternion(v.reals[0], v.reals[1], v.reals[2], v.reals[3]); }

  Value sample(calx::random::Stream& s, double = 1) const override {
    for (;;) {
      const double w = s.normal(), x = s.normal(), y = s.normal(), z = s.normal();
      if (w * w + x * x + y * y + z * z > 1e-12) return value_of(Rotation::from_quaternion(w, x, y, z));
    }
  }
  Value mutate(const Value& x, double scale, calx::random::Stream& s) const override {
    detail::require(*this, x, "mutate");
    const double sigma = detail::clamp_scale(scale) * unit() / std::sqrt(3.0);
    const physis::geometry::Vec3 v{sigma * s.normal(), sigma * s.normal(), sigma * s.normal()};
    return value_of(rotation_of(x) * Rotation::exp(v));
  }
  double distance(const Value& x, const Value& y) const override {
    detail::require(*this, x, "distance");
    detail::require(*this, y, "distance");
    return (rotation_of(x).inverse() * rotation_of(y)).angle();  // physis's geodesic distance, the angle of x⁻¹y
  }
  double unit() const override { return std::numbers::pi / 2 + 2 / std::numbers::pi; }
  std::optional<std::string> invalid(const Value& x) const override {
    if (x.kind != Value::Kind::point) return std::string("a ") + name_of(x.kind) + " value, not a point";
    if (x.reals.size() != 4) return "a quaternion of " + std::to_string(x.reals.size()) + " parts, not 4";
    double r = 0;
    for (double v : x.reals) {
      if (!std::isfinite(v)) return std::string("a part not finite");
      r += v * v;
    }
    if (std::abs(r - 1) > 1e-9) return "a quaternion of norm " + std::to_string(std::sqrt(r)) + ", not 1";
    const double w = x.reals[0], a = x.reals[1], b = x.reals[2], c = x.reals[3];
    if (w < 0 || (w == 0 && (a < 0 || (a == 0 && (b < 0 || (b == 0 && c < 0)))))) return std::string("a quaternion whose sign is not the canonical one");
    return std::nullopt;
  }
  bool can_recombine() const override { return true; }
  Value recombine(const Value& x, const Value& y, calx::random::Stream& s) const override {
    detail::require(*this, x, "recombine");
    detail::require(*this, y, "recombine");
    if (x == y) return x;
    return value_of(slerp(rotation_of(x), rotation_of(y), s.uniform()));
  }
  Properties properties() const override { return {true, true, false, true, Unit::mean, true}; }
  bool can_repair() const override { return true; }
  Value repair(const Value& x) const override {
    if (valid(x)) return x;
    if (x.kind != Value::Kind::point || x.reals.size() != 4) throw Refusal("repair on " + describe() + ": a value of another shape");
    std::vector<double> q = x.reals;
    double r = 0;
    for (double& v : q) {
      if (!std::isfinite(v)) v = 0;
      r += v * v;
    }
    if (!(r > 0)) return value_of(Rotation::identity());
    return value_of(Rotation::from_quaternion(q[0], q[1], q[2], q[3]));
  }
  double change_floor() const override { return 1; }
  double diameter() const override { return std::numbers::pi; }
  bool has_chart() const override { return true; }
  std::size_t chart_dimension() const override { return 3; }
  std::vector<double> log(const Value& x, const Value& y) const override {
    detail::require(*this, x, "log");
    detail::require(*this, y, "log");
    const auto v = (rotation_of(x).inverse() * rotation_of(y)).log();
    return {v[0], v[1], v[2]};
  }
  Value retract(const Value& x, std::span<const double> v) const override {
    detail::require(*this, x, "retract");
    if (v.size() != 3) throw Refusal("retract on " + describe() + ": a tangent vector of another dimension");
    return value_of(rotation_of(x) * Rotation::exp({v[0], v[1], v[2]}));
  }
  double trust_radius(const Value&) const override { return std::numbers::pi; }
  std::optional<std::size_t> embedding_dimension() const override { return 4; }
  std::vector<double> embed(const Value& x) const override {
    detail::require(*this, x, "embed");
    return x.reals;
  }
  Value unembed(std::span<const double> z) const override {
    if (z.size() != 4) throw Refusal("unembed on " + describe() + ": coordinates of another dimension");
    return repair(point_value({z[0], z[1], z[2], z[3]}));
  }
};

}  // namespace heuresis::spaces
