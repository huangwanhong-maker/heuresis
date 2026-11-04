// Checks of heuresis landscapes: OneMax elementary with λ = 2 and ρ(1) = 1 − 2/n exactly, and a long walk's
// autocorrelation against (1 − 2/n)ˢ; an Ising energy without fields elementary with λ = 4, with fields
// following ρ(1) = 1 − (2Σh² + 4ΣJ²)/(n(Σh² + ΣJ²)) exactly; NK at k = 0 elementary, at k = 3 near
// 1 − (k + 1)/n; OneMax's fitness–distance correlation 1, Trap's negative; Trap's 2ᵇ local optima and their
// basins counted in closed form; LeadingOnes' neutrality.

#include <heuresis/benchmarks/benchmarks.hpp>
#include <heuresis/landscapes/landscapes.hpp>

#include <calx/verification/check.hpp>

#include <cmath>

using namespace heuresis::landscapes;
using heuresis::search::Draws;
using heuresis::search::tag;

int main() {
  calx::verification::Suite s;

  // OneMax.
  {
    const std::size_t n = 12;
    const auto h = Hypercube::tabulate(n, heuresis::benchmarks::onemax);
    const auto e = elementary(h);
    s.check("OneMax is elementary: λ = 2", e.lambda - 2, 1e-12);
    s.check("OneMax: the wave equation's residual", e.residual, 1e-12);
    s.check("OneMax: ρ(1) = 1 − 2/n exactly", one_step_correlation(h) - (1 - 2.0 / n), 1e-14);
    s.require("OneMax has one local optimum", local_minima(h).size() == 1);

    const std::size_t m = 20;
    calx::random::Stream r = Draws(1)(tag("check.landscapes.start"), 0, 0);
    const auto walk = random_walk(heuresis::variation::random_bits(m, r), heuresis::benchmarks::onemax, [](const Bits& x, calx::random::Stream& st) {
      Bits y = x;
      heuresis::variation::flip_exactly(y, 1, st);
      return y;
    }, 400000, Draws(2));
    const auto rho = autocorrelation(walk, 5);
    double dev = 0;
    for (std::size_t k = 1; k <= 5; k++) dev = std::max(dev, std::fabs(rho[k] - std::pow(1 - 2.0 / m, static_cast<double>(k))));
    s.check("OneMax: a walk's ρ(s) against (1 − 2/n)ˢ, s ≤ 5", dev, 0.02);
    s.check("OneMax: correlation length −1/ln(1 − 2/n)", correlation_length(rho[1]) / correlation_length(1 - 2.0 / m) - 1, 0.1);

    std::vector<Bits> all;
    for (std::size_t k = 0; k < (std::size_t{1} << 10); k++) all.push_back(Hypercube::point(k, 10));
    const double r1 = fitness_distance_correlation(std::span<const Bits>(all), heuresis::benchmarks::onemax, [](const Bits& x) { return heuresis::variation::hamming(x, Bits(x.size(), 1)); });
    s.check("OneMax: fitness–distance correlation 1", r1 - 1, 1e-12);
  }

  // Ising energies: without fields elementary with λ = 4; with fields, the exact mixture.
  {
    const std::size_t n = 10;
    calx::random::Stream r = Draws(3)(tag("check.landscapes.ising"), 0, 0);
    std::vector<double> h(n), J(n * n, 0.0);
    double H2 = 0, J2 = 0;
    for (auto& v : h) { v = r.normal(); H2 += v * v; }
    for (std::size_t i = 0; i < n; i++)
      for (std::size_t j = i + 1; j < n; j++) { J[i * n + j] = r.normal(); J2 += J[i * n + j] * J[i * n + j]; }
    auto energy = [&](const Bits& x, bool fields) {
      double E = 0;
      for (std::size_t i = 0; i < n; i++) {
        const double si = 2.0 * x[i] - 1;
        if (fields) E += h[i] * si;
        for (std::size_t j = i + 1; j < n; j++) E += J[i * n + j] * si * (2.0 * x[j] - 1);
      }
      return E;
    };
    const auto pure = Hypercube::tabulate(n, [&](const Bits& x) { return energy(x, false); });
    const auto mixed = Hypercube::tabulate(n, [&](const Bits& x) { return energy(x, true); });
    const auto e = elementary(pure);
    s.check("an Ising energy without fields is elementary: λ = 4", e.lambda - 4, 1e-10);
    s.check("its wave equation's residual", e.residual, 1e-10);
    s.check("with fields: ρ(1) = 1 − (2Σh² + 4ΣJ²)/(n(Σh² + ΣJ²))", one_step_correlation(mixed) - (1 - (2 * H2 + 4 * J2) / (n * (H2 + J2))), 1e-12);
    const auto q = Hypercube::tabulate(n, heuresis::benchmarks::Qubo::random(n, 4));
    s.require("a random QUBO (orders 1 and 2 mixed) is not elementary", elementary(q).residual > 0.05 * std::sqrt(q.variance()));
  }

  // NK landscapes.
  {
    const std::size_t n = 12;
    const auto k0 = Hypercube::tabulate(n, heuresis::benchmarks::NK(n, 0, 5));
    s.check("NK at k = 0 is elementary with λ = 2", std::fabs(elementary(k0).lambda - 2) + elementary(k0).residual, 1e-12);
    double mean = 0;
    const int instances = 30;
    for (int t = 0; t < instances; t++) mean += one_step_correlation(Hypercube::tabulate(n, heuresis::benchmarks::NK(n, 3, 6, t)));
    mean /= instances;
    s.check("NK at k = 3: the mean ρ(1) against 1 − (k + 1)/n (Weinberger 1991)", mean - (1 - 4.0 / n), 0.05);
  }

  // Trap: 2^b local optima; their basins (C(k,k−1) + 1)^a (Σ_{u ≤ k−2} C(k,u))^{b−a}; a negative FDC.
  {
    const auto h = Hypercube::tabulate(8, [](const Bits& x) { return heuresis::benchmarks::trap(x, 4); });
    const auto b = basins(h);
    std::vector<std::size_t> sizes(4, 0);
    for (std::size_t i = 0; i < b.optima.size(); i++) {
      const auto x = Hypercube::point(b.optima[i], 8);
      const std::size_t ones_a = x[0] + x[1] + x[2] + x[3], ones_b = x[4] + x[5] + x[6] + x[7];
      sizes[(ones_a == 4) + 2 * (ones_b == 4)] = b.sizes[i];
    }
    s.require("Trap (two blocks of four): exactly four local optima", local_minima(h).size() == 4 && b.optima.size() == 4);
    s.require("Trap: the basins hold 121, 55, 55 and 25 points", sizes == std::vector<std::size_t>{121, 55, 55, 25});
    std::vector<double> v(h.values().begin(), h.values().end()), d;
    for (std::size_t m = 0; m < h.size(); m++) d.push_back(static_cast<double>(heuresis::variation::hamming(Hypercube::point(m, 8), Bits(8, 1))));
    s.require("Trap: fitness–distance correlation negative (deceptive)", fitness_distance_correlation(v, d) < 0);
  }

  // LeadingOnes' neutrality.
  {
    const std::size_t n = 10;
    const auto h = Hypercube::tabulate(n, heuresis::benchmarks::leading_ones);
    Bits x(n, 0);
    x[0] = x[1] = x[2] = 1;  // three leading ones
    s.check("LeadingOnes: a point with three leading ones has n − 4 neutral neighbours", neutral_degree(h, Hypercube::index(x)) - 6.0 / n, 1e-15);
    s.require("LeadingOnes: one neutral network for each of its n + 1 values", neutral_networks(h) == n + 1);
  }

  return s.done();
}
