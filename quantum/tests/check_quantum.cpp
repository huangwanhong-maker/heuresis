// Checks of heuresis quantum: the observation law of a product register equal to the product Bernoulli law (and an
// entangled register's not a product law); QIEA's observations equal to the product family's draws from the same
// stream, and QIEA solving OneMax; Grover's law against a simulation of the amplitudes (success probability, and
// uniformity within the marked set); the emulator's frequencies against the law; Dürr and Høyer's bound on the
// expected oracle calls and their √N scaling; Grover adaptive search finding the minima of a QUBO and an NK landscape
// against enumeration; the same search on another device through the seam; one emulator, read only, shared by
// searches on threads.

#include <heuresis/benchmarks/benchmarks.hpp>
#include <heuresis/quantum/quantum.hpp>
#include <heuresis/search/search.hpp>

#include <calx/verification/check.hpp>

#include <cmath>

using namespace heuresis::quantum;
using heuresis::search::Budget;
using heuresis::search::Draws;
using heuresis::search::Ending;
using heuresis::search::tag;

namespace {

// The amplitudes of r Grover iterations on N points with the given marked set, computed directly: the oracle
// negates the marked amplitudes, the diffusion reflects every amplitude about their mean.
std::vector<double> grover_amplitudes(const std::vector<char>& marked, std::uint64_t r) {
  const std::size_t N = marked.size();
  std::vector<double> a(N, 1 / std::sqrt(static_cast<double>(N)));
  for (std::uint64_t k = 0; k < r; k++) {
    double mean = 0;
    for (std::size_t i = 0; i < N; i++) mean += (a[i] = marked[i] ? -a[i] : a[i]);
    mean /= static_cast<double>(N);
    for (auto& v : a) v = 2 * mean - v;
  }
  return a;
}

// A device that simulates the amplitudes themselves (small N): another GroverDevice.
class AmplitudeDevice {
 public:
  explicit AmplitudeDevice(std::vector<double> values) : values_(std::move(values)) {}
  std::uint64_t size() const { return values_.size(); }
  Measurement measure(double threshold, std::uint64_t r, calx::random::Stream& s) {
    std::vector<char> marked(values_.size());
    for (std::size_t i = 0; i < values_.size(); i++) marked[i] = values_[i] < threshold;
    const auto a = grover_amplitudes(marked, r);
    double u = s.uniform(), c = 0;
    for (std::size_t i = 0; i < a.size(); i++)
      if (u < (c += a[i] * a[i])) return {i, r};
    return {a.size() - 1, r};
  }

 private:
  std::vector<double> values_;
};

}  // namespace

int main() {
  calx::verification::Suite s;
  const Draws draws(41);

  // The observation law of a product register is the product Bernoulli law.
  {
    const std::vector<double> phi{0.3, 1.1, 0.785, 1.5};
    const auto p = observation_probabilities(phi);
    const heuresis::distributions::ProductBernoulli f{};
    double dev = 0;
    for (std::size_t m = 0; m < 16; m++) {
      Bits x(4);
      double amplitude = 1;
      for (int i = 0; i < 4; i++) { x[i] = (m >> i) & 1; amplitude *= x[i] ? std::sin(phi[i]) : std::cos(phi[i]); }
      dev = std::max(dev, std::fabs(amplitude * amplitude - std::exp(f.log_density(p, x))));
    }
    s.check("a product register measured: |⟨x|ψ⟩|² = the product Bernoulli law, pᵢ = sin²φᵢ", dev, 1e-15);
    const double bell = 0.5 * 0.5 - 0 * 0;  // P(00)P(11) − P(01)P(10) for (|00⟩ + |11⟩)/√2
    const double product = p[0] * p[1] * (1 - p[0]) * (1 - p[1]) - p[0] * (1 - p[1]) * (1 - p[0]) * p[1];
    s.require("an entangled register's law is no product law (Bell: ¼; every product law: 0)", bell == 0.25 && std::fabs(product) < 1e-16);
  }

  // QIEA observes exactly the product family's draws, and solves OneMax.
  {
    QuantumInspiredOptions opt;
    opt.individuals = 6;
    QuantumInspired q(25, opt, Draws(3));
    auto onemax = [](const Bits& x) { return heuresis::benchmarks::onemax(x); };
    heuresis::search::run(q, onemax, Budget{6 * 40, INFINITY, -INFINITY});  // forty generations, so the angles have moved
    const auto angles = q.angles();
    const auto c = q.ask(heuresis::search::unlimited);
    const heuresis::distributions::ProductBernoulli f{};
    bool same = true;
    for (std::size_t k = 0; k < c.size(); k++) {
      calx::random::Stream r = Draws(3)(QuantumInspired::observe_tag, q.generation(), k);
      same = same && f.sample(observation_probabilities(angles[k]), r) == c[k].genotype;
    }
    bool moved = false;
    for (double a : angles[0]) moved = moved || std::fabs(a - std::numbers::pi / 4) > 1e-12;
    s.require("QIEA's observations are the product Bernoulli family's draws from the same stream", same && moved);

    QuantumInspired solve(40, QuantumInspiredOptions{}, Draws(4));
    const auto o = heuresis::search::run(solve, onemax, Budget{200000, INFINITY, 0});
    s.require("QIEA solves OneMax (n = 40)", o.ending == Ending::target);
  }

  // Grover's law against the amplitudes.
  {
    const std::size_t N = 64;
    std::vector<char> marked(N, 0);
    for (std::size_t i : {3, 17, 40, 41, 60}) marked[i] = 1;
    double dev = 0, spread = 0;
    for (std::uint64_t r = 0; r <= 12; r++) {
      const auto a = grover_amplitudes(marked, r);
      double pm = 0, lo = INFINITY, hi = -INFINITY, ulo = INFINITY, uhi = -INFINITY;
      for (std::size_t i = 0; i < N; i++) {
        if (marked[i]) { pm += a[i] * a[i]; lo = std::min(lo, a[i]); hi = std::max(hi, a[i]); }
        else { ulo = std::min(ulo, a[i]); uhi = std::max(uhi, a[i]); }
      }
      dev = std::max(dev, std::fabs(pm - grover_success(5, N, r)));
      spread = std::max({spread, hi - lo, uhi - ulo});
    }
    s.check("P(marked) after r iterations = sin²((2r + 1)θ), r = 0..12", dev, 1e-13);
    s.check("the amplitudes are uniform within the marked and within the unmarked sets", spread, 1e-14);

    std::vector<double> values(N);
    for (std::size_t i = 0; i < N; i++) values[i] = marked[i] ? 0.0 : 1.0;
    EmulatedGrover emu(N, [&](std::uint64_t i) { return values[i]; });
    double hits = 0;
    const int M = 200000;
    for (int t = 0; t < M; t++) { auto r = draws(tag("check.grover.emulate"), 0, t); hits += marked[emu.measure(0.5, 3, r).index]; }
    s.check("the emulator's frequency of a marked point = the law's, r = 3", hits / M - grover_success(5, N, 3), 0.003);
    s.require("the emulator counts its classical cost apart", emu.classical_evaluations() == N && emu.marked(0.5) == 5);
  }

  // Dürr and Høyer: the expected oracle calls before the minimum is held, against their bound and √N.
  {
    std::vector<double> mean_calls;
    for (std::uint64_t N : {256u, 1024u, 4096u}) {
      double calls = 0;
      const int runs = 300;
      for (int run = 0; run < runs; run++) {
        calx::random::Stream v = draws(tag("check.dh.values"), N, run);
        EmulatedGrover device(N, [&](std::uint64_t) { return v.uniform(); });
        GroverAdaptiveSearch gas(device, GroverSearchOptions{}, Draws(500 + run));
        const double least = device.minimum();
        calx::random::Stream w = draws(tag("check.dh.values"), N, run);
        std::vector<double> vals(N);
        for (auto& x : vals) x = w.uniform();
        heuresis::search::run(gas, [&](std::uint64_t i) { return vals[i]; }, Budget{INFINITY, INFINITY, least});
        calls += static_cast<double>(gas.oracle_calls());
      }
      calls /= runs;
      mean_calls.push_back(calls);
      s.require("Dürr–Høyer: mean oracle calls ≤ (45/4)√N + (7/10)lg²N, N = " + std::to_string(N) + " (" + std::to_string(calls) + " ≤ " + std::to_string(dh_expected_bound(N)) + ")", calls <= dh_expected_bound(N));
    }
    // The exponent of growth from N = 256 to 4096: ½ asymptotically (the bound), 1 for any classical search.
    const double exponent = std::log(mean_calls[2] / mean_calls[0]) / std::log(16.0);
    s.require("the oracle calls grow sublinearly, as N^" + std::to_string(exponent) + " from N = 256 to 4096 (below N^0.75; ½ asymptotically, 1 classically)", exponent < 0.75);
  }

  // Grover adaptive search finds the minima of a QUBO and an NK landscape, against enumeration.
  {
    const std::size_t n = 10, N = std::size_t{1} << n;
    auto bits = [&](std::uint64_t m) { Bits x(n); for (std::size_t i = 0; i < n; i++) x[i] = (m >> i) & 1; return x; };
    const auto qubo = heuresis::benchmarks::Qubo::random(n, 7);
    const heuresis::benchmarks::NK nk(n, 3, 8);
    for (int which = 0; which < 2; which++) {
      auto f = [&](std::uint64_t m) { return which == 0 ? qubo(bits(m)) : nk(bits(m)); };
      double least = INFINITY;
      for (std::uint64_t m = 0; m < N; m++) least = std::min(least, f(m));
      EmulatedGrover device(N, f);
      GroverAdaptiveSearch gas(device, GroverSearchOptions{6.0 / 5.0, 50 * dh_cutoff(N)}, Draws(9 + which));
      const auto o = heuresis::search::run(gas, f, Budget{INFINITY, INFINITY, least});
      s.require(std::string(which == 0 ? "QUBO" : "NK") + " (n = 10): Grover adaptive search reaches the minimum found by enumeration (" + std::to_string(gas.oracle_calls()) + " oracle calls, " + std::to_string(gas.measurements()) + " measurements; N = 1024)", o.ending == Ending::target && o.best->evaluation.value() == least);
    }
  }

  // One emulator shared by four searches on a pool of threads: measure only reads it, and every run is the run it
  // makes on an emulator of its own.
  {
    static_assert(GroverDevice<const EmulatedGrover>);
    const std::uint64_t N = 256;
    std::vector<double> values(N);
    calx::random::Stream v = draws(tag("check.shared.values"), 0, 0);
    for (auto& x : values) x = v.uniform();
    auto f = [&](std::uint64_t i) { return values[i]; };
    const EmulatedGrover shared(N, f);
    std::vector<std::uint64_t> calls(4), alone(4);
    calx::scheduler::ThreadPool pool(4);
    calx::scheduler::parallel_for(pool, 4, 1, [&](std::size_t b, std::size_t e, std::size_t) {
      for (std::size_t k = b; k < e; k++) {
        GroverAdaptiveSearch gas(shared, GroverSearchOptions{6.0 / 5.0, dh_cutoff(N)}, Draws(20 + k));
        heuresis::search::run(gas, f, Budget{});
        calls[k] = gas.oracle_calls();
      }
    });
    for (std::size_t k = 0; k < 4; k++) {
      EmulatedGrover own(N, f);
      GroverAdaptiveSearch gas(own, GroverSearchOptions{6.0 / 5.0, dh_cutoff(N)}, Draws(20 + k));
      heuresis::search::run(gas, f, Budget{});
      alone[k] = gas.oracle_calls();
    }
    s.require("one const emulator serves four searches on four threads, each as on an emulator of its own", calls == alone);
  }

  // The same search on another device: one that computes the amplitudes.
  {
    std::vector<double> values(128);
    calx::random::Stream v = draws(tag("check.device.values"), 0, 0);
    for (auto& x : values) x = v.uniform();
    const double least = *std::min_element(values.begin(), values.end());
    AmplitudeDevice device(values);
    GroverAdaptiveSearch gas(device, GroverSearchOptions{}, Draws(12));
    const auto o = heuresis::search::run(gas, [&](std::uint64_t i) { return values[i]; }, Budget{INFINITY, INFINITY, least});
    s.require("Grover adaptive search runs unchanged on a device that simulates amplitudes, and finds the minimum", o.ending == Ending::target);
  }

  return s.done();
}
