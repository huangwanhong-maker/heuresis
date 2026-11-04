// Checks of heuresis genetic: the (1+1) EA's mean runtime on OneMax against its exact expectation from the
// Markov chain on the number of ones, and on LeadingOnes against Böttcher, Doerr and Neumann's closed form;
// the (1+(λ,λ)) GA's linear time on OneMax against the (1+1) EA's n log n; the genetic algorithm's states
// independent of the order of telling, the executor and the workers; the steady-state GA reproducible on
// workers; migration's semantics.

#include <heuresis/benchmarks/benchmarks.hpp>
#include <heuresis/genetic/genetic.hpp>

#include <calx/verification/check.hpp>

#include <algorithm>
#include <cmath>

using namespace heuresis::genetic;
using namespace heuresis::search;
using heuresis::benchmarks::leading_ones;
using heuresis::benchmarks::onemax;

namespace {

double binomial_pmf(int n, int k, double p) {
  if (k < 0 || k > n) return 0;
  return std::exp(std::lgamma(n + 1.0) - std::lgamma(k + 1.0) - std::lgamma(n - k + 1.0) + k * std::log(p) + (n - k) * std::log1p(-p));
}

// The exact expected number of evaluations of the (1+1) EA at rate p to reach OneMax's optimum from a uniform
// string: the chain on the number of ones i, where a mutation turns a of the i ones and b of the n − i zeros.
double onemax_expected(int n, double p) {
  std::vector<double> E(n + 1, 0.0);
  for (int i = n - 1; i >= 0; i--) {
    double up = 0, acc = 0;
    for (int a = 0; a <= i; a++)
      for (int b = 0; b <= n - i; b++) {
        const int j = i - a + b;
        if (j <= i) continue;
        const double pr = binomial_pmf(i, a, p) * binomial_pmf(n - i, b, p);
        up += pr;
        acc += pr * E[j];
      }
    E[i] = (1 + acc) / up;
  }
  double total = 1;
  for (int i = 0; i <= n; i++) total += binomial_pmf(n, i, 0.5) * E[i];
  return total;
}

struct Sample {
  double mean = 0, se = 0;
};
Sample summarise(const std::vector<double>& t) {
  Sample s;
  for (double v : t) s.mean += v;
  s.mean /= static_cast<double>(t.size());
  double var = 0;
  for (double v : t) var += (v - s.mean) * (v - s.mean);
  s.se = std::sqrt(var / static_cast<double>(t.size() - 1) / static_cast<double>(t.size()));
  return s;
}

template <class A, class F>
double runtime(A& a, F&& f) {
  const auto o = run(a, f, Budget{1e8, INFINITY, 0});
  return o.ending == Ending::target ? o.evaluations : INFINITY;
}

}  // namespace

int main() {
  calx::verification::Suite s;

  // The (1+1) EA on OneMax: the sample mean of 600 runs against the exact expectation.
  {
    const int n = 30, R = 600;
    const double exact = onemax_expected(n, 1.0 / n);
    std::vector<double> t;
    for (int r = 0; r < R; r++) {
      auto ea = one_plus_one_ea(n, Draws(1000 + r));
      t.push_back(runtime(ea, onemax));
    }
    const auto m = summarise(t);
    s.check("(1+1) EA on OneMax (n = 30): mean runtime against the chain's exact expectation, in standard errors", (m.mean - exact) / m.se, 4.0);
    s.check("the exact expectation against e n ln n − 1.8925 n + ½e ln n (Hwang et al. 2018), relative", (exact - (std::exp(1.0) * n * std::log(n) - 1.89254 * n + 0.5 * std::exp(1.0) * std::log(n))) / exact, 0.03);
  }

  // The (1+1) EA on LeadingOnes against ((1 − p)^{1−n} − (1 − p))/(2p²), plus the initial evaluation.
  {
    const int n = 25, R = 500;
    const double p = 1.0 / n, exact = 1 + (std::pow(1 - p, 1 - n) - (1 - p)) / (2 * p * p);
    std::vector<double> t;
    for (int r = 0; r < R; r++) {
      auto ea = one_plus_one_ea(n, Draws(5000 + r));
      t.push_back(runtime(ea, leading_ones));
    }
    const auto m = summarise(t);
    s.check("(1+1) EA on LeadingOnes (n = 25): mean runtime against Böttcher, Doerr and Neumann, in standard errors", (m.mean - exact) / m.se, 4.0);
  }

  // Randomised local search on OneMax: the coupon collector's n H_{n/2}-like time, exactly from its chain:
  // from z zeros, a flip of a zero has probability z/n, so E = 1 + Σ_z P(z) Σ_{k ≤ z} n/k.
  {
    const int n = 40, R = 600;
    double exact = 1;
    for (int z = 1; z <= n; z++) {
      double h = 0;
      for (int k = 1; k <= z; k++) h += static_cast<double>(n) / k;
      exact += binomial_pmf(n, z, 0.5) * h;
    }
    std::vector<double> t;
    for (int r = 0; r < R; r++) {
      auto rls = randomised_local_search(n, Draws(9000 + r));
      t.push_back(runtime(rls, onemax));
    }
    const auto m = summarise(t);
    s.check("randomised local search on OneMax (n = 40): mean runtime against the coupon collector's, in standard errors", (m.mean - exact) / m.se, 4.0);
  }

  // The self-adjusting (1+(λ,λ)) GA: linear on OneMax, faster than the (1+1) EA at n = 800.
  {
    auto mean_runtime = [&](int n, int R, bool ga) {
      double total = 0;
      for (int r = 0; r < R; r++) {
        if (ga) {
          OnePlusLambdaLambda a(n, {}, Draws(20000 + r));
          total += runtime(a, onemax);
        } else {
          auto a = one_plus_one_ea(n, Draws(30000 + r));
          total += runtime(a, onemax);
        }
      }
      return total / R;
    };
    const double g100 = mean_runtime(100, 20, true), g800 = mean_runtime(800, 10, true), e100 = mean_runtime(100, 20, false), e800 = mean_runtime(800, 10, false);
    s.check("(1+(λ,λ)) GA: runtime/n at n = 800 over n = 100, near 1 (linear)", (g800 / 800) / (g100 / 100) - 1, 0.25);
    s.check("(1+1) EA: runtime/n at n = 800 over n = 100, near ln 800/ln 100 (n log n)", (e800 / 800) / (e100 / 100) - std::log(800.0) / std::log(100.0), 0.25);
    s.require("(1+(λ,λ)) GA faster than the (1+1) EA at n = 800", g800 < e800);
  }

  // The genetic algorithm: the same states whatever the order of telling, the executor or the workers.
  {
    const std::size_t n = 40;
    Operators<Bits> op{random_bits(n), bit_flip(), uniform_crossover<std::uint8_t>()};
    GaOptions o;
    o.mu = 20;
    auto tickets = [](const GeneticAlgorithm<Bits>& g) {
      std::vector<Ticket> t;
      for (const auto& x : g.population()) t.push_back(x.ticket);
      return t;
    };
    GeneticAlgorithm<Bits> serial(op, o, Draws(77));
    const auto o1 = run(serial, onemax, Budget{2000, INFINITY, -INFINITY});

    GeneticAlgorithm<Bits> shuffled(op, o, Draws(77));
    for (int g = 0; g < 100; g++) {
      auto c = shuffled.ask(unlimited);
      std::vector<Told> t;
      for (auto& x : c) t.push_back({x.ticket, scalar(onemax(x.genotype))});
      std::reverse(t.begin(), t.end());
      std::rotate(t.begin(), t.begin() + g % t.size(), t.end());
      shuffled.tell(t);
    }
    calx::scheduler::ThreadPool pool(4);
    GeneticAlgorithm<Bits> pooled(op, o, Draws(77));
    run(pooled, onemax, Budget{2000, INFINITY, -INFINITY}, Pool(pool));
    GeneticAlgorithm<Bits> parts(op, o, Draws(77));
    run(parts, onemax, Budget{2000, INFINITY, -INFINITY}, Serial{}, 7);
    GeneticAlgorithm<Bits> async(op, o, Draws(77));
    Workers w;
    w.count = 5;
    w.duration = [](const Evaluation&, std::size_t k, calx::random::Stream& r) { return 0.5 + k + r.exponential(); };
    run_asynchronous(async, onemax, Budget{2000, INFINITY, -INFINITY}, w);
    s.require("GA: telling in any order gives the same population", tickets(shuffled) == tickets(serial));
    s.require("GA: a pool of threads gives the same population", tickets(pooled) == tickets(serial));
    s.require("GA: generations cut into parts give the same population", tickets(parts) == tickets(serial));
    s.require("GA: simulated workers give the same population", tickets(async) == tickets(serial));
    s.require("GA: the best of the run is in the final population (elitism)", o1.best && std::any_of(serial.population().begin(), serial.population().end(), [&](const auto& x) { return x.evaluation.value() == o1.best->evaluation.value(); }));
    s.require("GA on OneMax (n = 40) reaches the optimum within 2000 evaluations", o1.best->evaluation.value() == 0);

    // Migration: an injected optimum displaces the worst and is sent first.
    GeneticAlgorithm<Bits> m(op, o, Draws(78));
    run(m, onemax, Budget{40, INFINITY, -INFINITY});
    const Individual<Bits> star{make_ticket(9, 0), Bits(n, 1), scalar(0), {}};
    m.inject(std::span<const Individual<Bits>>(&star, 1));
    s.require("GA: an immigrant better than the worst enters, and is the first emigrant", m.emigrants(1).size() == 1 && m.emigrants(1)[0].ticket == star.ticket && m.population().size() == o.mu);
  }

  // A real-coded GA (SBX and polynomial mutation) on the sphere in ten variables.
  {
    const auto box = heuresis::variation::Box::cube(10, -5, 5);
    Operators<Real> op{uniform_point(box), polynomial(box, 20), simulated_binary(box, 15)};
    GaOptions o;
    o.mu = 60;
    o.elites = 2;
    GeneticAlgorithm<Real> ga(op, o, Draws(3));
    const auto out = run(ga, heuresis::benchmarks::sphere, Budget{30000, INFINITY, -INFINITY});
    s.check("real-coded GA on the sphere (n = 10, 30000 evaluations): best value", out.best->evaluation.value(), 1e-2);
  }

  // The steady-state GA on workers: reproducible, every worker busy, OneMax solved.
  {
    const std::size_t n = 30;
    Operators<Bits> op{random_bits(n), bit_flip(), uniform_crossover<std::uint8_t>()};
    GaOptions o;
    o.mu = 20;
    Workers w;
    w.count = 4;
    w.duration = [](const Evaluation&, std::size_t, calx::random::Stream& r) { return r.exponential(); };
    SteadyStateGA<Bits> a(op, o, Draws(4)), b(op, o, Draws(4));
    const auto oa = run_asynchronous(a, onemax, Budget{20000, INFINITY, 0}, w), ob = run_asynchronous(b, onemax, Budget{20000, INFINITY, 0}, w);
    s.require("steady-state GA on workers: two runs are the same", oa.evaluations == ob.evaluations && oa.best->ticket == ob.best->ticket && oa.time == ob.time);
    s.require("steady-state GA on workers solves OneMax (n = 30)", oa.ending == Ending::target);
    // Under the serial driver with a finite budget, it breeds at most `breadth` from one state of its population.
    SteadyStateGA<Bits> c(op, o, Draws(4));
    const auto first = c.ask(1000);
    std::vector<Told> t;
    for (auto& x : first) t.push_back({x.ticket, scalar(onemax(x.genotype))});
    c.tell(t);
    s.require("steady-state GA: the initial μ together, then one offspring per ask", first.size() == o.mu && c.ask(1000).size() == 1);
    SteadyStateGA<Bits> d(op, o, Draws(4));
    s.require("steady-state GA under the serial driver solves OneMax (n = 30)", run(d, onemax, Budget{20000, INFINITY, 0}).ending == Ending::target);
  }

  // The (μ+1) EA: elitist, so its best never worsens; OneMax solved.
  {
    MuPlusOne<Bits> mu(5, random_bits(30), bit_flip(), Draws(6));
    const auto out = run(mu, onemax, Budget{20000, INFINITY, 0});
    bool monotone = true;
    for (std::size_t i = 1; i < out.progress.size(); i++) monotone = monotone && out.progress[i].value < out.progress[i - 1].value;
    s.require("(μ+1) EA solves OneMax (n = 30)", out.ending == Ending::target && monotone);
    MuPlusOne<Bits> ga(5, random_bits(30), bit_flip(), Draws(6), uniform_crossover<std::uint8_t>(), 0.5);
    s.require("(μ+1) GA solves OneMax (n = 30)", run(ga, onemax, Budget{20000, INFINITY, 0}).ending == Ending::target);
  }

  return s.done();
}
