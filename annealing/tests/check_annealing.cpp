// Checks of heuresis annealing: the schedules' closed forms; the Metropolis chain at a fixed temperature sampling
// the exact Boltzmann law of a QUBO, by enumeration; the ask–tell chain equal to the Metropolis chain driven by hand
// with the same streams; parallel tempering's replicas each sampling their own Boltzmann law; population annealing's
// ln Z against the exact partition function; simulated annealing reaching the enumerated ground state; the modified
// Lam schedule holding its target acceptance; reproducibility and independence of the executor.

#include <heuresis/annealing/annealing.hpp>
#include <heuresis/benchmarks/pseudo_boolean.hpp>
#include <heuresis/search/run.hpp>

#include <calx/verification/check.hpp>

#include <cmath>

using namespace heuresis::annealing;
using heuresis::benchmarks::Qubo;
using heuresis::search::Draws;
using heuresis::search::tag;
using heuresis::variation::Bits;

namespace {

std::size_t index_of(const Bits& x) {
  std::size_t k = 0;
  for (std::size_t i = 0; i < x.size(); i++) k |= static_cast<std::size_t>(x[i]) << i;
  return k;
}

Bits bits_of(std::size_t k, std::size_t n) {
  Bits x(n);
  for (std::size_t i = 0; i < n; i++) x[i] = (k >> i) & 1;
  return x;
}

std::vector<double> energies(const Qubo& q) {
  std::vector<double> e(std::size_t{1} << q.n());
  for (std::size_t k = 0; k < e.size(); k++) e[k] = q(bits_of(k, q.n()));
  return e;
}

std::vector<double> boltzmann(const std::vector<double>& e, double t) {
  const double lo = *std::min_element(e.begin(), e.end());
  std::vector<double> p(e.size());
  double z = 0;
  for (std::size_t k = 0; k < e.size(); k++) z += (p[k] = std::exp(-(e[k] - lo) / t));
  for (auto& v : p) v /= z;
  return p;
}

double total_variation(const std::vector<double>& counts, const std::vector<double>& p) {
  double n = 0, tv = 0;
  for (double c : counts) n += c;
  for (std::size_t k = 0; k < p.size(); k++) tv += std::fabs(counts[k] / n - p[k]);
  return tv / 2;
}

Bits flip_one(const Bits& x, calx::random::Stream& s) {
  Bits y = x;
  heuresis::variation::flip_exactly(y, 1, s);
  return y;
}

}  // namespace

int main() {
  calx::verification::Suite s;

  // Schedules.
  {
    Schedule g = Schedule::geometric(4, 0.5, 3);
    for (int k = 0; k < 7; k++) g.advance(true);
    s.check("geometric: T₇ = T₀ α^{⌊7/3⌋}", g.temperature() - 1, 1e-15);
    Schedule l = Schedule::logarithmic(2.5);
    for (int k = 0; k < 10; k++) l.advance(false);
    s.check("logarithmic: T₁₀ = c / ln 12", l.temperature() - 2.5 / std::log(12.0), 1e-15);
    s.check("Lam's target starts at 1", Schedule::lam_target(0) - 1, 1e-15);
    s.check("Lam's target is 0.44 from 15% to 65%", Schedule::lam_target(0.4) - 0.44, 1e-15);
    s.check("Lam's target ends at 0.44/440", Schedule::lam_target(1) - 0.001, 1e-15);
    s.require("Metropolis: downhill always, infinite energy never from finite, always from infinite",
              metropolis(1, 0.5, 1, 0.999) && !metropolis(1, INFINITY, 1, 0.0) && metropolis(INFINITY, 7, 1, 0.999) && metropolis(0, 1, 1, std::exp(-1.0) - 1e-12) && !metropolis(0, 1, 1, std::exp(-1.0) + 1e-12));
  }

  // The Metropolis chain at a fixed temperature samples the Boltzmann law exactly.
  {
    const Qubo q = Qubo::random(8, 17);
    const auto e = energies(q);
    const double t = 2.0;
    const auto p = boltzmann(e, t);
    SimulatedAnnealing<Bits> sa([](calx::random::Stream& r) { return heuresis::variation::random_bits(8, r); }, flip_one, Schedule::constant(t), Draws(3));
    std::vector<double> counts(e.size(), 0);
    for (int k = 0; k < 1000000; k++) {
      auto c = sa.ask(1);
      const heuresis::search::Told told{c[0].ticket, heuresis::search::scalar(q(c[0].genotype))};
      sa.tell(std::span<const heuresis::search::Told>(&told, 1));
      counts[index_of(sa.state()->genotype)] += 1;
    }
    s.check("Metropolis at T = 2 on a QUBO of 8 bits: total variation from the exact Boltzmann law", total_variation(counts, p), 0.01);

    // The same chain driven by hand with the same streams.
    SimulatedAnnealing<Bits> sb([](calx::random::Stream& r) { return heuresis::variation::random_bits(8, r); }, flip_one, Schedule::constant(t), Draws(5));
    const Draws d(5);
    auto r0 = d(SimulatedAnnealing<Bits>::initialise, 0, 0);
    Bits x = heuresis::variation::random_bits(8, r0);
    double ex = q(x);
    bool same = true;
    for (std::uint64_t k = 0; k < 2001; k++) {
      auto c = sb.ask(1);
      const heuresis::search::Told told{c[0].ticket, heuresis::search::scalar(q(c[0].genotype))};
      sb.tell(std::span<const heuresis::search::Told>(&told, 1));
      if (k == 0) { same = same && sb.state()->genotype == x; continue; }
      auto rp = d(SimulatedAnnealing<Bits>::propose, k - 1, 0);
      const Bits y = flip_one(x, rp);
      const double ey = q(y);
      if (metropolis(ex, ey, t, d(SimulatedAnnealing<Bits>::accept, k - 1, 0).uniform())) { x = y; ex = ey; }
      same = same && sb.state()->genotype == x;
    }
    s.require("the ask–tell chain is the Metropolis chain driven by hand with the same streams", same);
  }

  // Parallel tempering: each temperature samples its own Boltzmann law.
  {
    const Qubo q = Qubo::random(6, 23);
    const auto e = energies(q);
    const std::vector<double> temps{0.6, 1.1, 2.0, 3.6};
    ParallelTempering<Bits> pt([](calx::random::Stream& r) { return heuresis::variation::random_bits(6, r); }, flip_one, temps, Draws(8));
    std::vector<std::vector<double>> counts(temps.size(), std::vector<double>(e.size(), 0));
    for (int g = 0; g < 250000; g++) {
      auto c = pt.ask(heuresis::search::unlimited);
      std::vector<heuresis::search::Told> told;
      for (auto& x : c) told.push_back({x.ticket, heuresis::search::scalar(q(x.genotype))});
      pt.tell(told);
      for (std::size_t r = 0; r < temps.size(); r++) counts[r][index_of(pt.states()[r].genotype)] += 1;
    }
    double worst = 0;
    for (std::size_t r = 0; r < temps.size(); r++) worst = std::max(worst, total_variation(counts[r], boltzmann(e, temps[r])));
    s.check("parallel tempering: every replica's total variation from its Boltzmann law", worst, 0.01);
    s.require("exchanges are accepted between every pair of neighbouring temperatures", pt.exchange_rate(0) > 0.1 && pt.exchange_rate(1) > 0.1 && pt.exchange_rate(2) > 0.1);

    calx::scheduler::ThreadPool pool(4);
    ParallelTempering<Bits> a([](calx::random::Stream& r) { return heuresis::variation::random_bits(6, r); }, flip_one, temps, Draws(2));
    ParallelTempering<Bits> b([](calx::random::Stream& r) { return heuresis::variation::random_bits(6, r); }, flip_one, temps, Draws(2));
    const auto oa = heuresis::search::run(a, q, heuresis::search::Budget{4000, INFINITY, -INFINITY});
    const auto ob = heuresis::search::run(b, q, heuresis::search::Budget{4000, INFINITY, -INFINITY}, heuresis::search::Pool(pool), 3);
    bool same = true;
    for (std::size_t r = 0; r < temps.size(); r++) same = same && a.states()[r].ticket == b.states()[r].ticket;
    s.require("parallel tempering on a pool of threads, generations cut into parts, gives the same run", same && oa.best->ticket == ob.best->ticket);
  }

  // Population annealing: ln Z(β = 1.5) against the exact partition function of a 10-bit QUBO.
  {
    const Qubo q = Qubo::random(10, 41);
    const auto e = energies(q);
    const double beta = 1.5;
    double lo = *std::min_element(e.begin(), e.end()), z = 0, mean = 0;
    for (double v : e) z += std::exp(-beta * (v - lo));
    for (double v : e) mean += v * std::exp(-beta * (v - lo)) / z;
    const double log_z = -beta * lo + std::log(z);
    PopulationAnnealing<Bits>::Options o;
    o.population = 2000;
    o.sweeps = 10;
    for (int k = 0; k <= 30; k++) o.betas.push_back(beta * k / 30);
    // Four independent runs: their mean error within three standard errors of zero; each, a gross bound, within 0.1.
    std::vector<double> err;
    double mean_err = 0, energy_err = 0;
    bool ended = true;
    for (std::uint64_t seed = 6; seed < 10; seed++) {
      PopulationAnnealing<Bits> pa([](calx::random::Stream& r) { return heuresis::variation::random_bits(10, r); }, flip_one, o, Draws(seed));
      const auto out = heuresis::search::run(pa, q);
      ended = ended && out.ending == heuresis::search::Ending::stopped && pa.beta() == beta;
      err.push_back(pa.log_partition(10 * std::log(2.0)) - log_z);
      mean_err += err.back() / 4;
      energy_err = std::max(energy_err, std::fabs(pa.mean_energy() - mean));
    }
    double var = 0, worst = 0;
    for (double v : err) { var += (v - mean_err) * (v - mean_err) / 3; worst = std::max(worst, std::fabs(v)); }
    s.require("population annealing ends at the last temperature", ended);
    s.check("population annealing: the mean error of ln Z(1.5) of a 10-bit QUBO in standard errors (4 runs)", mean_err / std::sqrt(var / 4), 3);
    s.check("each run's ln Z against enumeration", worst, 0.1);
    s.check("each final population's mean energy against the exact Boltzmann mean", energy_err, 0.05);
  }

  // Simulated annealing reaches the ground state of a 16-bit QUBO, found by enumeration of its 65536 states.
  {
    const Qubo q = Qubo::random(16, 3);
    const auto e = energies(q);
    const double ground = *std::min_element(e.begin(), e.end());
    const std::size_t steps = 60000;
    SimulatedAnnealing<Bits> sa([](calx::random::Stream& r) { return heuresis::variation::random_bits(16, r); }, flip_one, Schedule::geometric(3, std::pow(0.01 / 3, 1.0 / steps)), Draws(11));
    const auto out = heuresis::search::run(sa, q, heuresis::search::Budget{static_cast<double>(steps), INFINITY, -INFINITY});
    s.check("geometric annealing finds the ground state", out.best->evaluation.value() - ground, 1e-12);
    SimulatedAnnealing<Bits> again([](calx::random::Stream& r) { return heuresis::variation::random_bits(16, r); }, flip_one, Schedule::geometric(3, std::pow(0.01 / 3, 1.0 / steps)), Draws(11));
    const auto out2 = heuresis::search::run(again, q, heuresis::search::Budget{static_cast<double>(steps), INFINITY, -INFINITY});
    s.require("the same seed gives the same run", out2.best->ticket == out.best->ticket && again.accepted() == sa.accepted());

    SimulatedAnnealing<Bits> lam([](calx::random::Stream& r) { return heuresis::variation::random_bits(16, r); }, flip_one, Schedule::modified_lam(3, steps), Draws(12));
    std::uint64_t at20 = 0, at60 = 0;
    for (std::size_t k = 0; k < steps; k++) {
      auto c = lam.ask(1);
      const heuresis::search::Told told{c[0].ticket, heuresis::search::scalar(q(c[0].genotype))};
      lam.tell(std::span<const heuresis::search::Told>(&told, 1));
      if (k + 1 == steps / 5) at20 = lam.accepted();
      if (k + 1 == steps * 3 / 5) at60 = lam.accepted();
    }
    const double middle = static_cast<double>(at60 - at20) / (0.4 * steps);
    s.check("the modified Lam schedule holds the acceptance at 0.44 from 20% to 60% of the run", middle - 0.44, 0.02);
    s.check("and ends at the ground state", lam.current_energy() - ground, 1e-12);
  }

  return s.done();
}
