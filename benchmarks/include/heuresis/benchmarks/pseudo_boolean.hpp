// Functions of bit strings with known structure and known runtimes.
//
// Every benchmark of heuresis is stated as a minimisation whose optimum value is 0 unless said otherwise; the
// classical form, maximised, is c − f for the constant c given. They name no object in nature: they are the
// problems on which the theory of evolutionary computation states its exact results (Doerr and Neumann 2020,
// Theory of Evolutionary Computation, Springer, ch. 1–2).
//
// - OneMax: the number of zeros (classically, the number of ones, c = n).
// - LeadingOnes: n − the length of the longest prefix of ones (Rudolph 1997).
// - Jump_k: the classical Jump_k(x) = k + |x|₁ when |x|₁ ≤ n − k or |x|₁ = n, else n − |x|₁; here n + k − Jump_k,
//   a valley of width k before the optimum (Droste, Jansen and Wegener 2002, "On the analysis of the (1+1)
//   evolutionary algorithm", Theor. Comput. Sci. 276, 51–81, §5).
// - Trap: concatenated deceptive traps of order k, each block worth k when all ones and k − 1 − u otherwise, u
//   its ones (Deb and Goldberg 1993, "Analyzing deception in trap functions", FOGA 2, 93–108); here the total
//   minus its maximum.
// - NK landscapes: n bits, each with k neighbours, f = (1/n) Σ fᵢ(xᵢ, x_neighbours) with tables uniform on (0, 1)
//   (Kauffman and Levin 1987, "Towards a general theory of adaptive walks on rugged landscapes", J. Theor. Biol.
//   128, 11–45; Kauffman 1993, The Origins of Order). k tunes ruggedness from smooth (k = 0) to random (k = n − 1);
//   minimised here as 1 − f, its optimum unknown except by enumeration.
// - QUBO: f(x) = Σ_{i≤j} Q_ij xᵢ xⱼ, quadratic unconstrained binary optimisation, the form of Ising problems
//   (s = 2x − 1) that annealers and quantum devices accept (Lucas 2014, "Ising formulations of many NP
//   problems", Front. Phys. 2, 5); its optimum is unknown except by enumeration.
// - Planted k-SAT: m random k-clauses over n variables, each satisfied by a hidden assignment, f the number of
//   unsatisfied clauses (optimum 0, attained at least by the hidden assignment).
// Instances are drawn from streams keyed by (seed, instance), so they are the same wherever they are made.

#pragma once

#include <heuresis/search/draws.hpp>
#include <heuresis/variation/binary.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace heuresis::benchmarks {

using heuresis::variation::Bits;

inline double onemax(const Bits& x) { return static_cast<double>(x.size() - heuresis::variation::ones(x)); }

inline double leading_ones(const Bits& x) {
  std::size_t k = 0;
  while (k < x.size() && x[k]) k++;
  return static_cast<double>(x.size() - k);
}

inline double jump(const Bits& x, std::size_t k) {
  const std::size_t n = x.size(), u = heuresis::variation::ones(x);
  const double classical = (u <= n - k || u == n) ? static_cast<double>(k + u) : static_cast<double>(n - u);
  return static_cast<double>(n + k) - classical;
}

inline double trap(const Bits& x, std::size_t k) {
  if (k == 0 || x.size() % k) throw std::invalid_argument("heuresis::benchmarks::trap: the length must be a multiple of the order");
  double total = 0;
  for (std::size_t b = 0; b < x.size(); b += k) {
    std::size_t u = 0;
    for (std::size_t i = b; i < b + k; i++) u += x[i];
    total += u == k ? static_cast<double>(k) : static_cast<double>(k - 1 - u);
  }
  return static_cast<double>(x.size()) - total;
}

class NK {
 public:
  // Neighbours drawn uniformly without replacement among the other bits ("random neighbourhood").
  NK(std::size_t n, std::size_t k, std::uint64_t seed, std::uint64_t instance = 0) : n_(n), k_(k), neighbours_(n), tables_(n) {
    if (k >= n) throw std::invalid_argument("heuresis::benchmarks::NK: k must be below n");
    const heuresis::search::Draws d(seed);
    for (std::size_t i = 0; i < n; i++) {
      calx::random::Stream s = d(heuresis::search::tag("benchmarks.nk"), instance, i);
      std::vector<std::size_t> others;
      for (std::size_t j = 0; j < n; j++)
        if (j != i) others.push_back(j);
      for (std::size_t t = 0; t < k; t++) {
        const std::size_t r = t + s.below(others.size() - t);
        std::swap(others[t], others[r]);
        neighbours_[i].push_back(others[t]);
      }
      tables_[i].resize(std::size_t{1} << (k + 1));
      for (auto& v : tables_[i]) v = s.uniform();
    }
  }
  double fitness(const Bits& x) const {
    double f = 0;
    for (std::size_t i = 0; i < n_; i++) {
      std::size_t idx = x[i];
      for (std::size_t t = 0; t < k_; t++) idx = (idx << 1) | x[neighbours_[i][t]];
      f += tables_[i][idx];
    }
    return f / static_cast<double>(n_);
  }
  double operator()(const Bits& x) const { return 1 - fitness(x); }
  std::size_t n() const { return n_; }
  std::size_t k() const { return k_; }
  const std::vector<std::size_t>& neighbours(std::size_t i) const { return neighbours_[i]; }

 private:
  std::size_t n_, k_;
  std::vector<std::vector<std::size_t>> neighbours_;
  std::vector<std::vector<double>> tables_;
};

class Qubo {
 public:
  explicit Qubo(std::size_t n) : n_(n), q_(n * n, 0.0) {}
  // A random instance: Q_ij ~ N(0, 1) on and above the diagonal.
  static Qubo random(std::size_t n, std::uint64_t seed, std::uint64_t instance = 0) {
    Qubo q(n);
    calx::random::Stream s = heuresis::search::Draws(seed)(heuresis::search::tag("benchmarks.qubo"), instance, 0);
    for (std::size_t i = 0; i < n; i++)
      for (std::size_t j = i; j < n; j++) q(i, j) = s.normal();
    return q;
  }
  double& operator()(std::size_t i, std::size_t j) { return q_[i * n_ + j]; }
  double operator()(std::size_t i, std::size_t j) const { return q_[i * n_ + j]; }
  double value(const Bits& x) const {
    double f = 0;
    for (std::size_t i = 0; i < n_; i++) {
      if (!x[i]) continue;
      for (std::size_t j = i; j < n_; j++)
        if (x[j]) f += q_[i * n_ + j];
    }
    return f;
  }
  double operator()(const Bits& x) const { return value(x); }
  // The change of f when bit i is flipped, in O(n).
  double flip_delta(const Bits& x, std::size_t i) const {
    double d = q_[i * n_ + i];
    for (std::size_t j = 0; j < n_; j++) {
      if (j == i || !x[j]) continue;
      d += j < i ? q_[j * n_ + i] : q_[i * n_ + j];
    }
    return x[i] ? -d : d;
  }
  std::size_t n() const { return n_; }

 private:
  std::size_t n_;
  std::vector<double> q_;
};

class PlantedSat {
 public:
  PlantedSat(std::size_t n, std::size_t m, std::size_t k, std::uint64_t seed, std::uint64_t instance = 0) : n_(n), k_(k), hidden_(n) {
    if (k > n || k == 0) throw std::invalid_argument("heuresis::benchmarks::PlantedSat: 0 < k ≤ n");
    const heuresis::search::Draws d(seed);
    calx::random::Stream h = d(heuresis::search::tag("benchmarks.sat.hidden"), instance, 0);
    hidden_ = heuresis::variation::random_bits(n, h);
    for (std::size_t c = 0; c < m; c++) {
      calx::random::Stream s = d(heuresis::search::tag("benchmarks.sat.clause"), instance, c);
      for (;;) {
        std::vector<std::int64_t> clause;  // literal v + 1 or −(v + 1)
        std::vector<std::size_t> vars;
        while (vars.size() < k) {
          const std::size_t v = s.below(n);
          if (std::find(vars.begin(), vars.end(), v) == vars.end()) vars.push_back(v);
        }
        bool satisfied = false;
        for (auto v : vars) {
          const bool positive = s.uniform() < 0.5;
          clause.push_back(positive ? static_cast<std::int64_t>(v + 1) : -static_cast<std::int64_t>(v + 1));
          satisfied = satisfied || (hidden_[v] == (positive ? 1 : 0));
        }
        if (satisfied) { clauses_.push_back(std::move(clause)); break; }
      }
    }
  }
  double operator()(const Bits& x) const {
    std::size_t unsat = 0;
    for (const auto& c : clauses_) {
      bool sat = false;
      for (auto l : c) sat = sat || (l > 0 ? x[static_cast<std::size_t>(l - 1)] == 1 : x[static_cast<std::size_t>(-l - 1)] == 0);
      unsat += !sat;
    }
    return static_cast<double>(unsat);
  }
  const Bits& hidden() const { return hidden_; }
  std::size_t clauses() const { return clauses_.size(); }

 private:
  std::size_t n_, k_;
  Bits hidden_;
  std::vector<std::vector<std::int64_t>> clauses_;
};

}  // namespace heuresis::benchmarks
