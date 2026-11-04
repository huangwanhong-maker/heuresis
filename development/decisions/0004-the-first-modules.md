# 0004. The First Modules

- **Status:** Accepted, provisionally
- **Date:** 2026-11-04
- **Decided by:** the user, who asked that the first slice be "comprehensive as possible"; Claude chose the modules, their order and their checks.

## 1. Context

Decision 0003 fixed the contracts. This record fixes the first modules, what each holds, what it uses, and how it is checked: against closed forms where the theory gives them (runtimes, takeover times, hypervolumes, invariances, stationary laws), against brute force on small spaces, and against reference implementations by their deterministic parts.

## 2. Decision

The modules, in their order (each may use those above it):

| Module | Holds | Uses |
|---|---|---|
| `search` | evaluations and Deb's order; keyed draws; the ask–tell protocol, batches and pending tables; the drivers (serial, pool) and simulated asynchronous workers; records and progress; random search | calx `random`, `scheduler` |
| `variation` | real boxes and repairs; bit strings; integer boxes; permutations; Gaussian, Cauchy and polynomial mutation, SBX, BLX-α, intermediate recombination, iso+line; standard, positive, k-flip and heavy-tailed bit mutation; uniform and k-point crossover; reset, creep and Rudolph's integer mutation; swap, inversion, insertion and scramble; PMX, OX1 and CX; Hamming, Euclidean, Kendall and Cayley distances | `search` |
| `selection` | tournaments, truncation, linear and exponential ranking, roulette, universal sampling, Boltzmann weights, lexicase and ε-lexicase; comma, plus, generational and steady-state survivors | `search` |
| `benchmarks` | OneMax, LeadingOnes, Jump, Trap, NK, QUBO, planted SAT; the classical continuous functions, rotations and shifts, constrained G06; ZDT and DTLZ with their exact hypervolumes; Euclidean tours; the planar arm | `variation`, calx `linalg` |
| `genetic` | the (1+1) EA, (1+λ) and (μ+1) EAs, the (1+(λ,λ)) GA with its self-adjusting λ, a generic generational and steady-state GA of composable kernels | `variation`, `selection` |
| `local` | hill climbing (first and best improvement), randomised local search, iterated local search, tabu search, variable neighbourhood descent, 2-opt; memetic refinement by calx's Nelder–Mead and BFGS | `variation`, calx `optimisation` |
| `strategies` | the (1+1)-ES with the 1/5 rule; CMA-ES (Hansen 2016), with active updates, bounds and IPOP/BIPOP restarts, and its separable form; differential evolution (rand/1, best/1, current-to-best/1; bin and exp) and SHADE | `variation`, `selection`, calx `linalg` |
| `swarms` | particle swarms (inertia, constriction, SPSO 2011's rotation-invariant update), ant colonies on permutations (Ant System, MAX–MIN) | `variation`, `selection` |
| `distributions` | search distributions and their samplers: product Bernoulli (UMDA, PBIL, compact GA), product categorical, Gaussian (cross-entropy method), the IGO update for exponential families, Plackett–Luce on permutations | `variation`, `selection`, calx `linalg` |
| `quantum` | the quantum-inspired EA of Han and Kim as a product-Bernoulli search distribution; Grover adaptive search (Dürr–Høyer, Baritompa–Bulger–Wood) on an exact emulation of the device's law; the sampler contract for devices | `distributions` |
| `annealing` | simulated annealing with its schedules, parallel tempering, population annealing with its free-energy estimate | `variation`, `selection` |
| `multiobjective` | Pareto and constrained dominance; fast non-dominated sorting; crowding; exact hypervolume (two and three objectives by sweeps, any number by WFG) and its contributions; GD, IGD, IGD⁺, additive ε, spread; NSGA-II, NSGA-III, SMS-EMOA, MOEA/D | `variation`, `selection` |
| `diversity` | novelty search; MAP-Elites on grids and on centroidal Voronoi cells; quality-diversity scores; fitness sharing and clearing | `variation`, `selection` |
| `programs` | expression trees over typed signatures, ramped half-and-half, subtree crossover and mutations, bloat control; programs typed by physis's dimensions; grammar-guided programs on morphe's context-free grammars and grammatical evolution's mapping; symbolic regression | `variation`, `selection`, morphe `formal`, physis `units` |
| `populations` | island models (topologies, migration policies, synchronous and asynchronous), cellular algorithms on graphs, takeover times | `variation`, `selection` |
| `models` | surrogate models on tyche's Gaussian conditioning; expected improvement, probability of improvement, confidence bounds; Bayesian optimisation and its batches | `variation`, `selection`, tyche `spatial`, `probability`, calx `optimisation` |
| `landscapes` | random-walk autocorrelation and correlation length, fitness–distance correlation, elementary landscapes, local optima and their basins, neutrality | `variation`, tyche `descriptive` |
| `analysis` | runtimes to targets, empirical distribution functions of runtimes, expected running time, success rates, anytime curves, performance and data profiles, comparisons of algorithms | `search`, tyche `descriptive`, `probability`, `testing` |

Each check program draws its problems from `benchmarks`. The modules after `benchmarks` are written in parallel by sessions that share this record and decision 0003.

Requested from the siblings, when their working trees allow. Until then, the modules that need them say so and do without.
- **calx:** the fast Walsh–Hadamard transform, for Walsh spectra of landscapes; low-discrepancy sequences (Sobol, Halton), for space-filling designs and quasi-random sampling of behaviour spaces.
- **tyche:**
  - rank tests and effect sizes (Mann–Whitney, Wilcoxon signed-rank, Vargha–Delaney Â₁₂), for comparing algorithms;
  - k-means, for the centroids of Voronoi cells;
  - Latin hypercube designs, for the initial designs of surrogate models.
- **physis:** registers of qubits, gates and measurement, so that a simulated device can implement Grover's sampler and be held to its exact emulation.
- **morphe:** the algebraic simplification of terms, for the programs that symbolic regression finds.

## 3. Alternatives Considered

- **A narrow first slice** (search, variation, selection, strategies, benchmarks). Rejected by the user ("comprehensive as possible").
- **Graph genotypes by morphe's rewriting, coevolution, neuroevolution, noisy and dynamic optimisation, algorithm configuration and hyper-heuristics** in the first slice. Deferred: planned rows of the README, each to be written when a domain needs it.

## 4. Consequences

- The README's table lists every module with its status; the planned ones are named there.
- The checks of each module are listed in its README: 492 in all, every one passing (2026-11-05).
- Writing them taught four things, recorded where they apply:
  - an asynchronous algorithm must cap what it breeds per ask (decision 0003 §2.2);
  - CMA-ES on a box should search through Hansen's linear–quadratic map, not repair its samples (`variation/box_transform.hpp`);
  - the cross-entropy method needs dynamic smoothing of its covariance or it converges prematurely (`distributions`);
  - SPSO 2011's standard parameters do not contract its swarm above two dimensions, as the second moments of its update show (`swarms`).
