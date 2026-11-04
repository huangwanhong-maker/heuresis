# heuresis

*Search and optimisation over spaces too large to enumerate.* The name comes from the Greek *εὕρεσις*, "finding, discovery", the root of *heuristic* and of *εὕρηκα*.

## 1. Purpose

Discovery has stages: a question, the search for a solution, an experiment, an evaluation. heuresis is the search. It is for spaces of designs, plans, parameters, programs and structures that are too large to enumerate and have no gradient to follow. It searches them with populations that vary and select, with search distributions that move towards what was good, with archives that keep diverse solutions, and with models that predict what is not yet evaluated. Evolutionary computation is its first and largest part. Beside it are the other global, stochastic, black-box searches: annealing, swarms, local search and model-based search.

**No free lunch.** Averaged over all problems, every search does as well as random search (Wolpert and Macready 1997). A search succeeds only on problems whose structure it exploits, so heuresis states for each of its operators what structure it assumes. The domains supply the structure, as representations, constraints and models: a molecule's valence, a robot's body, a city's budget.

heuresis names no object in nature. It stands on the project's other foundations:
- [calx](https://github.com/huangwanhong-maker/calx), for random streams, linear algebra, local minimisers and threads;
- [morphe](https://github.com/huangwanhong-maker/morphe), for grammars;
- [physis](https://github.com/huangwanhong-maker/physis), for physical dimensions, which type programs;
- [tyche](https://github.com/huangwanhong-maker/tyche), for Gaussian conditioning (surrogate models) and the statistics of runs.

It is one of the foundation libraries of [The Science Infrastructures](https://github.com/huangwanhong-maker/The-Science-Infrastructures), and it stands on its own.

## 2. Design

The contracts ([decision 0003](development/decisions/0003-the-search-process-and-its-seams.md)) are what keeps it open to algorithms, devices and machines not yet known:

- **Ask and tell.** An algorithm is a state that offers candidates and takes their evaluations; it never calls its objective. The same algorithm runs serially, on threads, on remote workers, on a device, or with a person in the loop.
- **Order-free generations.** A generational algorithm updates once its whole generation is told, in ticket order. Its states do not depend on where or in what order its evaluations were made.
- **Keyed draws.** Every random draw is identified by what it is for (a purpose, an epoch, an index), so a run is the same on one thread or many machines.
- **Provenance.** Every individual carries its ticket, its parents and the operation that made it. A run is a genealogy that can be analysed and replayed.
- **Five seams**, each additive:
  - where evaluations run (executors, and simulated asynchronous workers in the core);
  - representations and their kernels;
  - search distributions and their samplers (a quantum device or an annealer is a sampler of a law, with the CPU's exact emulation of that law as its reference);
  - migration between islands (the core coordinates, the edges transport);
  - external proposers (a person, a language model or an agent), whose proposals enter as injected individuals with their origin.

## 3. Modules

| Module | Content | Status |
|---|---|---|
| [`search`](search/) | Evaluations and Deb's order; keyed draws; the ask–tell protocol; drivers on one thread, a pool, simulated asynchronous workers (with budgets of simulated time); records; independent restarts of any algorithm; random search | draft |
| [`variation`](variation/) | Real boxes and repairs, bit strings, integer boxes, permutations; their mutations, crossovers and distances | draft |
| [`selection`](selection/) | Tournaments with a rule for ties, truncation, ranking, roulette and universal sampling, Boltzmann weights, lexicase; comma, plus, generational and steady-state survivors | draft |
| [`benchmarks`](benchmarks/) | Pseudo-Boolean, continuous, constrained, multi-objective, permutation and repertoire problems with known optima, fronts and hypervolumes | draft |
| [`genetic`](genetic/) | Evolutionary algorithms on bit strings and any representation: (1+1), (1+λ), (μ+1), the (1+(λ,λ)) GA, generational and steady-state GAs | draft |
| [`local`](local/) | Hill climbing, iterated local search, tabu search, variable neighbourhood descent, 2-opt; memetic refinement by calx's minimisers | draft |
| [`strategies`](strategies/) | Evolution strategies: the (1+1)-ES, CMA-ES with its restarts and separable form; differential evolution and SHADE | draft |
| [`swarms`](swarms/) | Particle swarms; ant colonies on permutations | draft |
| [`distributions`](distributions/) | Search distributions and their samplers: UMDA, PBIL, the compact GA, the cross-entropy method, Plackett–Luce; the information-geometric update | draft |
| [`quantum`](quantum/) | The quantum-inspired EA as a product distribution; Grover adaptive search on an exact emulation of the device's law; the sampler contract for devices | draft |
| [`annealing`](annealing/) | Simulated annealing, parallel tempering, population annealing and its free energy | draft |
| [`multiobjective`](multiobjective/) | Pareto order, non-dominated sorting, exact hypervolume, indicators; NSGA-II, NSGA-III, SMS-EMOA, MOEA/D | draft |
| [`diversity`](diversity/) | Novelty search, MAP-Elites on grids and Voronoi cells, quality-diversity scores, sharing and clearing | draft |
| [`programs`](programs/) | Programs as typed terms and as derivations of grammars; programs typed by physical dimensions; grammatical evolution; symbolic regression | draft |
| [`populations`](populations/) | Island models, cellular algorithms on graphs, takeover times | draft |
| [`models`](models/) | Gaussian-process surrogates on tyche; expected improvement and its kin; Bayesian optimisation | draft |
| [`landscapes`](landscapes/) | Autocorrelation, fitness–distance correlation, elementary landscapes, local optima and basins | draft |
| [`analysis`](analysis/) | Runtimes, their distributions over targets, expected running time, profiles, comparisons of algorithms | draft |
| `graphs` | Graphs as genotypes, varied by morphe's double-pushout rewriting under application conditions | planned |
| `coevolution` | Competitive and cooperative coevolution, archives of opponents | planned |
| `neuroevolution` | Networks of tyche's `learning` evolved in weights (natural evolution strategies) and in topology (NEAT) | planned |
| `noise` | Noisy objectives: resampling, racing, uncertainty handling | planned |
| `dynamics` | Objectives that change in time: memory, diversity maintenance, tracking | planned |
| `configuration` | Algorithm configuration and selection: racing (F-race, irace), hyper-heuristics, portfolios | planned |

## 4. Principles

- **General methods, not particular problems.** heuresis knows spaces, kernels, orders and archives. It knows no molecule, gait or city; those are the domains'.
- **Reuse before writing.** What calx, morphe, physis or tyche hold is used, never written again. A general method heuresis needs from them is written there.
- **Truth from theory.** Every algorithm is checked against closed forms where the theory gives them: runtimes, invariances, stationary laws, takeover times, hypervolumes. Otherwise it is checked against brute force on small spaces, and against reference implementations by their deterministic parts.

## 5. Relation to Sibling Libraries

heuresis depends on calx, morphe, physis and tyche. Inside the infrastructure, it uses the copies under `foundations/`. Built on its own, it would fetch them through the build system (not yet written). No library of the project depends on the infrastructure's domains or applications.

## 6. Status

**Draft (2026-11-05).** The first eighteen modules are written and checked, 548 checks in all ([decision 0004](development/decisions/0004-the-first-modules.md)), and improved by what fourteen case studies of the infrastructure found ([decision 0005](development/decisions/0005-what-the-case-studies-taught.md)); the rest are planned. Inside The Science Infrastructures, build and check with CMake 3.20 or later (the siblings are taken from the neighbouring directories):

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

or, from the infrastructure's root, `node development/tools/run-cpp.mjs foundations/heuresis foundations/heuresis`.

## 7. License

heuresis is released under the MIT License. See [`LICENSE`](LICENSE).
