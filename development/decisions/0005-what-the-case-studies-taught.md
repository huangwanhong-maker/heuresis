# 0005. What the Case Studies Taught

- **Status:** Accepted
- **Date:** 2026-11-05
- **Decided by:** the user asked for case studies of the algorithms, and Claude recorded what they found and fixed it.

## 1. Context

Fourteen case studies in The Science Infrastructures (`research/experiments/2026-11-05-*`) ran heuresis on problems with known answers and on real ones: a rabbit's gaits, a walk past Melbourne's sensors, a benchmark of continuous search, a district's design, the fitting of neurons, fronts in three objectives, Kepler's third law, quantum search, an arm's repertoires, NK landscapes, search across machines, search on a landscape drawn whole, the genetic algorithm as a population, and an image pipeline tuned on phantoms. Each says in its caveats where the library stood in its way: work-arounds written beside the library, defaults that wasted a budget, results that needed a derivation the library did not hold. This record lists the findings, the change each led to, and what is left open.

## 2. Decision

### 2.1 The Search Process (`search`)

1. **An exception was not a failure.** *Fitting neurons* wrapped its objective in a handler that turned an exception into `search::failure`, and *Benchmarking continuous search* wrote its own driver so that an algorithm that threw would end its run with its reason kept. Decision 0003 §2.3 intends both. Now `evaluate` turns a `std::exception` thrown by the objective into a failure whose reason is its message, and `run` and `run_asynchronous` end with `Ending::error` and the message (`Outcome::error`) when the algorithm's ask or tell throws, keeping what was spent and found.
2. **Random search sampled its whole budget at once.** Under `run` with a finite budget an ask's limit is the remaining budget, and `RandomSearch` built every candidate in it (*A walk past Melbourne's sensors* and *Benchmarking continuous search* capped their asks at 1000 to avoid it). It now has a `breadth`, by default 1 as for every asynchronous algorithm (decision 0003 §2.2), and the k-th sample is still the same however the asks are cut.
3. **No simulated time in progress or budget.** *Search across machines* reran every run with each target to learn when it was reached, and found that equal shares of an evaluation budget let the slowest machine set the makespan. `Progress` now records the simulated time of each improvement (0 under `run`), and `Budget::time` is a budget of simulated time for `run_asynchronous`: no evaluation starts at or after it, and those that would finish after it are abandoned, as with the target.
4. **No restarts but CMA-ES's.** *Rugged landscapes* found that the searches that settle (the (1+1) EA, the compact GA, UMDA, tabu search) would gain most from restarts, and heuresis had them for CMA-ES alone. `search/restarts.hpp` holds `Restarts<A>`: independent restarts of any algorithm of the protocol, a factory making run r from its number and its own keyed draws, a new run when the last has stopped or spent an optional budget of its own, tickets issued by the wrapper and unique across runs, the runs one genealogy. It is independent restarts and no more; growth of a population between runs (IPOP) is the factory's business.

### 2.2 Selection (`selection`)

5. **Roulette cost O(M) a draw.** `roulette` builds the cumulative weights at every call, so a generation of M parents cost O(M²) and *The genetic algorithm as a population* stopped at M = 200. `roulette_draws(weights, count, stream)` builds them once, O(n + count log n), and gives exactly the draws of `count` successive calls on the same stream.
6. **Ties lowered a tournament's effective population.** The same study measured that ties to the earlier index make a tournament of two drift a third faster than Wright–Fisher's model under equal fitness: coalescence (4M² − 1)/(3M³) a generation, an effective size of about 3M/4. Tournaments now take a tie rule: `Ties::first`, the default and unchanged, and `Ties::random`, a tie broken by a fresh uniform draw from the tournament's stream, with coalescence 1/M. The header states the drift of each.

### 2.3 Benchmarks (`benchmarks`)

7. **Schwefel's function is unbounded below off its box.** In *Benchmarking continuous search* the searches of ℝⁿ "solved" it by leaving [−500, 500]ⁿ, to gaps of −3660. The table now says so, and `schwefel_bounded` adds the study's penalty Σ max(0, |xᵢ| − 500)², the form of COCO's f20 penalty; no point outside the box beats f*, which the checks hold on a scan and on random points.
8. **ZDT3's and DTLZ7's fronts were missing.** *Fronts in three objectives* derived them as running records of functions of one variable, and their hypervolumes as integrals of the running extremes (1.044426 for ZDT3 with respect to (1, 1), 2.457669 for DTLZ7 with respect to (1, 1, 7)). The construction is now general in `benchmarks/multiobjective.hpp`: `running_records`, `running_extreme_integral`, the sampled fronts and the exact hypervolumes, DTLZ7's in any number of objectives, checked against the study's values, against brute-force non-dominated sorting and against Monte Carlo.

### 2.4 Algorithms

9. **The (1+1)-ES never stopped** (`strategies`). Its default `min_sigma` was 0, so after it settled it spent the rest of every budget there (*Benchmarking continuous search* set 10⁻¹² by hand; in *Search on a landscape* it spent the rest of its budget at a spread of 6 × 10⁻¹⁶). The default is now 10⁻¹² σ₀; `min_sigma = 0` still never stops.
10. **A collapsed distribution searched on** (`distributions`). The cross-entropy method in *Search on a landscape* stalled with a spread of 0.014 and kept spending; the compact GA and UMDA of *Rugged landscapes* settled with nothing to report it. Each family now tests whether its law has collapsed at a tolerance, `collapsed(family, θ, tolerance)`: a Gaussian whose largest standard deviation √λ_max(C) is at most it (invariant under rotations), Bernoulli frequencies all within it of 0 or 1, categorical variables and Plackett–Luce laws whose mode has probability at least 1 − it. `DistributionSearch::stop_when_collapsed` makes `stopped()` report it.
11. **Bayesian optimisation's surrogate could not be drawn in the problem's units** (`models`). *Search on a landscape* recomputed the standardisation from every value told to draw the surrogate's mean. `standardisation()` (the mean and deviation by which the values are scaled, with `value` and `posterior` to undo it) and `to_unit` are now public.
12. **The ant colony recomputed τ^α η^β at every step of every ant** (`swarms`). In *A walk past Melbourne's sensors* its construction dominated the compute (n² powers an ant). The weights are now computed once a generation, with the same rounding, so the colony is unchanged bit for bit; on a 100-city tour with 100 ants a generation is 2.7 to 3.0 times faster.
13. **The emulator's measurement was not const** (`quantum`). *Quantum search* shared one `EmulatedGrover` between runs because its `measure` reads only; it is now const, and the checks run four searches on one emulator on four threads.
14. **MAP-Elites needed an identity mutation beside a variation** (`diversity`). *Repertoires of an arm* passed `mutate` as the identity to use iso+line alone. Either operator now suffices in MAP-Elites and novelty search: with a variation alone every child is a variation, drawn from the stream as with p_v = 1 and an identity mutation, so the two forms give the same run.

### 2.5 Analysis (`analysis`)

15. **Runs on problems of different optima could not be pooled.** *Benchmarking continuous search* subtracted f* from every value of a run's progress before its distributions of runtimes. `relative_to(run, f*)` and `run_of(outcome, f*)` read a run by its gaps f − f*, so that targets are precisions and runs on any problems pool.

### 2.6 Defaults

Two defaults change, as their findings demand:
- the (1+1)-ES stops once σ < 10⁻¹² σ₀ (it never stopped before);
- `RandomSearch` offers one candidate an ask (it offered the whole limit). Its check of samples independent of the cut now gives the search asked for ten at once a breadth of 10.

Every other change is additive and every other default is kept: tournaments break ties by index, distribution searches stop only when told, and exceptions of types not derived from `std::exception` are not caught.

## 3. Alternatives Considered

- **Catching every exception (`catch (...)`).** Rejected: a failure needs a reason, and an exception of unknown type has none to give; the standard exceptions are what objectives and algorithms throw.
- **A stopping rule for distribution searches by default.** Rejected: the borders of UMDA, PBIL and the compact GA keep their laws exploring, so "every frequency at a border" is a reason to restart, not convergence, and a default on the Gaussian would change existing runs. The rule is the family's, the tolerance the caller's.
- **A coordinate-wise collapse of the Gaussian** (CMA-ES's TolX, on √C_ii). Rejected for the law's own test: the largest standard deviation does not depend on the coordinates.
- **Ties at random by default.** Rejected: it would change every tournament's draws and every GA's run. The default stays and its drift is stated.
- **Restarts that carry state from run to run** (IPOP within the wrapper). Rejected: what one run hands the next is a choice of the algorithm, made in the factory; CMA-ES's IPOP and BIPOP stay in `strategies`.
- **COCO's penalty coefficient for Schwefel's function.** In this function's units COCO's f20 penalty is n times the one adopted. Either keeps every point outside the box above f*; the study's, independent of n, was taken.

## 4. Consequences

- 547 checks in all (492 before), every one passing (2026-11-05). Every earlier check passes unchanged except random search's, which gives its search a breadth of 10 (§2.6).
- Left open, each to be written when a study or a domain needs it:
  1. **A nearest-neighbour index for novelty search.** Sparseness by a scan over the population and the archive made novelty search the slowest search of *Repertoires of an arm* (89–105 s a run against 0.3–7 s). A k-d tree or ball tree is a general method, for calx or tyche.
  2. **HV3D⁺ for SMS-EMOA.** Exact contributions in three objectives as differences of hypervolumes cost SMS-EMOA 183–271 s a run in *Fronts in three objectives*, against 0.5–3 s for the others; Guerreiro and Fonseca's incremental algorithm (2018, "Computing and updating hypervolume contributions in up to four dimensions", IEEE Trans. Evol. Comput. 22, 449–463) is the remedy `multiobjective` names.
  3. **A degree of failure.** In *A body learns to move* a rabbit that falls at 0.3 s and one that falls at 8 s are the same failure, and the searches have no direction in the region of falls. A failure that carries a measure (the time survived, as a constraint) without becoming a made-up value needs a convention for evaluations, decided apart.
  4. **Constants and simplification of programs.** *Kepler's third law* found its constants only to about 10⁻³ (drawn and redrawn, never fitted) and printed its formulas unsimplified. Local fitting by calx's minimisers, or linear scaling, and the algebraic simplification of terms (requested of morphe in decision 0004) are the remedies.
  5. **A budget of simulated time for the asynchronous islands.** *Search across machines* asked it of "the asynchronous drivers"; only `run_asynchronous` has it. `populations`'s driver runs whole cycles, and needs its own rule for a cycle in flight at the budget.
  6. **The tie rule in the genetic algorithms.** `genetic`'s algorithms still break ties by index; passing `Ties` through their options is the next step.
  7. **MOEA/D without normalisation.** *Designing a district* gave MOEA/D objectives relative to the district as built so that their scales were comparable; normalising the objectives by the ideal and nadir points is the remedy.
