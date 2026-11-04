# genetic: Evolutionary and Genetic Algorithms

*Status: draft (2026-11-05): the elitist algorithms of the theory, the (1+(λ,λ)) GA, and the genetic algorithm of composable kernels, generational and steady-state ([decision 0004](../development/decisions/0004-the-first-modules.md)).* Estimation-of-distribution algorithms are in `distributions`, evolution strategies in `strategies`, island and cellular models in `populations`.

## 1. Concepts

An evolutionary algorithm keeps a population, breeds offspring by variation (mutation, crossover) and selects survivors. The theory's elitist algorithms keep the best: the (1+λ) EA keeps one parent and replaces it by its best offspring when that is not worse, so it drifts across plateaus; the (1+1) EA with standard bit mutation at rate 1/n is its simplest case, and randomised local search (one bit flipped) its local relative. Their runtimes on OneMax and LeadingOnes are known exactly, which makes them the calibration of every other check. The (μ+1) EA keeps μ and replaces the worst; with a crossover before the mutation it is the (μ+1) GA.

The (1+(λ,λ)) GA uses crossover differently: a mutation phase flips ℓ ≈ λ bits in λ mutants, and a crossover phase repairs the best mutant against the parent, taking each differing bit with probability 1/λ. Large steps explore and the crossover keeps what helped, so with λ adjusted by a one-fifth rule its time on OneMax is linear, where every mutation-only elitist algorithm needs Θ(n log n).

The genetic algorithm composes its kernels at run time: a sampler, a mutation, a crossover (each a `std::function`, from `kernels.hpp` or the caller), tournament selection, and survivors by generational replacement with elites, plus or comma selection. Its steady-state form breeds whenever asked, at most `breadth` offspring from one state of the population, and each offspring told replaces the worst; it keeps every worker busy, and its run depends on the order of arrival.

Every algorithm here follows the ask–tell protocol: the generational ones update once a generation is told, in ticket order, so their states do not depend on the order of telling, the executor or the workers; all of them are Migratory, sending their best and receiving immigrants in place of their worst.

## 2. Content

| Header | Content |
|---|---|
| `kernels.hpp` | `Sampler<G>`, `Mutation<G>`, `Crossover<G>` as `std::function`s; `random_bits`, `bit_flip`, `bit_flip_positive`, `flip_one`, `heavy_tailed`, `uniform_crossover<T>`, `k_point_crossover<T>`, `uniform_point`, `polynomial`, `simulated_binary` |
| `elitist.hpp` | `OnePlusLambda<G>(sample, mutate, λ, draws)`; `one_plus_one_ea(n, draws, p)`; `randomised_local_search(n, draws)`; `MuPlusOne<G>(μ, sample, mutate, draws, crossover, p_c)` |
| `lambda_lambda.hpp` | `OnePlusLambdaLambda(n, LambdaLambdaOptions{lambda, self_adjusting, F}, draws)`: the practice-aware (1+(λ,λ)) GA on bit strings (ℓ ≥ 1; copies of x or x′ not evaluated) |
| `algorithm.hpp` | `Operators<G>{sample, mutate, crossover}`; `GaOptions{mu, lambda, crossover_rate, tournament, survival, elites, breadth}`; `Survival{generational, plus, comma}`; `GeneticAlgorithm<G>`; `SteadyStateGA<G>` |
| `genetic.hpp` | everything above |

## 3. Checks

`check_heuresis_genetic` (21):
- the (1+1) EA's mean runtime on OneMax (n = 30, 600 runs) within 4 standard errors of the exact expectation, computed from the Markov chain on the number of ones; that expectation within 3% of e n ln n − 1.8925 n + ½ e ln n (Hwang et al. 2018);
- on LeadingOnes (n = 25) within 4 standard errors of 1 + ((1 − p)^{1−n} − (1 − p))/(2p²);
- randomised local search on OneMax (n = 40) within 4 standard errors of its coupon-collector expectation;
- the self-adjusting (1+(λ,λ)) GA linear on OneMax (runtime/n at n = 800 over n = 100 within 25% of 1, against the (1+1) EA's ratio ≈ ln 800/ln 100), and faster than the (1+1) EA at n = 800;
- the GA's final population the same, ticket for ticket, when told in shuffled orders, run on a pool of four threads, cut into parts of 7, or run on five simulated workers with random durations;
- elitism; OneMax solved; an immigrant entering and leaving first;
- a real-coded GA (SBX, polynomial mutation) reaching below 10⁻² on the ten-dimensional sphere (measured 6 × 10⁻⁶);
- the steady-state GA reproducible on workers, offering its initial μ together and then one offspring per ask, and solving OneMax under the serial driver and on workers;
- the (μ+1) EA and GA solving OneMax with a strictly decreasing anytime curve.

## 4. Dependencies

`search`, `variation`, `selection`; in the checks, `benchmarks`.

## 5. References

- Böttcher, S., Doerr, B. and Neumann, F. (2010), Optimal fixed and adaptive mutation rates for the LeadingOnes problem, *PPSN XI*, 1–10.
- Carvalho Pinto, E. and Doerr, C. (2018), Towards a more practice-aware runtime analysis of evolutionary algorithms, arXiv:1812.00493.
- Dang, D.-C., Friedrich, T., Kötzing, T., Krejca, M. S., Lehre, P. K., Oliveto, P. S., Sudholt, D. and Sutton, A. M. (2018), Escaping local optima using crossover with emergent diversity, *IEEE Trans. Evol. Comput.* 22, 484–497.
- De Jong, K. A. (2006), *Evolutionary Computation: A Unified Approach*, MIT Press.
- Doerr, B. and Doerr, C. (2018), Optimal static and self-adjusting parameter tuning for the (1+(λ,λ)) genetic algorithm, *Algorithmica* 80, 1658–1709.
- Doerr, B., Doerr, C. and Ebel, F. (2015), From black-box complexity to designing new genetic algorithms, *Theor. Comput. Sci.* 567, 87–104.
- Droste, S., Jansen, T. and Wegener, I. (2002), On the analysis of the (1+1) evolutionary algorithm, *Theor. Comput. Sci.* 276, 51–81.
- Goldberg, D. E. (1989), *Genetic Algorithms in Search, Optimization, and Machine Learning*, Addison-Wesley.
- Holland, J. H. (1975), *Adaptation in Natural and Artificial Systems*, University of Michigan Press.
- Hwang, H.-K., Panholzer, A., Rolin, N., Tsai, T.-H. and Chen, W.-M. (2018), Probabilistic analysis of the (1+1)-evolutionary algorithm, *Evol. Comput.* 26, 299–345.
- Syswerda, G. (1991), A study of reproduction in generational and steady-state genetic algorithms, *FOGA 1*, 94–101.
- Whitley, D. (1989), The GENITOR algorithm and selection pressure, *ICGA 3*, 116–121.
- Witt, C. (2006), Runtime analysis of the (μ+1) EA on simple pseudo-Boolean functions, *Evol. Comput.* 14, 65–86.

## 6. Open Questions

1. Fitness-dependent and other parameter control (the (1+1) EA with self-adjusting rates, the (1+λ) EA's two-rate scheme) as a part to compose rather than one class each.
2. Whether the GA should take its parent and survivor selection as `std::function`s too, so that lexicase or Boltzmann selection can be swapped in without a new class.
