# selection: Parents and Survivors

*Status: draft (2026-11-05): tournaments with a rule for ties, truncation, ranking, fitness-proportional sampling, Boltzmann weights, lexicase; comma, plus, generational and steady-state survivors ([decision 0004](../development/decisions/0004-the-first-modules.md), [decision 0005](../development/decisions/0005-what-the-case-studies-taught.md)).*

## 1. Concepts

**Selection by order.** Selection reads a population in one of two ways. Most methods read it through an order: a comparator on indices, by default Deb's rule on the evaluations. Selection by order (tournaments, truncation, ranking, lexicase) is invariant under every strictly increasing transformation of the objectives, so a search driven by it is invariant too.

**Selection by weights.** Fitness-proportional sampling and Boltzmann weights read the population through numbers the caller derives from it. They depend on the scale of those numbers, and say which scale they assume.

**Probabilities.** Selection probabilities are known in closed form, and the checks hold the frequencies to them.
- A tournament of k drawn with replacement chooses the individual of rank i (1 = best) of N with probability ((N − i + 1)ᵏ − (N − i)ᵏ)/Nᵏ.
- Linear ranking with pressure s gives the best s/N and the worst (2 − s)/N.
- Baker's stochastic universal sampling gives each individual ⌊e⌋ or ⌈e⌉ copies of its expected number e, at every spin.

**Lexicase.** When individuals are scored case by case (data points, tests, scenarios), lexicase selection filters the population through the cases in a random order. Specialists survive that an average would bury. ε-lexicase admits, at each case, everyone within the median absolute deviation of the best.

**Ties and drift.** On a plateau the contestants of a tournament tie, and the tie rule sets how fast the population drifts. Two children share a parent with probability c = Σ q_j², q_j the probability that j wins; a neutral label's heterozygosity falls by 1 − c a generation.
- `Ties::first` (the default) gives a tie to the earlier index. Under equal fitness the population's order is then read as a ranking, and a tournament of two has c = (4N² − 1)/(3N³) ≈ 4/(3N): an effective population of about 3N/4.
- `Ties::random` gives a tie to a contestant drawn uniformly among the tied, by a fresh draw from the tournament's stream. Under equal fitness the winner is uniform and c = 1/N, the drift of Wright–Fisher's model.

**Survivors.** Comma selection forgets the parents, so the search can leave an optimum and its step size can adapt. Plus selection is elitist. Ties fall to the earlier index throughout, so every selection is a function of its stream.

## 2. Content

| Header | Content |
|---|---|
| `parents.hpp` | `Less`, `by_evaluation`, `order`; `tournament` (with `Ties::first` or `Ties::random`), `tournament_probability`; `truncation`; `linear_ranking`, `exponential_ranking`, `by_rank`; `boltzmann_weights`; `roulette`, `roulette_draws` (many draws from cumulative weights built once, O(n + count log n)), `universal_sampling` |
| `survivors.hpp` | `comma`, `plus`, `generational` (with elites), `worst`, `reverse_tournament` |
| `lexicase.hpp` | `lexicase`, `case_epsilons` (ε-lexicase) |
| `selection.hpp` | all of the above |

## 3. Checks

`check_heuresis_selection` (23):
- Tournament frequencies by rank against the closed form, which sums to 1, under either tie rule.
- Drift under equal fitness (M = 20, 400 000 pairs of tournaments of two): coalescence (4M² − 1)/(3M³) with ties to the earlier index and 1/M with ties at random, each within four standard errors.
- Tournaments and truncation identical, under the same stream, after the transformation x ↦ e^{3x} − 40.
- Linear ranking's sum and extreme probabilities, and exponential ranking's ratio.
- Universal sampling within one of every expected count at every one of 20 000 spins, and unbiased. Roulette's frequency. `roulette_draws` equal to 200 000 successive calls of roulette on the same stream, and its frequencies.
- Comma, plus and generational survivors on a worked example.
- Lexicase choosing each of two specialists with probability ½ and never the generalist.
- ε as the median absolute deviation, and ε-lexicase keeping both near-best individuals.

## 4. Dependencies

`search` (evaluations, Deb's order); calx `random`.

## 5. References

- Baker, J. E. (1985), Adaptive selection methods for genetic algorithms, *ICGA 1*, 101–111.
- Baker, J. E. (1987), Reducing bias and inefficiency in the selection algorithm, *ICGA 2*, 14–21.
- Blickle, T. and Thiele, L. (1996), A comparison of selection schemes used in evolutionary algorithms, *Evol. Comput.* 4, 361–394.
- De Jong, K. A. (1975), *An Analysis of the Behavior of a Class of Genetic Adaptive Systems*, PhD thesis, University of Michigan.
- Goldberg, D. E. and Deb, K. (1991), A comparative analysis of selection schemes used in genetic algorithms, *FOGA 1*, 69–93.
- Helmuth, T., Spector, L. and Matheson, J. (2015), Solving uncompromising problems with lexicase selection, *IEEE Trans. Evol. Comput.* 19, 630–643.
- La Cava, W., Spector, L. and Danai, K. (2016), Epsilon-lexicase selection for regression, *GECCO '16*, 741–748.
- Schwefel, H.-P. (1981), *Numerical Optimization of Computer Models*, Wiley.
- Spector, L. (2012), Assessment of problem modality by differential performance of lexicase selection in genetic programming, *GECCO '12 Companion*, 401–408.
- Whitley, D. (1989), The GENITOR algorithm and selection pressure, *ICGA 3*, 116–121.

## 6. Open Questions

1. **Selection under noise.** When evaluations are noisy, selection can resample or race the candidates it compares. This belongs with noisy optimisation, which is planned.
