# distributions: Search Distributions and Their Samplers

*Status: draft (2026-11-05): product Bernoulli, product categorical, Gaussian and Plackett–Luce families; the reference sampler and the sampler seam; the information-geometric update in its generic and closed forms; UMDA, PBIL, the compact GA, categorical UMDA, the cross-entropy method, the pure rank-μ strategy and the Plackett–Luce EDA as one generational search ([decision 0004](../development/decisions/0004-the-first-modules.md)).* Mallows models on permutations, Bayesian networks of variables (BOA), mixtures and natural evolution strategies (xNES) are planned.

## 1. Concepts

Many searches keep no population, only a law P_θ on the space, and move θ towards what was good. The library keeps three things apart ([decision 0003](../development/decisions/0003-the-search-process-and-its-seams.md) §2.8): the **family** (θ, the density, its geometry), the **sampler** (the device that draws x ~ P_θ; the CPU's reference sampler defines the law, and any other device — an accelerator, a quantum device, an annealer — is tested against it), and the **rule** that moves θ after a generation is ranked.

The information-geometric optimisation of Ollivier, Arnold, Auger and Hansen (2017) makes the rules one: rank i of λ gets a weight ŵᵢ = W((i − ½)/λ)/λ (tied points share), and θ moves by δt Σ ŵᵢ I(θ)⁻¹∇ ln P_θ(xᵢ), a natural-gradient step that depends on the objective only through its order and on the parametrisation only to second order in δt. In the expectation parameters η of an exponential family the natural gradient is T(x) − η, so

- on bits (η = p) the step is PBIL's, and at δt = 1 with truncation weights UMDA's;
- on Gaussians in (m, C) it is the rank-μ update of CMA-ES with equal learning rates; in expectation parameters at δt = 1 with truncation weights it is the cross-entropy method's refit; the two differ by exactly δt² d̄d̄ᵀ, d̄ the mean's step;
- the compact GA is the step with λ = 2, signed weights ±½ and δt = 2/K.

`igo_step` computes the step generically from a family's score and Fisher information (a Cholesky solve); the checks hold it to the closed forms, which checks the theory as well as the code.

The cross-entropy method's covariance, smoothed geometrically, shrinks faster than its mean can travel on a slope and stalls short of the optimum (premature convergence); Kroese, Porotsky and Rubinstein's dynamic smoothing, β_t = β − β(1 − 1/t)^q, decays it only polynomially, and carries it to the optimum.

Plackett–Luce draws a ranking item by item with probabilities proportional to positive weights; equivalently, it orders the items by log wᵢ plus a standard Gumbel variable. Its weights are fitted by Hunter's minorise–maximise iteration, which raises the likelihood at every step.

**A law that has collapsed** samples (nearly) one point for ever, and a search on it spends the rest of its budget there. Each family says when its law has collapsed at a tolerance, `collapsed(family, θ, tolerance)`, and `stop_when_collapsed` makes a search stop then:
- a Gaussian, when its largest standard deviation √λ_max(C) is at most the tolerance, in every direction, a criterion invariant under rotations;
- Bernoulli frequencies, when every one is within ε of 0 or 1. Without borders, ε = 0 says that the law is a point mass. With the borders [1/n, 1 − 1/n], ε = 1/n says that every frequency is at a border: the law still explores, so this is a criterion for restarts;
- categorical variables, when each has a value of probability at least 1 − ε; a Plackett–Luce law, when its mode has that probability.

No search stops by default.

## 2. Content

| Header | Content |
|---|---|
| `family.hpp` | concepts `Family`, `ExponentialFamily` (statistic, expectation, from_expectation), `Riemannian` (coordinates, score, Fisher information) and `Sampler`; `Reference`; `quantile_weights`, `truncation_weights`, `logarithmic_weights` (CMA-ES's), `tied_weights`; `expectation_step`, `igo_step` |
| `bernoulli.hpp` | `ProductBernoulli`; rules `Umda` (with the borders [1/n, 1 − 1/n]), `Pbil`, `Compact`; `collapsed` |
| `categorical.hpp` | `ProductCategorical{sizes}`; `CategoricalUmda` with the multi-valued borders 1/((k − 1)n); `clamp_categorical`; `collapsed` |
| `gaussian.hpp` | `Gaussian{n}` with `GaussianParameters{m, C}` (score and Fisher information in (m, C)); rules `RankMu`, `CrossEntropy` (smoothing, dynamic smoothing of the covariance, a variance floor); `largest_deviation`, `collapsed` |
| `plackett_luce.hpp` | `PlackettLuce` (sequential and Gumbel samplers); `plackett_luce_fit` (Hunter's MM, weighted rankings, a floor); `PlackettLuceRule`; `mode_probability`, `collapsed` |
| `distribution_search.hpp` | `DistributionSearch<F, Rule, S>`, the generational ask–tell search of any family, rule and sampler, with a stopping predicate on θ (`stop_when`) or the family's test of a collapsed law (`stop_when_collapsed`); `IgoRule`, `ExpectationStep`, `FisherStep` |

## 3. Checks

`check_heuresis_distributions` (43):
- the generic IGO step equal to PBIL's update on bits and to the rank-μ update on Gaussians, to rounding; the expectation step at δt = 1 equal to UMDA's and the cross-entropy refit; the two Gaussian parametrisations differing by δt² d̄d̄ᵀ exactly;
- the Gaussian score centred with covariance the Fisher information (Monte Carlo), its log-density, the Bernoulli law summing to 1;
- the compact GA's mean step on OneMax against its expectation by enumeration of pairs;
- UMDA, PBIL and the compact GA solving OneMax (n = 40), categorical UMDA a four-valued OneMax;
- the cross-entropy method stalling with constant smoothing (f > 0.1 on the sphere) and reaching 10⁻⁶ on the sphere (n = 5) and 10⁻⁴ on Rosenbrock's function (n = 3) with dynamic smoothing;
- both Plackett–Luce samplers against the law on S₃; Hunter's MM monotone and recovering six weights from 20 000 rankings within 0.01; the Plackett–Luce EDA finding a hidden permutation of 8, also with the Gumbel sampler through the device seam;
- tied weights; independence of the order of telling;
- a collapsed law: a Gaussian's largest deviation √λ_max(C) = 4 × 10⁻⁶ for a rotated C whose coordinates' deviations are 2.9 × 10⁻⁶; Plackett–Luce's mode probability; categorical and Plackett–Luce tests on hand examples; the stalled cross-entropy method stopped once collapsed to 10⁻⁸ (4,700 evaluations), UMDA without borders once a point mass, the compact GA once every frequency is at a border (982 evaluations).

## 4. Dependencies

heuresis `search`, `variation`; calx `linalg` (Cholesky, solves, determinants) and `random`.

## 5. References

- Baluja, S. (1994), Population-based incremental learning, Carnegie Mellon University CMU-CS-94-163.
- Ben Jedidia, F., Doerr, B. and Krejca, M. S. (2023), Estimation-of-distribution algorithms for multi-valued decision variables, GECCO, 230–238.
- Ceberio, J., Mendiburu, A. and Lozano, J. A. (2013), The Plackett–Luce ranking model on permutation-based optimization problems, IEEE CEC, 494–501.
- de Boer, P.-T., Kroese, D. P., Mannor, S. and Rubinstein, R. Y. (2005), A tutorial on the cross-entropy method, *Ann. Oper. Res.* 134, 19–67.
- Grahl, J., Bosman, P. A. N. and Rothlauf, F. (2006), The correlation-triggered adaptive variance scaling IDEA, GECCO, 397–404.
- Hansen, N. (2016), The CMA evolution strategy: a tutorial, arXiv:1604.00772.
- Harik, G. R., Lobo, F. G. and Goldberg, D. E. (1999), The compact genetic algorithm, *IEEE Trans. Evol. Comput.* 3, 287–297.
- Hunter, D. R. (2004), MM algorithms for generalized Bradley–Terry models, *Ann. Statist.* 32, 384–406.
- Kroese, D. P., Porotsky, S. and Rubinstein, R. Y. (2006), The cross-entropy method for continuous multi-extremal optimization, *Methodol. Comput. Appl. Probab.* 8, 383–407.
- Magnus, J. R. and Neudecker, H. (2019), *Matrix Differential Calculus with Applications in Statistics and Econometrics*, 3rd ed., Wiley.
- Mühlenbein, H. and Paaß, G. (1996), From recombination of genes to the estimation of distributions I. Binary parameters, PPSN IV, 178–187.
- Ollivier, Y., Arnold, L., Auger, A. and Hansen, N. (2017), Information-geometric optimization algorithms: a unifying picture via invariance principles, *JMLR* 18(18), 1–65.
- Plackett, R. L. (1975), The analysis of permutations, *Appl. Statist.* 24, 193–202.
- Rubinstein, R. Y. (1999), The cross-entropy method for combinatorial and continuous optimization, *Methodol. Comput. Appl. Probab.* 1, 127–190.
- Witt, C. (2019), Upper bounds on the running time of the univariate marginal distribution algorithm on OneMax, *Algorithmica* 81, 632–667.
- Yellott, J. I. (1977), The relationship between Luce's choice axiom, Thurstone's theory of comparative judgment, and the double exponential distribution, *J. Math. Psychol.* 15, 109–144.

## 6. Open Questions

1. Families with dependencies (Bayesian networks, Mallows models, Gaussian mixtures), and natural evolution strategies as an `IgoRule` on a Cholesky parametrisation.
2. Batch sampling on a device: a sampler that draws a whole generation at once, so that a backend can fill it in bulk.
