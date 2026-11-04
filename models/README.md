# models: Surrogate Models and Bayesian Optimisation

*Status: draft (2026-11-05): Gaussian-process surrogates on tyche's conditioning, with squared-exponential and Matérn-5/2 kernels and type-II maximum likelihood; expected improvement, the probability of improvement and the lower confidence bound, maximised by multistart; Bayesian optimisation as an asynchronous ask–tell algorithm with batches by the kriging believer and the constant liar ([decision 0004](../development/decisions/0004-the-first-modules.md)).* Surrogate-assisted evolution (a model screening an evolutionary algorithm's offspring), trust regions (TuRBO), the tree-structured Parzen estimator, constraint and failure classifiers, and multi-fidelity models are planned.

## 1. Concepts

When an evaluation costs hours — a simulation, an experiment — the cost of choosing the next point well is negligible. A **surrogate** models the objective as a draw from a Gaussian process with constant mean and a stationary covariance; its posterior at any point, given the evaluations, is tyche's Gaussian conditioning, extended one evaluation at a time in O(n²) with the log marginal likelihood read from its factor. heuresis writes none of that algebra again; it supplies the covariances. The kernel's variance, length scales (one per coordinate) and noise are chosen by maximising the marginal likelihood with calx's Nelder–Mead on their logarithms, within bounds that keep the covariance well conditioned.

An **acquisition** says where an evaluation is worth most: the expected improvement over the best value seen, (y* − μ − ξ)Φ(z) + σφ(z) with z = (y* − μ − ξ)/σ; the probability of improvement Φ(z); or the lower confidence bound μ − κσ. It is maximised over the box by keyed uniform samples, the best few refined by Nelder–Mead.

**Bayesian optimisation** maps the box onto the unit cube and standardises the values, so one prior on the hyperparameters serves every problem (its `standardisation()` and `to_unit` carry the surrogate back to the problem's units, to draw it there); it offers an initial design of 2d + 2 uniform points, then at most `breadth` points an ask, each maximising the acquisition. Points asked but not yet told — the rest of a batch, or evaluations still running on other workers — are filled in by a lie (the surrogate's mean, or the best value seen), which spreads a batch out and lets the search run on asynchronous workers. Failed or infeasible evaluations enter the surrogate with the worst value seen.

## 2. Content

| Header | Content |
|---|---|
| `process.hpp` | `Kernel{kind, variance, lengthscales}` (`KernelKind::squared_exponential`, `matern52`); `GaussianProcess(kernel, noise, mean)` (`observe`, `predict` → `Posterior{mean, variance}`, `log_likelihood`); `log_marginal_likelihood`; `fit_hyperparameters` → `Fit{kernel, noise, log_likelihood}`; `HyperparameterBounds` |
| `acquisition.hpp` | `expected_improvement`, `probability_of_improvement`, `lower_confidence_bound`; `Acquisition{kind, xi, kappa}`; `maximise_over_box` → `Maximum{x, value}` |
| `bayesian.hpp` | `BayesianOptimisation(box, BayesianOptions{initial, breadth, acquisition, lie, kernel, relevance, samples, starts, fit_evaluations}, draws)`; `Lie::kriging_believer`, `constant_liar`; `surrogate()` (in the cube and standardised units), `standardisation()` → `Standardisation{mean, deviation}` (`value`, `posterior`, to the problem's units), `to_box`, `to_unit` |

## 3. Checks

`check_heuresis_models` (21):
- for both kernels, the posterior mean, the posterior variance and the log marginal likelihood against a direct computation by calx's Cholesky factorisation, to 10⁻¹²; the Matérn kernel at a hand-computed distance;
- expected improvement and the probability of improvement against 10⁶ draws of the posterior; their limits without uncertainty;
- type-II maximum likelihood raising the likelihood and recovering the length scale of a draw from the prior within 25% (8% here);
- Bayesian optimisation reaching Branin's minimum within 10⁻⁴ and Hartmann-3's within 10⁻³ in 60 evaluations (to 2 × 10⁻⁸ and 5 × 10⁻⁸ in fact); the same seed giving the same run;
- a batch of four by the kriging believer spread apart; `breadth` capping an ask; a run on four simulated asynchronous workers reaching Branin within 10⁻³;
- the standardisation equal to the sample mean and deviation of the 60 values told on Branin; `to_unit` inverting `to_box`; the surrogate's mean, unscaled, reproducing every value at its point within 10⁻³ of the deviation (2.5 × 10⁻⁵ in fact).

## 4. Dependencies

heuresis `search`, `variation`; tyche `spatial` (Gaussian conditioning) and `probability` (the normal law); calx `optimisation` (Nelder–Mead), `linalg` and `random`.

## 5. References

- Ginsbourger, D., Le Riche, R. and Carraro, L. (2010), Kriging is well-suited to parallelize optimization, in *Computational Intelligence in Expensive Optimization Problems*, Springer, 131–162.
- Jones, D. R., Schonlau, M. and Welch, W. J. (1998), Efficient global optimization of expensive black-box functions, *J. Global Optim.* 13, 455–492.
- Kushner, H. J. (1964), A new method of locating the maximum point of an arbitrary multipeak curve in the presence of noise, *J. Basic Eng.* 86, 97–106.
- Močkus, J., Tiešis, V. and Žilinskas, A. (1978), The application of Bayesian methods for seeking the extremum, in *Towards Global Optimization* 2, North-Holland, 117–129.
- Rasmussen, C. E. and Williams, C. K. I. (2006), *Gaussian Processes for Machine Learning*, MIT Press.
- Shahriari, B., Swersky, K., Wang, Z., Adams, R. P. and de Freitas, N. (2016), Taking the human out of the loop: a review of Bayesian optimization, *Proc. IEEE* 104, 148–175.
- Snoek, J., Larochelle, H. and Adams, R. P. (2012), Practical Bayesian optimization of machine learning algorithms, NeurIPS.
- Srinivas, N., Krause, A., Kakade, S. and Seeger, M. (2010), Gaussian process optimization in the bandit setting: no regret and experimental design, ICML.

## 6. Open Questions

1. Hyperparameters by their posterior (slice sampling) rather than their maximum, and a prior on them; the space-filling initial design (Latin hypercubes and Sobol sequences belong to tyche or calx when written).
2. Constraints and failures: a classifier of feasibility, and the expected improvement weighted by its probability.
3. Higher dimensions: trust regions, additive kernels, and the cubic cost of the surrogate (sparse or local models).
4. Surrogate-assisted evolution: a model that ranks an evolutionary algorithm's offspring before they are evaluated, composing with `strategies` and `genetic` at the edges of the ask–tell protocol.
