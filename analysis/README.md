# analysis: The Performance of Searches

*Status: draft (2026-11-05): runtimes to targets read from a run's progress; the expected running time; the empirical distribution of runtimes over runs and targets; success rates with Clopper and Pearson's interval; anytime curves and their quantiles; performance and data profiles; a permutation test between two samples ([decision 0004](../development/decisions/0004-the-first-modules.md)).* Rank tests and effect sizes (Mann–Whitney, Vargha and Delaney's Â₁₂) wait for tyche; bootstrapped intervals of ERT, simulated restarts, and the aggregation of fixed-budget results are planned.

## 1. Concepts

A search is judged by what it spends to reach what it reaches. A run's outcome records each improvement of its best feasible value with the evaluations and cost spent so far ([search](../search/)), and what it spent in all; its **runtime** to a target is the evaluations at its first improvement at or below the target.

- The **expected running time**, ERT = (evaluations of every run until its success, or in all when it failed) / (successes), is the expected cost of restarting independent runs until one succeeds; +∞ without a success.
- The **empirical distribution of runtimes** over pairs (run, target) is the anytime measure of the COCO platform: at each budget, the fraction of pairs solved. Several targets per run turn a single success rate into a curve. To pool problems of different optima, each run is read **relative to its optimum**, its values replaced by their gaps f − f*, so that targets are precisions Δf, the same for every problem.
- **Success rates** come with Clopper and Pearson's exact interval, from tyche's beta quantiles.
- **Anytime curves** give the best value after each budget, and their quantiles over runs.
- Over a set of problems, the **performance profile** ρ_s(τ) is the fraction of problems on which solver s is within a factor τ of the best; the **data profile** d_s(κ) the fraction solved within κ(n_p + 1) evaluations, n_p the dimension, with Moré and Wild's convergence test.
- Two samples of results are compared by a **permutation test** of the difference of their means, with tyche's exact permutation p-value (b + 1)/(B + 1), never zero.

## 2. Content

| Header | Content |
|---|---|
| `runtimes.hpp` | `Run{progress, evaluations, cost}`, `run_of(outcome)`, `run_of(outcome, f*)`, `relative_to(run, f*)`; `runtime`, `best_at`; `expected_running_time` → `Ert{value, evaluations, successes, runs}`; `runtime_distribution`; `clopper_pearson` → `Proportion{estimate, lower, upper}`; `anytime_quantile` |
| `profiles.hpp` | `performance_profile`, `data_profile`, `evaluations_to_reduce` (Moré and Wild's test); `permutation_test` → `Comparison{difference, p_value, permutations}` |

## 3. Checks

`check_heuresis_analysis` (22), against examples computed by hand:
- runtimes and anytime values read from progress; ERT = (10 + 50 + 30)/2 = 45 on three runs, +∞ without success;
- the distribution of runtimes over six (run, target) pairs at six budgets;
- two problems of optima 5 and −3 pooled by their gaps: the distribution of runtimes to precisions 1 and 0.01, which the unshifted values misread, and ERT (8 + 5)/2 = 6.5;
- Clopper and Pearson's interval: 1 − 0.025^{1/10} for 0 of 10, its mirror for 10 of 10, [0.187086, 0.812914] for 5 of 10;
- anytime medians and a quantile that reaches a run with no value yet (+∞);
- performance and data profiles of two solvers on four problems, exactly; Moré and Wild's test;
- the permutation test: a shift of five told apart (p < 0.01), identical samples at p = 1, never below 1/(B + 1);
- a real run's runtime equal to the position of its first evaluation at the target.

## 4. Dependencies

heuresis `search`; tyche `descriptive` (quantiles), `probability` (the beta law) and `testing` (the permutation p-value).

## 5. References

- Auger, A. and Hansen, N. (2005), Performance evaluation of an advanced local search evolutionary algorithm, IEEE CEC, 1777–1784.
- Clopper, C. J. and Pearson, E. S. (1934), The use of confidence or fiducial limits illustrated in the case of the binomial, *Biometrika* 26, 404–413.
- Dolan, E. D. and Moré, J. J. (2002), Benchmarking optimization software with performance profiles, *Math. Program.* 91, 201–213.
- Hansen, N., Auger, A., Ros, R., Mersmann, O., Tušar, T. and Brockhoff, D. (2021), COCO: a platform for comparing continuous optimizers in a black-box setting, *Optim. Methods Softw.* 36, 114–144.
- Moré, J. J. and Wild, S. M. (2009), Benchmarking derivative-free optimization algorithms, *SIAM J. Optim.* 20, 172–191.
- Phipson, B. and Smyth, G. K. (2010), Permutation p-values should never be zero, *Stat. Appl. Genet. Mol. Biol.* 9, art. 39.
- Price, K. V. (1997), Differential evolution vs. the functions of the 2nd ICEO, IEEE ICEC, 153–157.

## 6. Open Questions

1. Rank tests (Mann–Whitney, Wilcoxon's signed ranks) and Vargha and Delaney's effect size are general statistics: requested of tyche's `testing`.
2. Bootstrapped confidence intervals of ERT and of runtime distributions (COCO's simulated restarts), and their place (tyche's resampling, when it exists).
3. Fixed-budget views (the value reached after b evaluations) aggregated over problems of different scales.
