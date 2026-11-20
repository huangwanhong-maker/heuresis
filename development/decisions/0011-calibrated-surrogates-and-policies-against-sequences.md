# 0011. Surrogates with Calibrated Intervals, and Policies Against Sequences

- **Status:** Accepted, provisionally (2026-11-21).
- **Date:** 2026-11-21
- **Decided by:** Claude, at the infrastructure's Stage 8 of evolutionary allocation (the user, before sleeping: "please continue all remain stage if unfinish"), after its survey (*Evolutionary heterogeneous resource allocation*): Part VII §6.1, item 3 (surrogates need calibrated uncertainty, checked against the true model, and nothing chosen on them alone), and the gap of Parts I and II (no study compared evolved sequences against evolved policies on one allocation problem).

## 1. Context

`assessment` measured a surrogate's agreement with its true evaluator (Kendall's τ, the coverage of ±zσ intervals that assume the surrogate's errors normal) but held no interval whose coverage is guaranteed. `sequential` searched words, one instance's order of grants; a rule that decides each grant from the state, learned once for a family of instances, was not represented, and the two had never been set side by side.

## 2. Decision

`assessment/conformal.hpp`:

- `Prediction`, `Predictor` (a centre and a scale at a point), `Interval`;
- `calibrate(predictor, x, y, α)` → `Conformal`: split conformal intervals (Lei et al. 2018), the score |y − ŷ|/σ̂ on runs the predictor was not fitted on, the ⌈(n + 1)(1 − α)⌉-th smallest as the quantile, the whole line when n is too small, the guarantee [1 − α, 1 − α + 1/(n + 1)] reported;
- `gaussian_process(x, y, normalised)`: models' Gaussian process fitted by its marginal likelihood, as a plain or normalised predictor;
- `screen(candidates, conformal, truth)` → `Screen`: candidates whose lower end lies above the least upper end are screened out, the rest assessed truly, the least true value chosen, and the coverage achieved on what was assessed reported.

`sequential/policies.hpp`:

- `Instance<S>` (a grant system, features of a grant at a state, a worth); `follow` (each step the allowed grant of greatest score, until none or below a threshold); `linear` and `program` scores; `worth`, `mean_worth`;
- `train_linear` (CMA-ES over the weights on a family's training instances) and `train_program` (genetic programming over the features);
- `search_sequence` (CMA-ES over random keys decoded by the deferring rule, for one instance), `best_order` (every order of a small instance);
- `Comparison`, the record of a comparison.

## 3. Alternatives Considered

- **Bayesian intervals from the Gaussian process alone.** Their coverage holds only if the process is the truth; the conformal interval keeps the process's adaptivity (as its scale) and gets its coverage from exchangeability.
- **Policies by reinforcement learning.** A policy as a scored choice among the allowed grants is what dispatching rules are, searchable by the library's own CMA-ES and genetic programming; learned value functions are a later representation.

## 4. Consequences

`check_heuresis_conformal` (8 checks): the order statistic exactly; the whole line when n is too small; over 200 replications coverage 0.902 (plain and normalised) within the theorem's band [0.90, 0.91] for n = 99, α = 0.1; normalised widths narrower where the noise is small; a screen keeping the true minimiser while assessing 48 of 200 candidates; a predictor calibrated where it was right covering 0.52 of what it screened, and saying so. `check_heuresis_policies` (10 checks): on knapsacks whose price rises 12% with every grant, thirty training and thirty held-out instances of eight grants, each instance's best order enumerated: no policy or searched sequence above the best; sequences searched per held-out instance reach 93.5%, 98.1%, 98.6% and 98.9% of the best with 10, 30, 100 and 300 evaluations each; a linear policy learned once from 600 passes over the training instances reaches 95.5% on held-out instances (and 95.9% on its training ones, relative to their best), a program 95.3% with the score value · ratio²: the policy is beaten by a sequence searched with 30 evaluations per instance and beats one searched with 10.
