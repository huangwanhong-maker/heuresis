# assessment: Candidates Assessed by Evaluators of Many Kinds

*Status: draft (2026-11-20; [decision 0010](../development/decisions/0010-assessment.md)). It replaces the planned `noise`.*

## 1. Concepts

A candidate may be assessed by a closed form, a simulation, an agent model, a surrogate, a language model or a person, and each establishes something different. A closed form gives its declared quantity exactly; a simulation shows that a mechanism suffices for a pattern; a surrogate ranks fast but only as far as it has been checked; a language model or a person proposes and prefers. Summing their scores into one fitness would let the weakest evidence decide.

So an **assessment** keeps every metric with its role (objective, constraint, descriptor, report), its value, its uncertainty and replicates, the per-unit outcomes it summarises, and its **standing**: verified, computed, estimated or proposed. Candidates are ordered by **tiers** before any Pareto comparison: failures last; then the binding constraints by Deb's rule; then Pareto dominance over the objectives strong enough to select; the weaker objectives only as tie-breakers and descriptors. A model's opinion can never outrank what a computed objective decides, even when it claims to be verified: a metric is never stronger than its evaluator.

Noise is handled where it costs: replicates by a power calculation, spent where selection is in doubt; **races** that select with a stated guarantee; common random numbers kept only where they help; **cascades** from cheap to costly evaluators with a declared error bound; a **cache** of every assessment; the **agreement** of a surrogate with what it stands for; and **Bradley–Terry** for pairwise judgements, with the effect of the order they were shown in.

## 2. Content

| Header | What it holds |
|---|---|
| `record.hpp` | `Metric`, `Provenance`, `Assessment` (roles, standings, kinds, statuses); `merge`, keeping the worst status; `from_samples` |
| `tiers.hpp` | `Tiering`; `to_evaluation` (only what may select becomes an objective, only what may bind a constraint); `tiered` and `tiered_ranking` |
| `noise.hpp` | `summarise`; `replicates_for` (normal quantiles from tyche); `in_doubt`; `race` under Hoeffding's or the empirical Bernstein bound; `common_numbers` |
| `cascade.hpp` | `cascade` by fidelity with its guarantee; `AssessmentCache` by input, evaluator, version, fidelity and stream; `Agreement` of a surrogate (Kendall's τ, RMSE, coverage) and `Trust` |
| `pairwise.hpp` | `bradley_terry` with an order effect, by Hunter's MM iteration; Ford's condition, or a declared prior |
| `evaluators.hpp` | `Evaluator` (name, version, kind, standing, fidelity, cost, stochastic); `Assessor`: every evaluator on a candidate, cached, standings capped, merged, given to the evolutionary loop by the tiers; `reassess` on fresh replicates |
| `assessment.hpp` | all of the above |

## 3. Checks

`tests/check_assessment.cpp`, 30 checks:

- merging keeps the worst status and refuses two metrics of one name;
- on 300 random sets, what stronger objectives decide no weaker one reverses; failures last; the infeasible after the feasible; an estimated constraint does not bind; a proposed objective travels as a descriptor;
- 63 replicates for a difference of half a standard deviation at 5% with power 0.8, the closed form, reaching that power in 4,000 simulated experiments;
- Hoeffding and empirical Bernstein races: no wrong selection among 393 and 250 decided races at δ = 0.1; where the variance is small, Bernstein's race spends 7,006 samples against Hoeffding's 70,906;
- common random numbers kept where they lower the variance of a difference, refused where they raise it;
- cascades select as exhaustive evaluation does on 200 instances, at 6% of its cost;
- the cache returns what it recorded without running again, and a new version runs afresh;
- a surrogate that agrees is trusted; one that does not is distrusted with its reason; coverage counted exactly;
- Bradley–Terry recovers known log-strengths within 0.1 and the advantage of being shown first from 8,000 judgements, its likelihood rising at every step; a lopsided set refused, then answered with a prior;
- an evolution assessed by a computed loss, a noisy estimate and an adversarial model finds the computed optimum; re-assessed, the estimate carries its error and the model's opinion stays proposed though it claimed to be verified.

## 4. Dependencies

heuresis's `search` (evaluations and keyed streams), `multiobjective` (non-dominated sorting), `spaces` (values and their keys); tyche's `probability` (normal quantiles); calx's random streams.

## 5. References

- Deb, K. (2000). An efficient constraint handling method for genetic algorithms. *Computer Methods in Applied Mechanics and Engineering* 186, 311–338.
- Maron, O., Moore, A. W. (1994). Hoeffding races: accelerating model selection search for classification and function approximation. *NIPS 6.*
- Mnih, V., Szepesvári, C., Audibert, J.-Y. (2008). Empirical Bernstein stopping. *ICML.* Audibert, J.-Y., Munos, R., Szepesvári, C. (2009). Exploration–exploitation tradeoff using variance estimates in multi-armed bandits. *Theoretical Computer Science* 410, 1876–1902.
- Chow, S.-C., Shao, J., Wang, H. (2008). *Sample Size Calculations in Clinical Research.* Chapman and Hall/CRC.
- Jin, Y. (2005). A comprehensive survey of fitness approximation in evolutionary computation. *Soft Computing* 9, 3–12.
- Bradley, R. A., Terry, M. E. (1952). Rank analysis of incomplete block designs. *Biometrika* 39, 324–345. Ford, L. R. (1957). Solution of a ranking problem from binary comparisons. *American Mathematical Monthly* 64, 28–33. Hunter, D. R. (2004). MM algorithms for generalized Bradley–Terry models. *Annals of Statistics* 32, 384–406. Agresti, A. (1990). *Categorical Data Analysis.* Wiley.

## 6. Open Questions

- **Not yet:** racing inside the evolutionary loop's survival (it races here as a tool); surrogates wired to `models`' Gaussian processes as evaluators with their agreement updated on every re-evaluation; quality-diversity archives over descriptors from several evaluators; a language model's judgement with its order swapped and k samples (the infrastructure's edges ask the model; here its judgements are read).
