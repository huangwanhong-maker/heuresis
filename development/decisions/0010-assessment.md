# 0010. Assessment by Evaluators of Many Kinds

- **Status:** Accepted, provisionally (2026-11-20)
- **Date:** 2026-11-20
- **Decided by:** Claude, at the infrastructure's request that the fitness of an allocation may be "a economics value circulation simulation or even ABM, a LLM evaluator, a machine learning model", after its survey *Evolutionary heterogeneous resource allocation* (Part V) and its design (Stage 4). The user said "Continuer" after reviewing Stages 1–3.

## 1. Context

heuresis's `Evaluation` holds objectives, constraints, descriptors, a cost and a status. It does not say where a value came from, how certain it is, or what kind of evidence it is, and its planned module `noise` (decision 0004) was to handle noisy objectives. The survey's Part V found that evaluators of different kinds establish different things, that a language model's score must never be summed into fitness, and that noise is cheapest handled by replicates where selection is in doubt, races with stated guarantees and cascades by fidelity.

## 2. Decision

A module `assessment`, which takes the planned `noise`'s place:

- **The assessment record:** metrics with roles, values, uncertainties, replicates, per-unit outcomes and standings (verified, computed, estimated, proposed); a status with its reason; provenance.
- **Tiers before Pareto:** failures last, then binding constraints by Deb's rule, then Pareto over objectives strong enough to select, the weaker objectives only as tie-breakers and descriptors. By default what is computed binds and selects. The design asked that only verified constraints bind; a feasibility computed exactly by a declared model is as hard as a verified one, and it is estimates and proposals that must not bind, so the floor is declared and its default is computed.
- **A metric is never stronger than its evaluator.**
- **Noise:** replicates by a power calculation; races under Hoeffding's and the empirical Bernstein bounds, each stating its guarantee; common random numbers judged by the variance of differences.
- **Cascades** with a declared error bound and the guarantee that follows from it; **a cache** never overwritten, kept in order for sealing; **agreement** of surrogates with trust declared; **Bradley–Terry** with an order effect for pairwise judgements.
- **An assessor** that runs evaluators, caches, caps standings, merges, and gives the loop an `Evaluation` by the tiers, with re-assessment on fresh replicates.

## 3. Alternatives Considered

- **Extending `Evaluation`.** It is used by every algorithm of heuresis; the record is richer and maps onto it, so nothing that works now changes.
- **One weighted score with a weight per kind of evaluator.** The survey found no defensible weight between a closed form and a model's opinion; tiers state the order of evidence instead of pricing it.

## 4. Consequences

- 30 checks in `check_heuresis_assessment`; heuresis 753.
- The planned `noise` row of the README is replaced by `assessment`.
- The allocation domain's planned `evolution` assesses allocations through an assessor; the infrastructure's edges supply the language model's and people's judgements.
- **Not yet:** racing inside survival; surrogates wired to `models`; quality-diversity over descriptors from several evaluators.
