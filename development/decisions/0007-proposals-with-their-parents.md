# 0007. Proposals with Their Parents

- **Status:** Accepted (2026-11-18)
- **Date:** 2026-11-18
- **Decided by:** Claude, at the infrastructure's request that a language model take part in every stage of its search for needs: seeding, varying one member, joining two, refining a kept one (its ADR 0126).

## 1. Context

`Evolution::propose` (decision 0006) takes genotypes from outside and credits them to their source, but records them as having no parents. A person's or a model's variation of a member is then cut from its ancestry. The lineage lies, and the member it came from is never credited as useful, even when its child survives.

## 2. Decision

`propose(genotypes, source, parents)` takes, optionally, the tickets of the members each proposal was made from: one list for each genotype, or none.

- **Lineage.** A proposal's lineage records its parents.
- **Age.** It inherits the oldest living parent's age, so that aging and ALPS treat a variation as a variation.
- **Credit.** When it survives, each parent still living or kept is credited as useful, as for a child of the evolution's own operators.
- **Refusal.** Parents given for some proposals and not for others are refused.

Proposals stay outside the operators' schedule: they are credited to their source in the statistics, not to an operator.

## 3. Alternatives Considered

- **Proposals as operators.** A schedule would choose and credit a person or a model like a mutation. But a proposal arrives when its author sends it, not when the schedule asks, and its source's survival is already counted in the statistics, where a caller can allocate requests by it (the needs domain does).

## 4. Consequences

- Two checks in `check_heuresis_evolution`: a proposal made from two members records them and credits them when it survives; parents for some proposals only are refused. heuresis 592 checks.
- The genealogies a caller writes (W3C PROV in the infrastructure) now carry a model's variations back to the members they came from.
