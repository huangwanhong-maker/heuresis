# 0009. Orders of Grants

- **Status:** Accepted, provisionally (2026-11-20)
- **Date:** 2026-11-20
- **Decided by:** Claude, at the infrastructure's request for an evolutionary framework of allocation in which "allocate A first then B" differs from "B then A", after its survey *Evolutionary heterogeneous resource allocation* (Part II) and the design `development/ideas/2026-11-20-evolutionary-allocation-of-heterogeneous-resources.md` (its Stage 2).

## 1. Context

heuresis has permutations and six of their operators (`variation`), Plackett–Luce among its search distributions, and an evolutionary loop that evaluates each genotype once by its key (`evolution`). An allocation made one grant at a time needs more: a model of grants acting on a state, a test of which grants commute (where they do, order is redundant), decoders whose reach is stated (a decoder can exclude every optimum), operators chosen by the feature the objective depends on, and a way to tell whether a policy would do better than a sequence. morphe's `traces` now gives words their canonical forms (its decision 0021).

## 2. Decision

A module `sequential`:

- **Grant systems** over any state type: a partial action, optional footprints, an optional step of time.
- **Commutation for allocation:** either order, each grant skipped where not allowed, reaches the same state, and both orders grant both or neither. Tested at states of random walks, with witnesses; declared from footprints; contradictions between the two reported.
- **Evaluation by trace:** the loop's key becomes the trace of the decoded word, and a cache keyed so; redundancy measured.
- **Decoders with their reach:** serial skipping, deferring and stopping; stepped, with a separator letter for plans that wait.
- **Operators by feature:** PPX, OX2, POS, NWOX and edge recombination with shared edges kept, each with what it keeps exactly; the feature probe; operators for the loop by feature, which its schedule credits.
- **Words with fixed letter counts** as a space, for picking sequences and serial orders with quotas; Mallows's laws under three metrics, sampled exactly and fitted; the adaptivity bracket.

## 3. Alternatives Considered

- **Commutation as a partial monoid action** (both orders undefined counted as commuting). It is Mazurkiewicz's notion for executions, but for allocation it calls two claimants of the last unit independent; the checks found this when a stock of one unit was reported to commute.
- **Keying by the genotype.** Two orders of independent grants can decode to different words under a decoder that skips, so the key is the trace of the word granted.
- **Exact independence by kategoria.** Decided for finite systems in kategoria's `resources` (its decision 0004); heuresis does not depend on kategoria, and tests independence for systems of any size. The allocation domain, which may use both, can ask each.

## 4. Consequences

- 48 checks in `check_heuresis_sequential`.
- The allocation domain's planned `evolution` composes resources (kategoria), spaces (decision 0008) and orders (here).
- **Not yet:** evolved policies; the biased random-key GA; mechanisms run on words (the evaluator's).
