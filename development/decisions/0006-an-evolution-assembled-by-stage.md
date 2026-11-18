# 0006. An Evolution Assembled by Stage

- **Status:** Accepted
- **Date:** 2026-11-18
- **Decided by:** the user asked, for the infrastructure's search for needs (its ADR 0125), for "an evolutionary algorithm's based searching method" in which "in each stage of GA framework" as many strategies as possible are implemented, the strategy flexible in the foundation, old members taking part in evolution but recorded and thrown away "when efficient long time", and new members generated in many ways (a language model among them). Claude wrote the module `evolution`.

## 1. Context

heuresis holds many evolutionary algorithms (decision 0004), each a fixed composition: the generational GA, the (μ+λ) EA, NSGA-II, MAP-Elites, the island model. A caller who wants tournament parents with aging survival, an adaptive choice of operators, novelty, and an archive that keeps good members and retires idle ones has to write the loop again. The infrastructure's needs domain wants exactly that, over a genotype of its own (a statement of need), with generators of several kinds, proposals from outside, and an archive whose admission is its own judgement and whose members breed.

## 2. Decision

1. **A module `evolution`**, on `selection` and `multiobjective`: one loop over any copyable genotype, assembled from a choice at every stage, each a value read from its name (`of<E>`), so that a page, a script or a request states a whole algorithm as data (`strategies.hpp`):
   - **scoring:** the first objective by Deb's rule; a weighted sum; the objectives in order; Pareto fronts with crowding;
   - **parents:** uniform, tournament, roulette, linear ranking, Boltzmann, truncation, lexicase and ε-lexicase over the objectives as cases, a tournament on novelty; a share drawn from the archive;
   - **schedule** of operators (generators, mutations, crossovers): fixed weights, probability matching, adaptive pursuit (Thierens 2005), UCB1, Thompson sampling;
   - **credit:** a child better than its best parent, a child that survives, the improvement;
   - **survival:** generational with elites, (μ+λ), (μ,λ), steady state, aging (regularized evolution), deterministic crowding, restricted tournament, clearing, NSGA-II, ALPS age layers, MAP-Elites niches;
   - **novelty** as a further objective;
   - **archive:** admission by the caller's judge; retirement after a declared number of generations without a child that survives, by age, on the caller's recheck, or when the archive is full, each with its reason kept beside the member (`Retired`);
   - **termination:** generations, evaluations, stagnation, or a restart on stagnation.
2. **Operators are the caller's**: makers (generators of new genotypes, given the context of the population and the archive), mutations, crossovers, a distance (for crowding, restricted tournaments, clearing and novelty) and a key (identity). A child that copies a living member or a sibling is made again; one that repeats a genotype evaluated before is given its remembered evaluation, so that nothing is evaluated twice.
3. **Proposals from outside** (`propose`) enter the next generation as they are, credited to their source, a person or a language model, in the statistics (`proposals`, `proposals_surviving`). They are the seam of decision 0003 §2.5 made concrete for a population.
4. **The protocol is kept.** `ask` and `tell` as every heuresis algorithm; a generation's states do not depend on the order of telling. `State` holds everything to resume between generations, so that a caller can run a few generations a time and keep the state elsewhere; a resumed run equals one that never stopped.
5. **Records:** every child's lineage (operator, parents, objectives, whether it failed, survived, was admitted) and each generation's statistics (best and median objectives, diversity, births and survivals by operator, the schedule's probabilities, the archive's size, admissions and retirements).

## 3. Alternatives Considered

- **A strategy per class (policy-based design).** Composing templates of parents, survival and archive would fix the composition at compile time; a caller choosing on a page needs it at run time, so the stages are values.
- **The archive as a population of its own (two-population coevolution).** That is the planned `coevolution`. Here the archive is a store with admission and retirement, whose members are drawn as parents with a declared share.

## 4. Consequences

- 42 checks (`evolution/tests/check_evolution.cpp`): every survival, parent and schedule strategy reaches a known optimum; aging removes the oldest; the archive retires by inefficacy, recheck and capacity with each reason; proposals are credited; runs do not depend on the order of telling or on the seed's streams other than their own; a resumed run equals one that never stopped; NSGA-II on LOTZ; novelty keeps diversity; restart on stagnation; refusals of strategies that lack what they need. heuresis now holds 590 checks.
- The infrastructure's needs domain composes the module over statements of need (its ADR 0125).
- `coevolution` stays planned; the survey made for ADR 0125 (`research/literature/reports/Evolutionary exploration and validation of needs.md`, §7.2) names what it should hold: competitive fitness sharing and a hall of fame, Pareto coevolution with archives of distinguishing tests, Bongard and Lipson's choice of tests by disagreement.
