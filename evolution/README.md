# evolution: An Evolution Assembled by Stage

*Status: draft (2026-11-18; [decision 0006](../development/decisions/0006-an-evolution-assembled-by-stage.md)).*

## 1. Concepts

An evolutionary algorithm is a loop of stages, and each stage has many strategies in the literature. Here one loop is assembled from a choice at every stage, over any copyable genotype, so that a caller states a whole algorithm as data:

| Stage | Strategies |
|---|---|
| scoring | the first objective by Deb's rule; a weighted sum; lexicographic; Pareto fronts with crowding |
| parents | uniform; tournament; roulette; linear ranking; Boltzmann; truncation; lexicase and ε-lexicase; a tournament on novelty; a share from the archive |
| schedule of operators | fixed; probability matching; adaptive pursuit; UCB1; Thompson sampling |
| credit | better than the best parent; survival; improvement |
| survival | generational with elites; (μ+λ); (μ,λ); steady state; aging (regularized evolution); deterministic crowding; restricted tournament; clearing; NSGA-II; ALPS; MAP-Elites niches |
| novelty | none, or the mean distance to the k nearest as a further objective |
| archive | admission by the caller's judge; retirement after long inefficacy, by age, on recheck, when full, each with its reason |
| termination | generations; evaluations; stagnation; restart on stagnation |

The operators are the caller's: makers (generators of new genotypes), mutations, crossovers, a distance and a key. Proposals from outside (a person, a language model) enter a generation as they are and are credited to their source.

## 2. Content

- `strategies.hpp`: the stages' enumerations, their names, and `Strategy`, every parameter in one value, validated.
- `credit.hpp`: the credit of operators and the schedules that choose them.
- `evolution.hpp`: `Evolution<G>` (ask–tell, `propose`, `inject`, `emigrants`, `step`, `state`, resumption from `State<G>`), `Member`, `Kept`, `Retired`, `Statistics`, `Lineage`, and `evolve`.

## 3. Checks

`tests/check_evolution.cpp`, 44 checks: every survival, parent and schedule strategy reaches a known optimum (OneMax, LeadingOnes); aging removes the oldest; the archive admits by its judge and retires by inefficacy, recheck and capacity, each reason kept; proposals credited, and a proposal made from members kept with its parents (decision 0007); runs independent of the order of telling; a resumed run equals one that never stopped; NSGA-II on LOTZ's front; novelty keeps a population diverse; restart on stagnation; genealogies; refusals.

## 4. Dependencies and References

On `selection` and `multiobjective` (and through them `search`, calx). Sources are in the header: Goldberg and Deb (1991); Baker (1987); Spector (2012); La Cava, Spector and Danai (2016); Lehman and Stanley (2011); De Jong (1975); Schwefel (1981); Whitley (1989); Real, Aggarwal, Huang and Le (2019); Mahfoud (1995); Harik (1995); Pétrowski (1996); Deb et al. (2002); Hornby (2006); Mouret and Clune (2015); Thierens (2005).
