# 0002. Scope, Boundaries and Dependencies

- **Status:** Accepted
- **Date:** 2026-11-04
- **Decided by:** the user, who asked for "a general evolutionary computing framework" for solution search in every science ("even in chemistry domain, it can be so challenging"), first as a domain named optimisation; Claude proposed a foundation instead, since search names no object in nature; the user agreed ("it should be a foundation"), chose the name *heuresis*, asked that the first slice be "comprehensive as possible", and that the design be generalisable and extensible: "future evolutionary algorithms, emerging computation devices, like even distributed evolutionary computing, or quantum evolutionary algorithms". Recorded in the infrastructure as its [ADR 0073](https://github.com/huangwanhong-maker/The-Science-Infrastructures/blob/main/development/decisions/0073-the-search-foundation-heuresis.md).

## 1. Context

Knowledge discovery has stages: a question, the search for a solution, an experiment (often a simulation), an evaluation. The infrastructure holds the experiments (the domains' models and simulations) and the evaluation (tyche), but not the search: calx's `optimisation` holds local, deterministic methods on ℝⁿ (Brent, Nelder–Mead, BFGS, the simplex method, the Hungarian method), morphe holds exact combinatorial optimisation. Spaces of designs, plans, programs and molecules are too large to enumerate and have no gradient; they are searched by populations, by search distributions, by archives of diverse solutions. Such methods name no object in nature and serve every domain, so by the infrastructure's principle they are written once, in a foundation.

## 2. Decision

1. **A library of its own.** Search and optimisation over spaces too large to enumerate is the library heuresis (Greek *εὕρεσις*, "finding, discovery", the root of *heuristic* and *eureka*), namespace `heuresis`, one target `heuresis::<module>` per module, in `foundations/heuresis`. Evolutionary computation is its first and largest part; it also holds the other global, stochastic, black-box searches beside it (annealing, swarms, local search, model-based search).
2. **The boundaries.**
   - calx keeps local, deterministic methods on ℝⁿ with convergence theory, and linear programming; heuresis calls them (a memetic algorithm's local refinement) and never writes them again.
   - morphe keeps exact combinatorial optimisation, grammars and graph rewriting; heuresis uses its grammars for programs and its rewriting for graphs as genotypes.
   - tyche keeps the fitting of statistical models and the statistics of comparison; heuresis uses its Gaussian conditioning for surrogate models and its descriptive statistics and tests for the analysis of runs.
   - physis keeps physical law; heuresis uses its dimensions to type programs, so that symbolic regression generates only dimensionally consistent formulas.
   - What names an object in nature is a domain's: a molecule's valence rules, a robot's gait, a city's budget. A domain states its problem (a space, objectives, constraints) and heuresis searches it.
   - What touches the world is an edge's or a work's: evaluators that run on clusters or devices, transports between islands, proposers that call a language model.
3. **Dependencies.** heuresis depends on calx, morphe, physis and tyche, and never on the infrastructure's domains, edges or works. Inside the infrastructure it uses the siblings under `foundations/`; built alone, it would fetch them. It does not nest them as submodules.
4. **Pure.** heuresis performs no I/O, holds no global state and calls no objective except through the driver it is given (decision 0003).
5. **Language.** C++20, header-only while the modules are templates and small classes (the infrastructure's ADR 0003 and ADR 0035).

## 3. Alternatives Considered

- **A domain named optimisation.** The user's first proposal. Rejected by the user on reflection: a domain names objects in nature, a foundation cannot use a domain, and every domain and some foundations need search.
- **A module of calx.** Rejected: calx is numerical methods on floating-point arrays; evolutionary computation searches bit strings, permutations, trees, grammars and graphs, and needs morphe, physis and tyche, which calx may not use.
- **Borrowing a library whole** (pagmo, DEAP, pymoo, Nevergrad, EvoX). Rejected for the core: their interfaces are not ours, most call the objective themselves (decision 0003 §3), and several are copyleft (DEAP is LGPL, pagmo GPL or LGPL). They remain references to compare against; pycma (BSD) and pymoo (Apache 2.0) give fixtures.

## 4. Consequences

- A change that spans libraries is committed in dependency order: calx and morphe, then physis, then tyche, then heuresis, then the infrastructure.
- A domain that searches (genomics' sequence design, robotics' gaits, urban science's plans, a future molecular science's molecules) states its problem and composes heuresis's algorithms; it writes no search of its own.
