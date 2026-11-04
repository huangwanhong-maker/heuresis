# CLAUDE.md

This file guides Claude Code in the heuresis repository.

heuresis is one of the foundation libraries of [The Science Infrastructures](https://github.com/huangwanhong-maker/The-Science-Infrastructures), which includes it as the git submodule `foundations/heuresis`. It is a project in its own right and must make sense without the infrastructure.

## 1. Purpose and Scope

heuresis implements search and optimisation over spaces too large to enumerate:

- the search process: evaluations, keyed draws, the ask–tell protocol, its drivers and records;
- representations and their kernels: real boxes, bit strings, integer boxes, permutations, programs, and later graphs;
- selection of parents and survivors;
- evolutionary algorithms, evolution strategies, differential evolution, particle swarms, ant colonies, local search, annealing;
- search distributions and their samplers, quantum-inspired and quantum search among them;
- many objectives; diversity and quality-diversity; island and cellular models; surrogate models;
- the analysis of landscapes and of runs; benchmarks with known optima.

It builds on four sibling libraries:

- **calx**, for random streams, linear algebra, local minimisers (Nelder–Mead, BFGS) and the thread pool;
- **morphe**, for grammars and, later, graph rewriting;
- **physis**, for physical dimensions;
- **tyche**, for Gaussian conditioning and the statistics of comparison.

Out of scope:

| Topic | Where it belongs |
|---|---|
| Local, deterministic methods on ℝⁿ with convergence theory (line searches, quasi-Newton, trust regions, SQP), linear programming | calx `optimisation` |
| Exact combinatorial optimisation, grammars, graph rewriting | morphe |
| Statistical models, their fitting, tests, clustering | tyche |
| Physical law, quantum states and circuits | physis |
| A particular problem: a molecule's rules, a robot's gait, a plan's budget | the domains of The Science Infrastructures |
| Evaluators on clusters or devices, transports between islands, proposers that call a language model | the infrastructure's edges and works |

The boundary is [decision 0002](development/decisions/0002-scope-boundaries-and-dependencies.md): a search method that names no object in nature is heuresis's. What knows a molecule, a body or a city is a domain's. What touches the world is an edge's.

## 2. Invariants

1. **Dependencies.** heuresis depends only on calx, morphe, physis and tyche among project libraries, and never on the infrastructure's domains, edges or applications.
2. **No particular objects.** Nothing in heuresis names an object in nature or a device. Benchmarks are functions; a device is a sampler of a law.
3. **Pure.** No I/O, no global state. Algorithms never call their objectives; only the drivers do, through the executor they are given ([decision 0003](development/decisions/0003-the-search-process-and-its-seams.md)).
4. **Keyed randomness.** Every draw comes from a `search::Draws` stream identified by (purpose, epoch, index). Nothing draws from a generator shared across individuals or threads.
5. **The protocol.** Every algorithm keeps the ask–tell semantics of `search/protocol.hpp`. A generational algorithm's states do not depend on the order of telling, the executor or the workers, and its checks say so.
6. **Reuse.** What a sibling holds is used through its public headers, never written again here. A general method heuresis needs that is not a search belongs in the sibling where it fits.
7. **Truth from theory.** Every algorithm is checked against a known truth: a closed form, brute force on a small space, or a reference implementation's deterministic part. Reference values come from running the reference (a script in `tests/fixtures/`), never from memory.
8. **Cited sources.** Every implemented method cites its source; equation or section numbers are given only where they are certain.

## 3. Design Philosophy

heuresis shares its governing principle with its sibling repositories: **modular, extensible, revisable**.

- **Status vocabulary:**
  - `planned`: named, no content yet;
  - `seed`: an idea, possibly wrong;
  - `draft`: being worked out;
  - `stable`: tested and relied upon;
  - `deprecated`: superseded, with a pointer to its replacement.
- **Structure assumed is structure stated.** By the No Free Lunch theorems a search is only as good as the structure it exploits. Each kernel's header says which symmetries it commutes with and which structure it assumes.
- **Parts before algorithms.** An algorithm is a composition of parts (a representation, kernels, selections, an adaptation, an archive). Its parts stay usable without it.

## 4. Repository Layout

```
heuresis/
├── CMakeLists.txt   the project: one header-only target heuresis::<module> per module (heuresis_module); the siblings added once
├── search/ · variation/ · selection/ · benchmarks/ · analysis/
├── genetic/ · local/ · strategies/ · swarms/ · distributions/ · quantum/ · annealing/
├── multiobjective/ · diversity/ · programs/ · populations/ · models/ · landscapes/
├── docs/            the theory note
└── development/     decisions · ideas · release notes · tools (check-tags.mjs)
```

Each module directory holds:
- `README.md`: concepts first, then the content, then the checks;
- `include/heuresis/<module>/`;
- `tests/check_<module>.cpp`, a check program on calx's `verification`, registered with CTest by `heuresis_module` in the top-level `CMakeLists.txt`.

Module directories are created with their first content; until then a module is a `planned` row in the README. The modules and their order are in [decision 0004](development/decisions/0004-the-first-modules.md).

## 5. Working Procedures

### 5.1 Module Contract

Each module has:

- a `README.md` giving its concepts, content, checks, dependencies, references and open questions;
- public headers in `include/heuresis/<module>/`, header-only C++20 for now;
- checks in `tests/`, with reference values in `tests/fixtures/` and the scripts that made them;
- optionally, Python bindings in `bindings/python/`.

### 5.2 Decisions and Ideas

- Decisions are numbered records in `development/decisions/`. Records are never rewritten; a later record supersedes an earlier one.
- Ideas are files named `development/ideas/YYYY-MM-DD-slug.md`, with status `seed`, `promoted` or `abandoned`.
- **Release notes.** At each commit, add to `development/release-notes/YYYY-MM-DD.md` (one file per day, one short section per commit) a concise summary in plain prose of what is new, what one can now do and what works better. It is not a list of changes (the git log keeps those), and it is not written between commits. This convention is shared with the sibling repositories.

### 5.3 Change Discipline

- Before restructuring or changing an invariant (§2), propose the change and wait for the user's agreement.
- Keep each change to one concept.

### 5.4 Shared Conventions

calx, morphe, physis, tyche, nomos, eikon, heuresis and the infrastructure keep these conventions identical:

- writing style;
- status vocabulary;
- decision, idea and release-note conventions.

Change them in every repository of [The Science Infrastructures](https://github.com/huangwanhong-maker/The-Science-Infrastructures) together: the foundation libraries and the infrastructure's repository.

### 5.5 Working Inside the Infrastructure

When you edit heuresis inside the infrastructure's `foundations/heuresis`:

1. commit in heuresis first;
2. then commit the updated pointer in the infrastructure.

When a change needs a sibling to change as well, commit them in dependency order: calx and morphe first, then physis, then tyche, then heuresis.

## 6. Conventions

- **Minimisation.** Every objective is minimised. A quantity to maximise is negated where the problem is stated.
- **Constraints.** A constraint value g(x) is satisfied when g(x) ≤ 0; its violation is max(0, g(x)).
- **Order.** On one objective, evaluations are compared by Deb's feasibility rule. Ties are broken by ticket, or by index within a generation.
- **Failures.** A failed evaluation says why, and is worse than any other. It is never replaced by a made-up value.
- **Tags.** Purposes are named `"<module>.<algorithm>.<purpose>"` and hashed by `search::tag`. `node development/tools/check-tags.mjs` verifies that no two names in the library share a hash; run it when a module is added.
- **Writing style.**
  - Headings are declarative, numbered and academic, never questions.
  - Concepts come before structure.
  - Label asides with *Rationale.*, *Remark.* or *Caveat.*
  - British spelling; English throughout.

## 7. Technology

- **Language: C++.** This follows ADR 0003 of The Science Infrastructures.
- **Provisional:** C++20, header-only targets `heuresis::<module>`; CMake 3.20 or later. Checks are programs on calx's `verification`, registered with CTest. On the development machine the compiler is GCC in WSL.
- **Dependency resolution.** Inside the infrastructure, heuresis uses the targets of `foundations/calx`, `foundations/morphe`, `foundations/physis` and `foundations/tyche`, adding each that no larger project has added. Never nest submodules.
- **Threads.** Evaluations run on calx's thread pool through `search::Pool`. The algorithms' own arithmetic is serial until a calx backend serves it.
- **Licences.** heuresis is MIT-licensed. Check every dependency before adopting it. pycma (BSD) and pymoo (Apache 2.0) may give fixtures; DEAP (LGPL) and pagmo (GPL or LGPL) may be read and compared against, not borrowed.

## 8. Open Decisions

1. **Bulk arithmetic on devices.** When a calx backend can run a generation's sampling and updates, which algorithms move first (CMA-ES in high dimension, large populations of a GA) is open.
2. **Quantum samplers beyond Grover's law.** Registers, gates and measurement are physis's `quantum`. Variational and annealing samplers wait for them.
3. **Graph genotypes.** morphe's double-pushout rewriting with application conditions is the planned representation. Which domain needs it first (a molecular science, an urban plan) decides when.
4. **A serial format for records and migrants**, for the edges.

## 9. Current State

As of 2026-11-05, the first eighteen modules are written ([decision 0004](development/decisions/0004-the-first-modules.md)) and revised by what fourteen case studies found ([decision 0005](development/decisions/0005-what-the-case-studies-taught.md)), each with its checks (548 in all): `search`, `variation`, `selection`, `benchmarks`, `genetic`, `local`, `strategies`, `swarms`, `distributions`, `quantum`, `annealing`, `multiobjective`, `diversity`, `programs`, `populations`, `models`, `landscapes` and `analysis`. `graphs`, `coevolution`, `neuroevolution`, `noise`, `dynamics` and `configuration` are planned.
