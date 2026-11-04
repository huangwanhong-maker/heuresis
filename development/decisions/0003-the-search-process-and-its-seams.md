# 0003. The Search Process and Its Seams

- **Status:** Accepted, provisionally
- **Date:** 2026-11-04
- **Decided by:** the user, who required that the design be "generalizable, extendable" for "many future evolutionary algorithms, emerging computation devices, like even distributed evolutionary computing, or quantum evolutionary algorithms"; Claude designed the contracts and seams below.

## 1. Context

New evolutionary algorithms appear every year; most recombine a few parts (a representation, variation, selection, an adaptation, an archive). Evaluations are increasingly simulations that run elsewhere: on a pool of threads, on remote workers, on accelerators, on instruments. Searches are spread over islands on many machines, asynchronously. Some samplers are not classical: quantum devices, annealers and probabilistic bits draw from distributions that a CPU can only emulate. And the infrastructure's core must stay pure: no I/O, no hidden state. A design that hard-codes any of these (an algorithm that calls its objective, a random generator shared by threads, a population held in one process) closes the door on the others.

## 2. Decision

### 2.1 One Object: the Search Process

A search algorithm is a state S with two maps, `ask: S → S × candidates` and `tell: S × evaluations → S`, and draws keyed by what they are for. Mathematically it is a Markov chain on S driven by the evaluations; every algorithm of the library is one, and a new algorithm is a new state type. The library provides parts to compose (representations, kernels, selections, archives, adaptations) and algorithms built of them; an algorithm is never the only way to reach its parts.

### 2.2 The Ask–Tell Protocol (`search/protocol.hpp`)

- `ask(limit)` returns at most `limit` candidates, each with a ticket (unique in the run: the island in the top 16 bits, a serial below) and an origin (the operation's tag and the parents' tickets).
- `tell(told)` accepts evaluations of any subset of the outstanding tickets, in any order; an unknown or repeated ticket is refused.
- A **generational** algorithm offers its whole generation (in parts if asked so) and offers nothing more until all of it is told; it then updates once, reading the evaluations in ticket order. Its states therefore do not depend on the order in which evaluations return, nor on how the generation was cut.
- An **asynchronous** algorithm offers candidates whenever asked and updates at each tell, in the order told; a record of that order replays it. It offers at most `breadth` candidates bred from its current state per ask (by default 1). Without that cap, a driver with no batch limit would breed the whole budget from the initial state, which the first steady-state algorithm written (SMS-EMOA) showed: its hypervolume was 0.
- `stopped()` reports the algorithm's own termination; budgets belong to the driver.
- Optional: `inject(individuals)` accepts evaluated individuals from outside (migrants, archives, external proposers); `emigrants(k)` gives individuals to send. The concepts `Optimiser`, `Injectable` and `Migratory` state them.

The algorithm never calls its objective. That keeps it pure while the evaluation touches the world, and lets the same algorithm run serially, on threads, on a cluster, on a device, or with a person in the loop.

### 2.3 Evaluations (`search/evaluation.hpp`)

An evaluation carries objectives (all minimised), constraint values (satisfied at ≤ 0), descriptors (a behaviour, for the search for diversity), a cost (in the budget's unit), and a status with a reason: a failed evaluation is worse than any other and is never replaced by a made-up value. Single-objective comparison is Deb's feasibility rule; orders on several objectives are in `multiobjective`.

### 2.4 Provenance

Every individual records its ticket, its origin and its evaluation; a run's individuals form a genealogy (a directed acyclic graph). A `Record` keeps them in the order told: enough to analyse the run, draw its genealogy, or replay it.

### 2.5 Keyed Randomness (`search/draws.hpp`)

Every draw comes from a calx Stream whose key is the run's seed folded with the island, and whose identity is (purpose, epoch, index): a purpose tag (the FNV-1a hash of a name, so new algorithms declare their own), an epoch (a generation, a step), an index (an individual, a parent). A draw depends only on what it is for, never on the thread, process or machine that makes it. The consequence, checked for every generational algorithm: a run gives the same states serially, on a pool of threads, cut into parts, or on simulated workers with random latencies.

### 2.6 Seam 1: Where Evaluations Run (executors)

The driver (`search/run.hpp`) asks, evaluates through an executor, and tells. The core provides `Serial`, `Pool` (calx's thread pool) and simulated asynchronous workers (`search/workers.hpp`, a discrete-event simulation of the master–worker model with heterogeneous durations, reporting the simulated wall time). Any type with the executor's call can replace them at the edges: a cluster's job queue, an accelerator's batch, an instrument. An objective may take the ticket as well as the genotype, so that a noisy evaluation draws its noise from a stream keyed by the ticket and stays reproducible.

### 2.7 Seam 2: Representations and Their Kernels (`variation`)

A genotype is any value type. A representation is a space with its sampling, its kernels (mutations: one parent, one child; crossovers: two parents, two children; recombinations: many parents, one child), its repairs and its distances. Kernels draw from the stream they are given and from nothing else, so they commute with the symmetries they claim (the checks hold, for example, that standard bit mutation and uniform crossover commute exactly with the translations of (ℤ/2)ⁿ). Algorithms that compose kernels at run time take them as `std::function`s, so that a work can choose them from a registry; algorithms whose arithmetic is fixed (CMA-ES) are written directly.

### 2.8 Seam 3: Search Distributions and Their Samplers (`distributions`, `quantum`)

Many algorithms are a search distribution P_θ whose parameters move towards what was good: PBIL, UMDA and the compact GA on bits, the cross-entropy method and CMA-ES on ℝⁿ, and, in the information-geometric view, all of them as one natural-gradient flow (Ollivier, Arnold, Auger and Hansen 2017). The library separates the family (its parameters, its update) from its **sampler**, the device that draws x ~ P_θ. The CPU sampler is the reference: it defines the correct law, as calx's reference interpreter defines correct results. A quantum device, an annealer or a probabilistic-bit machine is another sampler of a law it defines, and the core holds an exact classical emulation of that law wherever one exists, against which the device can be tested (the law after r Grover iterations is known in closed form). Quantum-inspired algorithms, whose "qubits" are unentangled, are shown to be search distributions of the product family.

### 2.9 Seam 4: Migration (`populations`)

An island is any `Migratory` algorithm; islands are joined by a topology (a directed graph) and exchange individuals (genotype, evaluation, provenance) under a policy (when, how many, whom to send, whom to replace). Synchronous migration is deterministic and placement-free; asynchronous migration is simulated in the core with keyed latencies. The core coordinates; transports (threads, processes, machines) are the edges'. A migrant is a plain value, so it can be serialised by whoever carries it.

### 2.10 Seam 5: External Proposers

A proposal from outside the library — a person, a language model, an agent reaching the infrastructure through its API — enters as an injected individual whose origin names it and its parents. The archive, selection and provenance stay in the pure core; the proposer lives at the edges. This is the shape of program search with language models as mutation operators (Romera-Paredes et al. 2024, FunSearch; Novikov et al. 2025, AlphaEvolve), with the infrastructure's simulations as their evaluators.

### 2.11 Bulk Arithmetic

Populations of real vectors are held as rows of contiguous arrays where an algorithm's arithmetic is bulk (sampling a generation, updating a covariance), so that a calx backend can later replace the loops on a device without changing an interface. heuresis addresses no device itself (the infrastructure's §3.5).

### 2.12 What Is Left Open

How a device generates code is calx's; how an island's migrants travel is the edges'; quantum circuits as such (registers, gates, measurement) are physis's `quantum`, when it holds them.

## 3. Alternatives Considered

- **Algorithms that call their objective** (an `evolve(problem)` method, as pagmo, DEAP and pymoo's `minimize` do). Rejected: the algorithm then owns the loop, so evaluations cannot be spread, delayed, batched across algorithms, or made by something that is not a function; and purity is lost.
- **One shared random generator.** Rejected: results would depend on the order in which threads draw.
- **A plug-in registry in the core.** Deferred: C++ concepts and `std::function` kernels are enough for composition; registries of named algorithms belong to the works that offer them in a page (as the Genome Workbench's toolbox does).

## 4. Consequences

- Every generational algorithm is checked for independence of the order of telling, the executor and the workers.
- A new algorithm, representation, sampler or transport is additive: a new type, nothing edited.
- Asynchronous algorithms are reproducible only with their record of arrivals; this is stated where they are.
