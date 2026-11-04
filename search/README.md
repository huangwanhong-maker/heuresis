# search: The Search Process and Its Drivers

*Status: draft (2026-11-05): evaluations and Deb's order, keyed draws, the ask–tell protocol, the drivers (serial, a thread pool, simulated asynchronous workers), records and random search ([decision 0003](../development/decisions/0003-the-search-process-and-its-seams.md)).*

## 1. Concepts

A search algorithm is a state with two maps: `ask`, which offers candidates, and `tell`, which takes their evaluations. It never calls its objective. Whoever evaluates (a function on this thread, a pool of threads, remote workers, an instrument, a device, a person) is the driver's business, so the algorithm stays pure while its evaluations touch the world.

Every candidate has a ticket, unique in the run, with the island in its top 16 bits so that it keeps its identity when it migrates, and an origin: the operation that made it and its parents' tickets. A run's individuals therefore form a genealogy, a directed acyclic graph.

Generational algorithms offer a whole generation and update once when all of it has been told, reading the evaluations in ticket order. Their states do not depend on the order in which evaluations return, on how a generation was cut into parts, or on where it was evaluated. Asynchronous algorithms update at each tell; their runs are reproducible from the order of arrival, which a record keeps.

Every random draw comes from a calx Stream identified by what it is for: a purpose (a tag, the FNV-1a hash of a name), an epoch and an index, under a key that folds the run's seed with the island. A draw never depends on which thread, process or machine makes it.

An evaluation carries objectives (minimised), constraint values (satisfied at ≤ 0), descriptors (a behaviour, for diversity), a cost, and a status. A failure says why and is worse than anything else; it is never replaced by a made-up value. On one objective, evaluations are ordered by Deb's feasibility rule: feasible before infeasible, then by objective among the feasible and by total violation among the infeasible.

The simulated workers are a discrete-event simulation of the master–worker model. Each evaluation takes a duration that depends on it, its worker and a keyed stream. The simulation reports the simulated wall time, so the cost of synchronisation can be measured before a search is spread over real machines.

## 2. Content

| Header | Content |
|---|---|
| `evaluation.hpp` | `Evaluation{objectives, constraints, descriptors, cost, status, reason}` with `failed`, `violation`, `feasible`, `value`; `scalar`, `evaluation`, `failure`; `better`, Deb's order (Deb 2000) |
| `draws.hpp` | `tag(name)` (FNV-1a, 32 bits), `mix` (SplitMix64's finaliser), `Draws(seed, island)(purpose, epoch, index)` → `calx::random::Stream` |
| `protocol.hpp` | `Ticket`, `make_ticket`, `island_of`, `serial_of`; `Origin`, `Candidate<G>`, `Told`, `Individual<G>`; the concepts `Optimiser`, `Injectable`, `Migratory`; `Batch<G, Note>` (a generation's bookkeeping), `Pending<G, Note>` (an asynchronous algorithm's outstanding candidates); `ranking` |
| `run.hpp` | `Budget{evaluations, cost, target}`, `Ending`, `Progress`, `Outcome<G>`, `Record<G>`; `evaluate` (an objective of the genotype, or of the genotype and its ticket; a number or an Evaluation); the executors `Serial` and `Pool`; `Elite<G>`; `run` |
| `workers.hpp` | `Workers{count, duration, seed}`; `run_asynchronous`, the discrete-event master–worker simulation |
| `random_search.hpp` | `RandomSearch<G>`, the asynchronous baseline (Wolpert and Macready 1997) |
| `search.hpp` | all of the above |

## 3. Checks

`check_heuresis_search` (26):
- FNV-1a's and SplitMix64's published vectors.
- Streams as functions of their identity.
- Deb's order, failures and not-a-number objectives.
- The protocol's refusals: unknown tickets, tickets told twice, a generation offering nothing until it is told.
- A generational search with the same states whether its evaluations are told in any order, run on a pool of four threads, cut into parts, or returned by simulated workers with random durations.
- The budget kept exactly, and the genealogy acyclic.
- A target ending the run.
- A generational search waiting for its slowest worker (makespan exactly 400). An asynchronous one keeping every worker busy (makespan 400/3.25).
- Random search's samples independent of how the run is cut into batches.

## 4. Dependencies

calx `random` (streams) and `scheduler` (the thread pool).

## 5. References

- Alba, E. and Tomassini, M. (2002), Parallelism and evolutionary algorithms, *IEEE Trans. Evol. Comput.* 6, 443–462.
- Deb, K. (2000), An efficient constraint handling method for genetic algorithms, *Comput. Methods Appl. Mech. Engrg.* 186, 311–338.
- Fowler, G., Noll, L. C., Vo, K.-P. and Eastlake, D., The FNV non-cryptographic hash algorithm, IETF draft-eastlake-fnv.
- Salmon, J. K., Moraes, M. A., Dror, R. O. and Shaw, D. E. (2011), Parallel random numbers: as easy as 1, 2, 3, *SC '11* (Philox, behind calx's streams).
- Scott, E. O. and De Jong, K. A. (2015), Understanding simulated asynchronous evolutionary algorithms, *FOGA '15*, 85–92.
- Steele, G. L., Lea, D. and Flood, C. H. (2014), Fast splittable pseudorandom number generators, *OOPSLA '14*, 453–472.
- Wolpert, D. H. and Macready, W. G. (1997), No free lunch theorems for optimization, *IEEE Trans. Evol. Comput.* 1, 67–82.

## 6. Open Questions

1. **A record's serialisation.** The genealogy of a run is a value; its format for the edges (JSON beside the API's) is not fixed.
2. **Evaluations that refine.** A multi-fidelity evaluation (a coarse simulation, then a finer one) would tell a ticket more than once. The protocol refuses that today; whether to admit it as a sequence of fidelities is open.
