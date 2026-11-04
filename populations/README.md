# populations: Structured and Distributed Populations

*Status: draft (2026-11-05): topologies, island models synchronous and asynchronous, cellular algorithms and their takeover times ([decision 0004](../development/decisions/0004-the-first-modules.md)).* Transports between real machines are the infrastructure's edges'.

## 1. Concepts

A population need not be one pool. In an island model, several searches run side by side and every few cycles send their best along the edges of a topology; islands keep diversity apart and share what they find, and the topology and interval set how fast a discovery spreads. Each island is any Migratory algorithm, and a migrant is a plain value (genotype, evaluation, provenance) whose island is written in its ticket's top bits, so it can be carried and serialised by whatever joins the machines (decision 0003 §2.9).

Two drivers keep the core pure. The synchronous one runs all islands through an epoch, then migrates in the order of the sending island; the islands of an epoch are independent, so they may run on a thread pool, and the run is the same either way. The asynchronous one is a discrete-event simulation of islands on machines of their own: a cycle lasts the sum of its evaluations' durations, migrants arrive after a latency and are injected when the receiver next begins a cycle. Events at one instant are taken completions first, then arrivals, then beginnings, so that with equal durations and no latency the asynchronous run is exactly the synchronous one; with random durations and latencies it is reproducible from the seed, and it reports the simulated wall time.

In a cellular algorithm each vertex holds one individual that breeds with its neighbours and is replaced when its offspring is not worse. With selection alone and each cell keeping the best of its closed neighbourhood, cell i holds after g generations the best of the ball of radius g about it, so one best individual at s fills the graph in exactly ecc(s) generations, its eccentricity: the takeover time, the measure of selection pressure on a structure.

## 2. Content

| Header | Content |
|---|---|
| `topology.hpp` | `Topology{out}`: `isolated`, `ring(k, bidirectional)`, `torus(rows, cols, moore)`, `complete`, `star`, `hypercube(d)` |
| `islands.hpp` | `Migration{interval, count}`; `IslandOutcome<G>`; `run_islands(islands, topology, migration, f, budget, pool)`; `IslandClock{duration, latency, seed}`; `run_islands_asynchronous(islands, topology, migration, f, budget, clock)` |
| `cellular.hpp` | `CellSelection{best, tournament}`; `CellularEA<G>(initial, neighbourhoods, selection, mutate, crossover, p_c, draws)`, `cells()` |
| `populations.hpp` | everything above |

## 3. Checks

`check_heuresis_populations` (17):
- the topologies' degrees;
- takeover times equal to the start's eccentricity, found by breadth-first search and in closed form: a ring of 20 (10), tori 8 × 8 with von Neumann's (8) and Moore's (4) neighbourhoods, a 6 × 9 torus (7), a hypercube of dimension 5 (5), a complete graph of 16 (1);
- four GA islands on a ring: the same final populations, ticket for ticket, on one thread and on a pool of four; the budget of 4000 shared exactly; migrants of other islands present;
- the asynchronous model with unit durations and no latency equal to the synchronous run ticket for ticket, its simulated time 1000; with random durations and latencies reproducible and slower;
- six (1+1) EA islands on a complete graph solving OneMax with 50 bits.

## 4. Dependencies

`search`, `variation`, `selection`; calx `scheduler` (through `search`); in the checks, `genetic` and `benchmarks`.

## 5. References

- Alba, E. and Dorronsoro, B. (2008), *Cellular Genetic Algorithms*, Springer.
- Alba, E. and Tomassini, M. (2002), Parallelism and evolutionary algorithms, *IEEE Trans. Evol. Comput.* 6, 443–462.
- Cantú-Paz, E. (2000), *Efficient and Accurate Parallel Genetic Algorithms*, Kluwer.
- Giacobini, M., Tomassini, M. and Tettamanzi, A. (2005), Takeover time curves in random and small-world structured populations, *GECCO 2005*.
- Manderick, B. and Spiessens, P. (1989), Fine-grained parallel genetic algorithms, *ICGA 3*, 428–433.
- Sarma, J. and De Jong, K. (1996), An analysis of the effects of neighborhood size and shape on local selection algorithms, *PPSN IV*, 236–244.
- Skolicki, Z. and De Jong, K. (2005), The influence of migration sizes and intervals on island models, *GECCO*, 1295–1302.
- Tanese, R. (1989), Distributed genetic algorithms, *ICGA 3*, 434–439.
- Whitley, D., Rana, S. and Heckendorn, R. B. (1999), The island model genetic algorithm: on separability, population size and convergence, *J. Comput. Inf. Technol.* 7, 33–47.

## 6. Open Questions

1. Heterogeneous archipelagos (islands of different algorithms) need one genotype; a common representation for mixed islands, and migration between representations, are open.
2. Migration policies beyond best-out, worst-in (random emigrants, replacement by tournament), and topologies that change in time.
3. Takeover curves (not only times) under tournaments, against the logistic and polynomial models of the literature.
4. The topologies' breadth-first distances in the checks would belong to morphe's `combinatorics` if the library itself needed them.
