# local: Local Search

*Status: draft (2026-11-05): neighbourhoods, hill climbing, iterated local search, tabu search, variable neighbourhood descent, 2-opt, memetic refinement ([decision 0004](../development/decisions/0004-the-first-modules.md)).* Randomised local search is the (1+1) EA with one bit flipped, in `genetic`.

## 1. Concepts

A neighbourhood turns a space into a graph, and a local optimum is a vertex no neighbour improves; which graph is chosen decides which points are local optima. A hill climber moves to an improving neighbour until none is left: best improvement evaluates the whole neighbourhood as one generation, first improvement scans it in a fresh random order a chunk at a time. Both stop at a local optimum, and both are generational in the protocol's sense, so their moves do not depend on the order in which a chunk's evaluations return.

Three ways past a local optimum. Iterated local search walks on the local optima themselves: kick the incumbent harder than one step (for tours, the double bridge, which improving 2-opt moves cannot undo), climb again, keep the result when it is not worse. Tabu search always moves to the best admissible neighbour, even uphill, and forbids undoing its recent moves for a tenure, unless a move would beat everything seen (aspiration). Variable neighbourhood descent climbs through a sequence of neighbourhoods, returning to the first after each improvement, and stops only where every neighbourhood is at an optimum.

Where an objective's structure is known, local search need not be black-box: 2-opt on tours computes a move's change of length from four distances, so it is a function used inside an evaluation, not an ask–tell algorithm. The same holds for calx's Nelder–Mead and BFGS, which own their loops; a memetic algorithm uses them either as a Baldwinian objective (the refined value is the score, its evaluations the cost) or as a Lamarckian step whose refined point is injected.

## 2. Content

| Header | Content |
|---|---|
| `neighbourhood.hpp` | `Neighbourhood<G>{size, apply, attribute}`; `one_flip()`, `two_flip()` on bit strings; `two_opt()`, `exchanges()`, `insertions()` on permutations |
| `climbing.hpp` | `Pivot{best, first}`; `HillClimber<G>(start, neighbourhood, pivot, chunk, draws)` with `restart(start, origin)`, `current()`, `steps()`, `evaluations()` |
| `iterated.hpp` | `Acceptance{better, not_worse}`; `IteratedLocalSearch<G>(start, neighbourhood, kick, pivot, acceptance, draws)`, `best()`, `kicks()`; `double_bridge(tour, stream)` |
| `tabu.hpp` | `TabuSearch<G>(start, neighbourhood, tenure, draws)`, `current()`, `best()` |
| `descent.hpp` | `VariableNeighbourhoodDescent<G>(start, neighbourhoods, draws)` |
| `tours.hpp` | `two_opt_descent(tour, distance, epsilon)`: first-improvement 2-opt with O(1) move evaluation |
| `memetic.hpp` | `Refiner{nelder_mead, bfgs}`, `RefineOptions`, `refine(f, x, options)` → `Refined{x, value, evaluations}`; `baldwinian(f, options)`; `lamarckian(f, individual, ticket, options)`; `refine_tag` |
| `local.hpp` | everything above |

## 3. Checks

`check_heuresis_local` (15):
- best improvement on OneMax from a point with z zeros: z moves and exactly 1 + n(z + 1) evaluations, for 20 random starts;
- first improvement solving OneMax in z moves, the same climb when every chunk is told in reverse;
- hill climbers (both pivots) stopping at one-flip local optima of a QUBO in 14 variables, as enumeration of its 2¹⁴ points finds them;
- tabu search reaching a 16-variable QUBO's enumerated optimum in 5000 evaluations, and satisfying a planted 3-SAT with 40 variables and 160 clauses;
- VND with one and two flips stopping at points optimal for both, by enumeration;
- iterated local search (2-opt, double bridge) finding the optimal tour of nine random points, as enumeration of 8! tours finds it;
- a 2-opt local optimum of 80 points with no two crossing edges (and less than half the length of the starting tour);
- Baldwinian Nelder–Mead on Rosenbrock from (−1.2, 1) to 10⁻¹⁸, its cost the number of calls; Lamarckian BFGS reaching (1, 1) within 2 × 10⁻⁹, its origin the original's ticket.

## 4. Dependencies

`search`, `variation`; calx `optimisation` (Nelder–Mead, BFGS); in the checks, `benchmarks`.

## 5. References

- Croes, G. A. (1958), A method for solving traveling-salesman problems, *Oper. Res.* 6, 791–812.
- Glover, F. (1989), Tabu search — Part I, *ORSA J. Comput.* 1, 190–206; (1990), Part II, *ORSA J. Comput.* 2, 4–32.
- Hansen, P. and Mladenović, N. (2001), Variable neighborhood search: principles and applications, *Eur. J. Oper. Res.* 130, 449–467.
- Hansen, P. and Mladenović, N. (2006), First vs. best improvement: an empirical study, *Discrete Appl. Math.* 154, 802–817.
- Hinton, G. E. and Nowlan, S. J. (1987), How learning can guide evolution, *Complex Systems* 1, 495–502.
- Lin, S. (1965), Computer solutions of the traveling salesman problem, *Bell Syst. Tech. J.* 44, 2245–2269.
- Lourenço, H. R., Martin, O. C. and Stützle, T. (2003), Iterated local search, in *Handbook of Metaheuristics*, Kluwer, 321–353.
- Martin, O., Otto, S. W. and Felten, E. W. (1991), Large-step Markov chains for the traveling salesman problem, *Complex Systems* 5, 299–326.
- Mladenović, N. and Hansen, P. (1997), Variable neighborhood search, *Comput. Oper. Res.* 24, 1097–1100.
- Moscato, P. (1989), On evolution, search, optimization, genetic algorithms and martial arts: towards memetic algorithms, Caltech Concurrent Computation Program report 826.
- Papadimitriou, C. H. and Steiglitz, K. (1982), *Combinatorial Optimization: Algorithms and Complexity*, Prentice-Hall, ch. 19.

## 6. Open Questions

1. The 2-opt neighbourhood as listed includes moves that leave the cycle unchanged (segments of length n − 1); harmless to improvement, wasteful to tabu search. A neighbourhood of the n(n − 3)/2 distinct 2-opt moves is planned.
2. Or-opt and 3-opt, and Lin–Kernighan, for tours; delta evaluation for QUBOs (benchmarks' `flip_delta`) behind a white-box one-flip climber.
3. Variable neighbourhood search with shaking (beyond descent), and guided local search's penalties.
