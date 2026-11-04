# diversity: Repertoires, Novelty and Niches

*Status: draft (2026-11-05): archives of elites over grids and Voronoi cells with their coverage and QD-score, MAP-Elites and CVT-MAP-Elites, novelty search with its adaptive archive, fitness sharing and clearing ([decision 0004](../development/decisions/0004-the-first-modules.md)).* Centroidal Voronoi sites by k-means (waiting for tyche's k-means), CMA-ME and gradient-assisted quality-diversity, unstructured archives, multi-objective MAP-Elites and deterministic crowding are planned.

## 1. Concepts

The search for quality and diversity does not ask for the best solution. It asks for the best solution of every kind, a repertoire, which says what is possible and where. Each evaluation places its point in a space of behaviours by its descriptors. A tessellation cuts that space into cells:
- a grid on a box;
- Voronoi cells of given sites, each point going to its nearest site.

An archive is a function from cells to elites. Offering an individual changes the archive only where the individual is better (by Deb's rule, so constraints and failures are handled as everywhere in heuresis) than the elite of its cell. So an archive's coverage (the share of cells filled) and its QD-score (Σ over filled feasible cells of offset − f) never decrease.

MAP-Elites breeds every candidate from elites drawn uniformly from the archive: a mutation of one, or a two-parent variation (iso+line on real vectors) followed by a mutation. Either operator suffices alone; with a variation and no mutation every child is a variation, as with an identity mutation at p_v = 1, run for run. With Voronoi cells it is CVT-MAP-Elites. Vassiliades et al. take the sites as the centroids of k-means on samples of the behaviour space, so that the cells have nearly equal volume in any dimension. k-means belongs to tyche and is not yet written, so the caller supplies the sites. Any sites give a valid tessellation.

Novelty search rewards being unlike what has been seen. A behaviour's sparseness is its mean distance to its k nearest neighbours among the population and an archive of past behaviours. Selection is by sparseness alone, and a behaviour joins the archive when its sparseness exceeds a threshold that adapts to the rate of additions. On deceptive problems, where the objective's gradient leads away from the goal, it is not misled.

Niching keeps several optima in one population:
- **Fitness sharing** divides fitness by a niche count, so that peaks hold shares of the population proportional to their heights.
- **Clearing** keeps the κ best of each niche and clears the rest.

All three algorithms follow the ask–tell protocol:
- **MAP-Elites** is asynchronous: it breeds from the archive as it stands, at most `breadth` candidates per ask, and is reproducible from the order of arrival.
- **Novelty search** is generational: it computes novelty for parents and offspring together and extends the archive in ticket order.

## 2. Content

| Header | Content |
|---|---|
| `archive.hpp` | `Grid(lower, upper, resolution)`: `cell`, `centre`; `Voronoi(sites)`: `cell`; `Archive<G, Tessellation>`: `offer`, `coverage`, `qd_score(offset)`, `filled`, `elite` |
| `map_elites.hpp` | `Kernels<G>{initialise, mutate, vary}`, at least one of `mutate` and `vary`; `MapElites<G, Tessellation>(kernels, tessellation, Options{initial, variation_probability, breadth}, draws)`: `ask`, `tell`, `inject`, `emigrants`, `archive` |
| `novelty.hpp` | `sparseness(b, others, k)`; `NoveltySearch<G>(kernels, Options{population, neighbours, threshold, crowded, quiet, tournament, variation_probability}, draws)`: `population`, `novelty`, `archive`, `threshold` |
| `niching.hpp` | `sharing_function`, `shared_fitness(fitness, distance, σ, α)`; `clearing(values, distance, σ, κ)` |
| `diversity.hpp` | all of the above |

## 3. Checks

`check_heuresis_diversity` (20 checks, under a second):
- a grid maps each cell's centre to the cell and points outside its box to its boundary cells; a Voronoi tessellation maps each site to its own cell;
- an archive fills a cell, refuses a worse individual and a failure, and takes a better one; over 2000 random offers its coverage and QD-score never decrease;
- MAP-Elites on the planar arm of eight links (30 000 evaluations, a 20 × 20 grid):
  - it fills 97.7% of the cells meeting the reachable disc and none outside it;
  - its QD-score never decreases over the run;
  - its elites at the edge of the disc are nearly straight (angle variance 10⁻³);
- CVT-MAP-Elites fills all 200 Voronoi cells of sites inside the disc (20 000 evaluations);
- novelty search (10 000 evaluations) visits 95% of the reachable cells, against 62% for random search with the same budget, and its archive only grows;
- MAP-Elites and novelty search with a variation alone give, bit for bit, the runs of the variation at p_v = 1 with an identity mutation, every bred candidate of two parents; kernels with neither operator are refused;
- sharing equalises the shared fitness when peaks of fitness 1 and ½ hold 20 and 10 individuals; clearing keeps exactly the κ = 2 best of each niche.

## 4. Dependencies

heuresis `search` (evaluations, the protocol, batches, pending tables, draws), `selection` (tournaments on novelty). The check uses `variation` (Gaussian mutation, iso+line, reflection) and `benchmarks` (the planar arm).

## 5. References

- Goldberg, D. E. and Richardson, J. (1987), Genetic algorithms with sharing for multimodal function optimization, *ICGA 2*, 41–49.
- Lehman, J. and Stanley, K. O. (2011), Abandoning objectives: evolution through the search for novelty alone, *Evol. Comput.* 19, 189–223.
- Mouret, J.-B. and Clune, J. (2015), Illuminating search spaces by mapping elites, arXiv:1504.04909.
- Pétrowski, A. (1996), A clearing procedure as a niching method for genetic algorithms, *IEEE ICEC*, 798–803.
- Pugh, J. K., Soros, L. B. and Stanley, K. O. (2016), Quality diversity: a new frontier for evolutionary computation, *Front. Robot. AI* 3, 40.
- Vassiliades, V., Chatzilygeroudis, K. and Mouret, J.-B. (2018), Using centroidal Voronoi tessellations to scale up the multidimensional archive of phenotypic elites algorithm, *IEEE Trans. Evol. Comput.* 22, 623–630.
- Vassiliades, V. and Mouret, J.-B. (2018), Discovering the elite hypervolume by leveraging interspecies correlation, *GECCO*, 149–156.

## 6. Open Questions

1. **Centroidal sites.** Taking them from k-means on samples needs tyche's k-means, and quasi-random samples need a low-discrepancy sequence in calx. Both are requested of those libraries.
2. **Nearest sites at scale.** The Voronoi cell is found by a scan, which serves up to a few thousand sites. A k-d tree would serve many more.
3. **Novelty with the objective.** Novelty is pure here. Mixing it with the objective (by a weighted sum, by Pareto ranks, or by local competition) is planned with the first problem that needs it.
4. **Higher-dimensional behaviours.** Grids grow exponentially with the dimension of the behaviour space. Beyond about four dimensions, Voronoi cells or unstructured archives should be preferred.
