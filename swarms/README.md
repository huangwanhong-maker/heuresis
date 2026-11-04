# swarms: Particle Swarms and Ant Colonies

*Status: draft (2026-11-05): particle swarms on boxes (the inertia and constriction rules, the Standard PSO 2011; global, ring and adaptive random informants) and ant colonies on cyclic sequences (the Ant System, MAX–MIN) ([decision 0004](../development/decisions/0004-the-first-modules.md)).* Ant colonies on other construction graphs, fully informed and comprehensive-learning swarms, and asynchronous swarms are planned.

## 1. Concepts

A particle swarm is a population with memory. Each particle has a position, a velocity and the best point it has found, and is pulled towards that point and towards the best one its informants have found. How it is pulled is a choice of symmetry:
- The inertia and constriction rules draw a random factor for each coordinate, so a swarm on a rotated problem behaves differently.
- SPSO 2011 draws a point uniformly in a ball about the centre of gravity of the particle and its two attractors, and so commutes with rotations and translations.

The checks hold both statements exactly.

SPSO 2011's invariance has a price. About a fixed attractor, a particle's second moments follow a linear map whose spectral radius at the standard parameters is below 1 in two dimensions and above it from three on. In ten dimensions the particles fly at the scale of the box, and the swarm progresses only through its personal bests. The inertia and constriction rules, at Clerc and Kennedy's χ = 0.7298, converge in ten dimensions.

An ant colony is a search distribution on permutations, built element by element from trails it learns and a heuristic the problem supplies (the domain's knowledge enters here). MAX–MIN keeps every trail within bounds set from the best tour so far, so that every edge keeps a probability of being chosen.

## 2. Content

| Header | Content |
|---|---|
| `particle_swarm.hpp` | `constriction(φ)`; `spso2011_move(x, v, p, l, same_best, w, c, direction, fraction)` and `inertia_move(…, r₁, r₂)`, the moves with their randomness as arguments; `ParticleSwarm(box, PsoOptions{swarm, PsoRule, PsoTopology, informants, w, c, inertia, c₁, c₂, φ}, Draws)` with `inject`, `emigrants`, personal bests, positions, velocities |
| `ant_colony.hpp` | `AntColony(n, η, AcoOptions{AcoVariant, ants, α, β, ρ, p_best, global_best}, Draws)`: tours as cyclic permutations, symmetric trails, `trails()`, `tau_min()`, `tau_max()` |
| `swarms.hpp` | the module's headers |

Both are generational: a generation's personal bests, topology and trails change once, after every evaluation is told, in ticket order.

## 3. Checks

`check_heuresis_swarms` (22):

- **Constriction:** χ(4.1) = 0.72984 against Clerc and Kennedy's 0.7298; χφ/2 against the customary 1.49618; the closed form for φ > 4.
- **Equivariance:**
  - SPSO 2011's move commutes with x ↦ Rx + b, the direction rotated with it, to 1.8 × 10⁻¹⁵ over 200 random states;
  - the inertia rule's move, with the same factors, does not (differences above 10⁻²).
- **SPSO 2011's second moments:** the spectral radius of their linear map is below 1 in 2-D and above it from 3-D on. The growth of E|x − p|² over 20,000 simulated particles matches it to 6 × 10⁻⁴ in 10-D and 6 × 10⁻³ in 2-D.
- **Convergence:**
  - SPSO 2011 reaches 10⁻¹⁰ on the 2-D sphere (3,640 evaluations), but in 10-D its particles still fly at the box's scale after 1,000 generations;
  - the inertia rule on a ring reaches 10⁻¹⁰ on the 10-D sphere (16,080 evaluations);
  - constriction with the whole swarm informing reaches 10⁻¹⁰ on the 5-D sphere;
  - SPSO 2011 finds Rastrigin's global minimum in 2-D;
  - confinement keeps every particle in the box;
  - a swarm told in reverse is the same swarm.
- **Ant colonies:**
  - MAX–MIN reaches the optimal 9-city tour found by enumerating all 8! orders. Every trail stays within [τ_min, τ_max], and both bounds equal Stützle and Hoos's expressions.
  - The Ant System reaches the optimal 7-city tour.
  - A colony told in reverse has the same trails, and every ant builds a permutation.

The program runs in about 1.2 s.

## 4. Dependencies

`search`, `variation`, `selection` (by the CMake target); `benchmarks` and calx `linalg` (`spectral_radius`) in the checks.

## 5. References

- Bonyadi, M. R. and Michalewicz, Z. (2014), SPSO 2011: analysis of stability; local convergence; and rotation sensitivity, *GECCO*.
- Clerc, M. (2012), Standard particle swarm optimisation, HAL hal-00764996.
- Clerc, M. and Kennedy, J. (2002), The particle swarm — explosion, stability, and convergence in a multidimensional complex space, *IEEE Trans. Evol. Comput.* 6, 58–73.
- Dorigo, M., Maniezzo, V. and Colorni, A. (1996), Ant system: optimization by a colony of cooperating agents, *IEEE Trans. Syst. Man Cybern. B* 26, 29–41.
- Kennedy, J. and Eberhart, R. (1995), Particle swarm optimization, *Proc. IEEE ICNN* 4, 1942–1948.
- Shi, Y. and Eberhart, R. (1998), A modified particle swarm optimizer, *IEEE CEC*, 69–73.
- Stützle, T. and Hoos, H. H. (2000), MAX–MIN Ant System, *Future Gener. Comput. Syst.* 16, 889–914.
- Zambrano-Bigiarini, M., Clerc, M. and Rojas, R. (2013), Standard particle swarm optimisation 2011 at CEC-2013: a baseline for future PSO improvements, *IEEE CEC*, 2337–2344.

## 6. Open Questions

1. **A stable invariant swarm.** A radial distribution in the ball (or parameters w, c depending on n) that keeps SPSO 2011's rotation invariance and settles in any dimension.
2. **Ant colonies beyond tours.** Construction graphs for assignments, schedules and paths, with pheromone on (position, element) as well as on (element, element).
