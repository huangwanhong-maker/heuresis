# multiobjective: Fronts, Hypervolumes and the Algorithms That Approach Them

*Status: draft (2026-11-05): the Pareto order and constrained dominance, fast non-dominated sorting, crowding, exact hypervolumes (sweeps in two and three objectives, WFG in any number, inclusion–exclusion for checks) and contributions, the indicators GD, IGD, IGD⁺, additive ε and spread, the simplex lattice and the scalarisations of decomposition, and NSGA-II, NSGA-III, SMS-EMOA and MOEA/D ([decision 0004](../development/decisions/0004-the-first-modules.md)).* Two-layer reference points for many objectives, R2 and hypervolume estimation by sampling for many objectives, MOEA/D-DE's operators, and preference-based methods are planned.

## 1. Concepts

With every objective minimised, a point a of ℝᵐ weakly dominates b when aᵢ ≤ bᵢ for every i; dominance is the strict part of this product order, a strict partial order. A finite population is a finite poset under it, and its minimal elements are its non-dominated set. Deb's constrained dominance extends the order to constraints and failures: a feasible point dominates every infeasible one, two infeasible points compare by their total violation, two feasible ones by dominance, and a failure is dominated by everything else. It is again a strict partial order, so all that follows applies to it.

The height of a point is the length of the longest chain of points that dominate it. The fronts are the level sets of the height. Each is an antichain, and front k is the minimal set of what remains when the fronts before it are removed. Deb's fast non-dominated sort computes them in O(mN²) by counting each point's dominators and peeling the points whose count falls to zero.

The hypervolume of a set A with respect to a reference point r is the measure of the union of the boxes [a, r]. It is strictly Pareto compliant: if A dominates B, then HV(A) > HV(B). This is why maximising it drives a set to the front. Four exact algorithms are given and checked against one another:
- **two objectives:** a sweep in f₁, O(n log n);
- **three objectives:** Beume et al.'s sweep in f₃, keeping the staircase of the (f₁, f₂) projections in a balanced tree with its area, O(n log n);
- **any number of objectives:** WFG's exclusive volumes of limit sets;
- **for checks:** inclusion–exclusion, which is the definition.

A point's exclusive contribution is what the set loses without it. In two objectives this is the rectangle up to its neighbours on the staircase.

The four algorithms differ in what they add to the order of fronts:
- **NSGA-II** cuts the last front by crowding distance.
- **NSGA-III** spreads the population among reference directions after normalising by the ideal point and the intercepts of the extreme points. Crowding fails in many objectives, where nearly everything is non-dominated.
- **SMS-EMOA** removes, one at a time, the point of the last front that contributes the least hypervolume, so the population's hypervolume never falls.
- **MOEA/D** turns the front into scalar problems, one per weight (by Tchebycheff, PBI or a weighted sum), solved together by neighbouring weights.

All four follow the ask–tell protocol:
- NSGA-II, NSGA-III and MOEA/D are generational, so their states do not depend on the order in which evaluations return. MOEA/D is the batch form, breeding a generation from the solutions at its start and applying the children in subproblem order.
- SMS-EMOA is asynchronous, offering at most `breadth` offspring per ask (one by default).

## 2. Content

| Header | Content |
|---|---|
| `dominance.hpp` | `weakly_dominates`, `dominates`, `constrained_dominates`; `non_dominated_sort` (for any strict partial order on indices, for objective vectors, for evaluations) → `Fronts{fronts, rank}`; `non_dominated` |
| `hypervolume.hpp` | `hypervolume_2d`, `hypervolume_3d` (Beume et al. 2009), `hypervolume_wfg` (While et al. 2012), `hypervolume_inclusion_exclusion`, `hypervolume` (the fastest for the dimension), `contributions` (closed form in two objectives, differences otherwise) |
| `indicators.hpp` | `crowding_distance`; `generational_distance`, `inverted_generational_distance`, `inverted_generational_distance_plus`; `additive_epsilon`; `spread` (Deb's Δ) |
| `decomposition.hpp` | `das_dennis(m, p)`; `Scalarisation{tchebycheff, pbi, weighted_sum}`, `scalarise` |
| `kernels.hpp` | `Kernels<G>{initialise, crossover, mutate}`; `real_kernels(box, η_c, η_m, p_c, rate)`: bounded SBX and polynomial mutation |
| `nsga2.hpp` | `objective_points`; `rank_and_crowding` (NSGA-II's environmental selection); `Nsga2<G>(kernels, μ, draws)`: `ask`, `tell`, `inject`, `emigrants`, `population`, `ranks`, `front` |
| `nsga3.hpp` | `nsga3_normalise`, `associate`; `Nsga3<G>(kernels, references, μ, draws)` |
| `sms_emoa.hpp` | `SmsEmoa<G>(kernels, μ, draws, breadth)`, steady state |
| `moead.hpp` | `Moead<G>(kernels, weights, Options{neighbours, replacements, scalarisation, theta}, draws)`: `solutions`, `ideal`, `front` |
| `multiobjective.hpp` | all of the above |

## 3. Checks

`check_heuresis_multiobjective` (44 checks, about 7 s):
- dominance is irreflexive, asymmetric and transitive; constrained dominance follows Deb's rules;
- fast non-dominated sorting gives every point its height in the dominance poset (memoised recursion over dominators, 60 random sets in 2–4 objectives, with ties); each front is an antichain, and each point of front k + 1 is dominated by one of front k;
- the two-objective sweep, Beume's three-objective sweep and WFG agree with inclusion–exclusion on 120 random sets each in 2, 3, 4 and 5 objectives (to 5 × 10⁻¹⁴, the cancellation of inclusion–exclusion);
- the hypervolume of evenly spaced points on ZDT1's and ZDT2's fronts converges to 2/3 and 1/3 at first order, from below. A grid of angles on DTLZ2's front converges to 1 − π/6 at first order in its step;
- strict monotonicity: improving a non-dominated point increases the hypervolume, and a dominated point adds nothing. Contributions equal HV(A) − HV(A ∖ {a}), and a point with a twin contributes nothing;
- IGD of the reference set is 0; the additive ε of a front against its translate by δ is −δ and +δ the other way; I_ε+(A, B) ≤ 0 exactly when A weakly dominates B; IGD⁺ does not increase when a set improves; Δ of an even set reaching the extremes is 0; crowding of an even line is 4h inside and ∞ at the ends; the simplex lattice has C(14, 2) = 91 points on the simplex for m = 3, p = 12;
- algorithms against their fronts:

  | Algorithm | Problem | Budget | Result | Required |
  |---|---|---|---|---|
  | NSGA-II | ZDT1 | 20 000 evaluations | HV 1.4% below 2/3 | within 1.5% |
  | SMS-EMOA | ZDT1 | 20 000 evaluations | HV 0.8% below 2/3 | within 1.5% |
  | MOEA/D (Tchebycheff) | ZDT1 | 20 000 evaluations | HV 1.1% below 2/3 | within 2% |
  | NSGA-III | DTLZ2 (3 objectives) | 250 generations | IGD 1.9 × 10⁻³ | below 0.01 |
  | NSGA-III | DTLZ1 (3 objectives) | 400 generations | IGD 2.6 × 10⁻³ | below 0.02 |
  | MOEA/D (PBI) | DTLZ2 (3 objectives) | 250 generations | IGD 6.6 × 10⁻⁴ | below 0.01 |

  IGD is measured to the intersections of the reference lines with the fronts. At 100 points, the best two-objective configuration still lies about 0.75% below 2/3;
- placement does not change a run:
  - NSGA-II's population is the same serially, on a pool of four threads, on three simulated workers with random durations, and told in reverse;
  - MOEA/D's solutions are the same with its generations cut into parts on a pool.

## 4. Dependencies

heuresis `search` (evaluations, the protocol, batches, draws), `variation` (SBX and polynomial mutation), `selection` (tournaments); calx `linalg` (NSGA-III's intercepts, by `solve`). The check uses `benchmarks` (ZDT1, ZDT2, DTLZ1, DTLZ2 and their exact hypervolumes).

## 5. References

- Auger, A., Bader, J., Brockhoff, D. and Zitzler, E. (2009), Theory of the hypervolume indicator: optimal μ-distributions and the choice of the reference point, *FOGA*, 87–102.
- Beume, N., Fonseca, C. M., López-Ibáñez, M., Paquete, L. and Vahrenhold, J. (2009), On the complexity of computing the hypervolume indicator, *IEEE Trans. Evol. Comput.* 13, 1075–1082.
- Beume, N., Naujoks, B. and Emmerich, M. (2007), SMS-EMOA: multiobjective selection based on dominated hypervolume, *Eur. J. Oper. Res.* 181, 1653–1669.
- Das, I. and Dennis, J. E. (1998), Normal-boundary intersection, *SIAM J. Optim.* 8, 631–657.
- Deb, K. and Jain, H. (2014), An evolutionary many-objective optimization algorithm using reference-point-based nondominated sorting approach, part I, *IEEE Trans. Evol. Comput.* 18, 577–601.
- Deb, K., Pratap, A., Agarwal, S. and Meyarivan, T. (2002), A fast and elitist multiobjective genetic algorithm: NSGA-II, *IEEE Trans. Evol. Comput.* 6, 182–197.
- Ishibuchi, H., Masuda, H., Tanigaki, Y. and Nojima, Y. (2015), Modified distance calculation in generational distance and inverted generational distance, *EMO*, LNCS 9019, 110–125.
- Jain, H. and Deb, K. (2014), part II: handling constraints and extending to an adaptive approach, *IEEE Trans. Evol. Comput.* 18, 602–622.
- Li, H. and Zhang, Q. (2009), Multiobjective optimization problems with complicated Pareto sets, MOEA/D and NSGA-II, *IEEE Trans. Evol. Comput.* 13, 284–302.
- Miettinen, K. (1999), *Nonlinear Multiobjective Optimization*, Kluwer.
- While, L., Bradstreet, L. and Barone, L. (2012), A fast way of calculating exact hypervolumes, *IEEE Trans. Evol. Comput.* 16, 86–95.
- Zhang, Q. and Li, H. (2007), MOEA/D: a multiobjective evolutionary algorithm based on decomposition, *IEEE Trans. Evol. Comput.* 11, 712–731.
- Zitzler, E. and Thiele, L. (1998), Multiobjective optimization using evolutionary algorithms — a comparative case study, *PPSN V*, 292–301.
- Zitzler, E., Thiele, L., Laumanns, M., Fonseca, C. M. and Grunert da Fonseca, V. (2003), Performance assessment of multiobjective optimizers: an analysis and review, *IEEE Trans. Evol. Comput.* 7, 117–132.

## 6. Open Questions

1. **Exact contributions in three objectives.** They are computed as differences, at O(n² log n). The incremental algorithm (Guerreiro and Fonseca's HV3D⁺) would make SMS-EMOA practical beyond a few thousand evaluations in three objectives.
2. **Many objectives.** WFG is exponential in m in the worst case. Beyond about eight objectives, estimation by sampling (HypE) or R2 may serve better; which to add first is open.
3. **MOEA/D's variants.** A batch generation departs from the original's one child at a time. Whether to offer an asynchronous MOEA/D as well, and MOEA/D-DE's operators, is open.
4. **Reference fixtures.** The checks rest on closed forms and on the algorithms' agreement. Fixtures of hypervolumes from moocore or pymoo, made by a script, would add an outside reference.
