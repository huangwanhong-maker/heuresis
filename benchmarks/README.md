# benchmarks: Problems with Known Optima

*Status: draft (2026-11-05): pseudo-Boolean, continuous, constrained, multi-objective, permutation and repertoire problems ([decision 0004](../development/decisions/0004-the-first-modules.md)).*

## 1. Concepts

An algorithm can only be checked on a problem whose answer is known: its optimum, its front, the expected time to reach it. These benchmarks are such problems. They name no object in nature: they are the functions on which the theory of evolutionary computation states its results, and on which every algorithm of the library is held to that theory.

**Minimisation.** Every benchmark is a minimisation whose optimum is 0 unless the header says otherwise. The classical maximised forms (OneMax as the number of ones) are a constant minus these.

**Instances.** Random instances (NK landscapes, QUBOs, planted SAT, rotations, tours) are drawn from streams keyed by (seed, instance), so an instance is the same wherever it is made.

**The fronts of ZDT and DTLZ.** Their fronts and hypervolumes are known in closed form. With respect to (1, …, 1), the region a front dominates in the unit box is:
- the area above a curve for ZDT;
- the box less a simplex for DTLZ1, giving 1 − 2⁻ᵐ/m!;
- the box less the positive part of a ball for DTLZ2–4, giving 1 − π^{m/2}/(2ᵐ Γ(m/2 + 1)).

These are the targets the hypervolume algorithms converge to.

## 2. Content

| Header | Content |
|---|---|
| `pseudo_boolean.hpp` | `onemax`, `leading_ones`, `jump`, `trap`; `NK` (Kauffman); `Qubo` (with `flip_delta`); `PlantedSat` |
| `continuous.hpp` | `sphere`, `ellipsoid`, `rosenbrock`, `rastrigin`, `ackley`, `griewank`, `schwefel`, `styblinski_tang`, `levy`, `branin`, `six_hump_camel`, `hartmann3`, `hartmann6`, with their minima in the header's table; constrained `disc_rosenbrock`, `g06`; `Rotation` (Haar, by QR of a Gaussian matrix), `transformed` (shift and rotation) |
| `multiobjective.hpp` | `zdt1`, `zdt2`, `zdt3`, `zdt4`, `zdt6`; `dtlz1`, `dtlz2`, `dtlz3`, `dtlz4`, `dtlz7`; `zdt6_least_f1`; the exact hypervolumes `zdt1_hypervolume`, `zdt2_hypervolume`, `zdt6_hypervolume`, `dtlz1_hypervolume`, `dtlz2_hypervolume`; front points |
| `combinatorial.hpp` | `Tour` (Euclidean, random instances); `planar_arm` (a repertoire problem: descriptor the tip, objective the variance of the angles) |
| `benchmarks.hpp` | all of the above |

## 3. Checks

`check_heuresis_benchmarks` (59):
- Every continuous function's f(x*) = f*, with no point among 20 000 random ones below f*. Schwefel's constant, G06's optimum and its feasibility, and the disc constraint.
- A rotation orthogonal to 10⁻¹⁴. A shifted, rotated ellipsoid least at its shift. E[R₀₀²] = 1/n under the Haar measure.
- ZDT1, ZDT2, DTLZ1, DTLZ2 and DTLZ4 on their fronts at the optimal distance variables.
- The closed-form hypervolumes against Monte Carlo (4 × 10⁵ points each), and ZDT6's least f₁.
- The pseudo-Boolean optima and valleys. NK at k = 0 separable, its bitwise optimum the global one by enumeration. NK's neighbourhoods. QUBO's flip deltas against differences. The planted assignment satisfying every clause.
- Tours of a square, and the straight arm.

## 4. Dependencies

`variation` (bit strings, permutations); `search` (evaluations, streams); calx `linalg` (QR, for rotations).

## 5. References

- Applegate, D. L., Bixby, R. E., Chvátal, V. and Cook, W. J. (2006), *The Traveling Salesman Problem: A Computational Study*, Princeton.
- Cully, A., Clune, J., Tarapore, D. and Mouret, J.-B. (2015), Robots that can adapt like animals, *Nature* 521, 503–507.
- Deb, K. and Goldberg, D. E. (1993), Analyzing deception in trap functions, *FOGA 2*, 93–108.
- Deb, K., Thiele, L., Laumanns, M. and Zitzler, E. (2005), Scalable test problems for evolutionary multiobjective optimization, in *Evolutionary Multiobjective Optimization*, Springer, 105–145.
- Doerr, B. and Neumann, F., eds. (2020), *Theory of Evolutionary Computation*, Springer.
- Droste, S., Jansen, T. and Wegener, I. (2002), On the analysis of the (1+1) evolutionary algorithm, *Theor. Comput. Sci.* 276, 51–81.
- Floudas, C. A. and Pardalos, P. M. (1990), *A Collection of Test Problems for Constrained Global Optimization Algorithms*, LNCS 455, Springer.
- Hansen, N., Finck, S., Ros, R. and Auger, A. (2009), Real-parameter black-box optimization benchmarking 2009: noiseless functions definitions, INRIA RR-6829.
- Jamil, M. and Yang, X.-S. (2013), A literature survey of benchmark functions for global optimisation problems, *Int. J. Math. Model. Numer. Optim.* 4, 150–194.
- Kauffman, S. A. and Levin, S. (1987), Towards a general theory of adaptive walks on rugged landscapes, *J. Theor. Biol.* 128, 11–45.
- Lucas, A. (2014), Ising formulations of many NP problems, *Front. Phys.* 2, 5.
- Mezzadri, F. (2007), How to generate random matrices from the classical compact groups, *Notices AMS* 54, 592–604.
- Surjanovic, S. and Bingham, D., *Virtual Library of Simulation Experiments: Test Functions and Datasets*, Simon Fraser University (the constants of Branin, the six-hump camel and Hartmann's functions).
- Vassiliades, V., Chatzilygeroudis, K. and Mouret, J.-B. (2018), Using centroidal Voronoi tessellations to scale up the multidimensional archive of phenotypic elites algorithm, *IEEE Trans. Evol. Comput.* 22, 623–630.
- Zitzler, E., Deb, K. and Thiele, L. (2000), Comparison of multiobjective evolutionary algorithms: empirical results, *Evol. Comput.* 8, 173–195.

## 6. Open Questions

1. **BBOB's 24 functions** with COCO's own instances. Their transformations (T_osz, T_asy, conditioning) can be written from RR-6829; matching COCO's instances needs fixtures from its Python package (`cocoex`).
2. **Noisy and expensive benchmarks**, for surrogate and noisy optimisation.
