# strategies: Evolution Strategies and Differential Evolution

*Status: draft (2026-11-05): the (1+1)-ES with the one-fifth rule; CMA-ES (Hansen 2016), active and passive, separable, bounded by a box transformation or by repair, with injection and the tutorial's termination criteria; IPOP and BIPOP restarts; differential evolution (rand/1, best/1, current-to-best/1, rand/2; binomial and exponential crossover) and SHADE ([decision 0004](../development/decisions/0004-the-first-modules.md)).* Natural evolution strategies, LM-MA-ES and other large-scale variants, and asynchronous differential evolution are planned.

## 1. Concepts

An evolution strategy searches ℝⁿ with a distribution it adapts. The (1+1)-ES keeps one point and one step size: a Gaussian step replaces the point when it is no worse, and the step size grows after a success and shrinks after a failure so that about one step in five succeeds (Rechenberg's rule). On the sphere this converges linearly, and σ follows the distance to the optimum.

CMA-ES samples λ points from N(m, σ²C), ranks them, and moves m towards the best, C towards the directions in which the best steps lay (the evolution path p_c for successive steps, the rank-μ update for the steps of one generation, and with active weights away from the worst), and σ by the length of a second path p_σ, whitened by C^{−1/2}: longer than expected under random selection means consecutive steps agree and σ should grow, shorter means they cancel. The update sees the objective only through the ranking, so the run is the same on g ∘ f for any increasing g. Because p_σ is whitened by the symmetric root of C, the update commutes with every map x ↦ αRx + b (R orthogonal), the state mapped with it; with σ held fixed it commutes with every affine map. For general linear maps, CSA is invariant only up to the rotations by which C^{−1/2} differs between generations, and the checks show where each invariance holds.

On a box, sampling ℝⁿ and mapping each sample onto the box by Hansen's linear–quadratic transformation makes an optimum on the boundary an interior optimum of the transformed problem. Repairing samples by reflection and telling the repaired points, by contrast, lets σ grow without bound when the optimum is a corner: the checks show σ above 10σ₀.

Restarts with growing populations make CMA-ES global on multimodal functions with global structure: IPOP doubles λ at each restart, and BIPOP interleaves it with short runs of small populations and small step sizes, each regime running when it has spent less.

Differential evolution moves a population by differences of its members, so its steps scale and orient themselves with the population. SHADE adapts its scale factor F and crossover rate CR from a memory of the values that succeeded.

## 2. Content

| Header | Content |
|---|---|
| `cma_es.hpp` | `CmaParameters::tutorial(n, λ, active)` (the tutorial's Table 1), `.separable()` (Ros and Hansen 2008), `.eigen_gap()`; `CmaState` (m, σ, C, p_c, p_σ, B, D, C^{−1/2}, the lazy decomposition); `cma_update(parameters, state, ranked)`, one generation's transition as a pure function of the ranked points; `Cma(mean, σ, CmaOptions, Draws)`, the ask–tell algorithm with `CmaBounds::{transform, repair}`, `inject`, `emigrants`, `mean_point`, `stop_reason()` (`CmaStop`: generations, TolX, TolFun, ConditionCov, NoEffectAxis, NoEffectCoord, Stagnation, TolUpSigma) |
| `restarts.hpp` | `CmaRestarts(RestartOptions{ipop or bipop, start box, σ₀, increase, max_large_runs, cma options}, Draws)`: restarts with tickets continuing and each run's streams keyed by its number |
| `one_plus_one.hpp` | `OnePlusOneEs(x₀, σ, OnePlusOneOptions{target_success, damping, bounds, repair, min_sigma}, Draws)`: the one-fifth rule in Kern et al.'s smooth form |
| `differential_evolution.hpp` | `DifferentialEvolution(box, DeOptions{population, F, CR, DeMutation, DeCrossover, midpoint or repair}, Draws)`; `Shade(box, ShadeOptions{population, memory, p_max}, Draws)`; each trial's origin names its target and the members its mutant read |
| `strategies.hpp` | the module's headers |

All the algorithms are generational: a generation is asked whole, and updated once in ticket order.

## 3. Checks

`check_heuresis_strategies` (68):

- **CMA-ES against pycma 4.5.0**, one generation at a time, over 8 generations in 6-D. Each generation loads pycma's state before `tell`, including the C^{−1/2} pycma uses (from its lazy eigendecomposition), tells our `cma_update` the same points, and compares the state after. The constants are pycma's (its c_σ, c_μ and d_σ differ from the tutorial's), the update rules ours. Fixtures: `tests/fixtures/make_cma_fixtures.py` writes `tests/fixtures/cma_fixture.hpp`.
  - Without active weights: everything agrees to 5 × 10⁻¹⁶.
  - With active weights: C agrees to 2.5 × 10⁻¹⁰, because pycma regularises the Mahalanobis norm of the negatively weighted steps as (‖·‖ + 10⁻⁹)², where the tutorial divides by ‖·‖². The rest agrees to rounding.
  - h_σ agrees in every generation. pycma's rule for h_σ differs from the tutorial's (‖p_σ‖²/n < 2 + 4/(n + 1) against (1.4 + 2/(n + 1))E‖N‖), but both held.
- **The tutorial's defaults:** λ and μ; positive weights summing to 1; negative weights summing to −min(α_μ⁻, α_μeff⁻, α_posdef⁻); 1 − c₁ − c_μΣw ≥ 0; the approximation of E‖N(0, I)‖ within 2 × 10⁻⁴ of √2Γ((n + 1)/2)/Γ(n/2).
- **Invariance:**
  - under x ↦ αRx + b, 40 generations: m, σ/α, RᵀCR, Rᵀp_c and Rᵀp_σ correspond to 1.3 × 10⁻¹³;
  - under a general linear map with σ held: m, A⁻¹CA⁻ᵀ and A⁻¹p_c correspond to 1.1 × 10⁻¹⁴;
  - under x ↦ e^{0.3f} − 7, and when a generation is told in reverse: the run is unchanged bit for bit.
- **Convergence to 10⁻¹⁰ in 10-D, over 5 seeds** (most evaluations: sphere 1,710; ellipsoid 4,400; rotated ellipsoid 4,630; Rosenbrock from 0 5,730). On the first three, going from 10⁻⁶ to 10⁻¹⁰ costs 0.6–1.6 times going from 10⁻² to 10⁻⁶, so convergence is linear.
  - Active CMA needs fewer evaluations than passive on the ellipsoid (means 4,316 and 6,084).
  - The separable variant solves the 30-D ellipsoid (6,776 evaluations) and keeps C diagonal.
- **Bounds, termination, injection:**
  - The box transformation evaluates only points of the box and converges to a corner optimum to 10⁻⁹. Reflection lets σ grow above 10σ₀ there.
  - The transformation inverts on the box and maps ℝ into it.
  - The tutorial's criteria stop CMA-ES on the sphere; the generation limit holds.
  - An injected optimum is ranked first, its step clipped.
- **Restarts on the 10-D Rastrigin function** (global minimum to 10⁻⁸):
  - IPOP needs 136,120 evaluations in 7 runs, λ doubling each time;
  - BIPOP needs 149,396 in 19 runs;
  - tickets stay unique across runs.
- **The (1+1)-ES:** reaches 10⁻¹⁰ on the 10-D sphere in 873 evaluations, converging linearly. log σ falls with log ‖x‖ (ratio within 15% of 1), so the success rate settles below 1/5, by the rate of convergence.
- **Differential evolution:**
  - with CR = 1, every trial is x_r1 + F(x_r2 − x_r3) exactly, or halfway from its target to a crossed bound;
  - with CR = 0, exactly one coordinate (j_rand) comes from the mutant;
  - telling a generation in reverse leaves the population unchanged;
  - it solves 5-D Rosenbrock (12,960 evaluations) and, with CR = 0.1, the separable 5-D Rastrigin (8,900).
- **SHADE:** solves the 10-D Rastrigin function (58,200 evaluations). Its memories stay in [0, 1] and (0, 1], and its archive at most NP. Its run is unchanged when told in reverse.

The program runs in about 1.5 s.

## 4. Dependencies

`search`, `variation`, `selection` (by the CMake target); calx `linalg` (the symmetric eigenproblem; `inverse` and `spectral_radius` in the checks); `benchmarks` in the checks.

## 5. References

- Akimoto, Y., Auger, A. and Glasmachers, T. (2018), Drift theory in continuous search spaces: expected hitting time of the (1+1)-ES with 1/5 success rule, *GECCO*.
- Auger, A. and Hansen, N. (2005), A restart CMA evolution strategy with increasing population size, *IEEE CEC*, 1769–1776.
- Hansen, N. (2009), Benchmarking a BI-population CMA-ES on the BBOB-2009 function testbed, *GECCO workshop*.
- Hansen, N. (2011), Injecting external solutions into CMA-ES, INRIA RR-7748.
- Hansen, N. (2016), The CMA evolution strategy: a tutorial, arXiv:1604.00772.
- Hansen, N. and Kern, S. (2004), Evaluating the CMA evolution strategy on multimodal test functions, *PPSN VIII*.
- Hansen, N. and Ostermeier, A. (2001), Completely derandomized self-adaptation in evolution strategies, *Evol. Comput.* 9, 159–195.
- Hansen, N. et al., pycma (BSD 3-clause), version 4.5.0, the fixtures' reference.
- Jägersküpper, J. (2007), Algorithmic analysis of a basic evolutionary algorithm for continuous optimization, *Theor. Comput. Sci.* 379, 329–347.
- Jastrebski, G. A. and Arnold, D. V. (2006), Improving evolution strategies through active covariance matrix adaptation, *IEEE CEC*.
- Kern, S., Müller, S. D., Hansen, N., Büche, D., Ocenasek, J. and Koumoutsakos, P. (2004), Learning probability distributions in continuous evolutionary algorithms — a comparative review, *Natural Computing* 3, 77–112.
- Rechenberg, I. (1973), *Evolutionsstrategie*, Frommann-Holzboog.
- Ros, R. and Hansen, N. (2008), A simple modification in CMA-ES achieving linear time and space complexity, *PPSN X*, 296–305.
- Storn, R. and Price, K. (1997), Differential evolution — a simple and efficient heuristic for global optimization over continuous spaces, *J. Global Optim.* 11, 341–359.
- Tanabe, R. and Fukunaga, A. (2013), Success-history based parameter adaptation for differential evolution, *IEEE CEC*, 71–78.
- Zhang, J. and Sanderson, A. C. (2009), JADE: adaptive differential evolution with optional external archive, *IEEE Trans. Evol. Comput.* 13, 945–958.

## 6. Open Questions

1. **Asynchronous variants.** Differential evolution and CMA-ES updated one evaluation at a time, for heterogeneous workers.
2. **Large dimensions.** LM-MA-ES and VD-CMA, for dimensions where O(n²) per sample is too much, and a calx backend for a generation's bulk arithmetic.
3. **The step size under general linear maps.** Whether to offer CSA's alternatives (two-point adaptation, median success rule), which would make the whole update invariant under them.
