# The Theory of Search

*Status: draft (2026-11-05). Learning: the mathematics on which heuresis's algorithms and checks rest, with the results that the checks hold numerically.*

## 1. Problems as Maps into Ordered Sets

**Definition 1.1.** A *problem* is a map f: X → V from a space X to a set V with a preorder ≼. The space is finite or a measurable space; points with f(x) ≼ f(y) are at least as good as y. A *minimiser* is a point x with f(x) ≼ f(y) for every y.

The library uses three orders on V.
- **One objective.** On V = ℝ ∪ {+∞}, the usual order, with failures at +∞.
- **Constraints.** On V = ℝ × ℝⁿ_c × {ok, failed}, *Deb's order*: failures last; otherwise compare the total violation v = Σ max(0, gⱼ), and among equal violations the objective (Deb 2000). It is a total preorder, and on the feasible set (v = 0) it is the order of the objective. A penalty method replaces it by f + λv, which needs λ and changes the minimisers when λ is too small. Deb's order needs neither.
- **Several objectives.** On V = ℝᵐ, the *Pareto order* (§6), which is only partial.

A problem may also carry *descriptors* d: X → D into a space of behaviours (§8), and a *cost* per evaluation, the unit of the budget.

## 2. No Free Lunch

Let X and V be finite. An algorithm *does not revisit*: it evaluates distinct points. After m evaluations it has seen a *trace*, the sequence of values (f(x₁), …, f(x_m)).

**Theorem 2.1** (Wolpert and Macready 1997, Theorem 1). For any two algorithms a₁, a₂ and any m, the trace distributions summed over all f ∈ V^X agree: Σ_f P(trace | f, m, a₁) = Σ_f P(trace | f, m, a₂).

**Theorem 2.2** (Schumacher, Vose and Whitley 2001). The equality holds for the sum over a set F of functions exactly when F is closed under permutations of X: f ∈ F implies f ∘ σ ∈ F for every bijection σ of X.

*Remark.* A search can therefore beat random search only on a class of problems that is not closed under permutations of the space, that is, only by exploiting a structure of X (a neighbourhood, a metric, a symmetry, a decomposition) on which the problems are regular. Every kernel of heuresis states which structure it assumes. Supplying structure is what the domains do: valence rules, a body's mechanics, a grammar of programs.

## 3. Algorithms as Markov Chains, and Invariance under the Order

**Definition 3.1.** A search algorithm is a state space S with an initial state s₀ and two maps. ask: S → S × X^λ draws candidates, and tell: S × V^λ → S updates the state (decision 0003). With its random draws, the pair is a Markov chain on S driven by f.

**Definition 3.2.** An algorithm is *comparison-based* when tell depends on the values only through the preorder they induce on the candidates (and on archived points): the ranks, with ties.

**Proposition 3.3.** Let g: V → V be strictly increasing. A comparison-based algorithm run on f and on g ∘ f, with the same draws, produces the same sequence of states.

*Proof.* By induction on the steps. The states agree at step 0. If they agree at step t, the candidates asked agree, being functions of the state and the draws. Their values under f and g ∘ f induce the same preorder, because g is strictly increasing. So tell produces the same state. ∎

Tournaments, truncation, ranking, lexicase, comma and plus selection, CMA-ES's weighted recombination and non-dominated sorting are comparison-based. Fitness-proportional and Boltzmann selection are not. The checks of `selection` and `strategies` verify Proposition 3.3 exactly, to the bit, under x ↦ e^{3x} − 40 and similar maps.

## 4. Invariance under Groups Acting on the Space

**Definition 4.1.** Let a group H act on X and on the states S. An algorithm is *H-invariant* when, for every h ∈ H, the run on f ∘ h⁻¹ from h·s₀ is the image under h of the run on f from s₀ (as laws of the chain; with the same draws when the action commutes with them).

An invariant algorithm performs identically on every problem of an orbit {f ∘ h⁻¹}. Its behaviour on one problem predicts its behaviour on all of them, and a benchmark result transfers along the orbit (Hansen, Ros, Mauny, Schoenauer and Auger 2011).

**Examples.**
1. **Bit strings.** The hypercube {0,1}ⁿ is the Cayley graph of (ℤ/2)ⁿ with the unit vectors as generators. Its automorphism group is the hyperoctahedral group (ℤ/2)ⁿ ⋊ Sₙ: translations x ↦ x ⊕ m and permutations of positions.
   - Standard bit mutation chooses the flipped positions independently of the values, so it commutes with every translation under the same draws, and with every permutation in law. Uniform crossover likewise.
   - k-point crossover commutes with translations but, among permutations of positions, only with the reversal i ↦ n − 1 − i. It treats adjacent positions as linked.
   - The checks of `variation` verify the translations exactly.
2. **Real vectors.** CMA-ES, with its initial mean and covariance transformed, is invariant under the affine group x ↦ Ax + b of ℝⁿ (Hansen 2016). Isotropic Gaussian mutation is invariant only under the Euclidean group, and a separable algorithm only under the permutations and reflections of the coordinates together with translations.
3. **Permutations.** On Sₙ a mutation is a step of a random walk on a Cayley graph Cay(Sₙ, T), where T is the set of transpositions, of adjacent transpositions, or of segment reversals. Kendall's distance is the graph distance of the adjacent transpositions and Cayley's that of all transpositions. Both are invariant under the action of Sₙ × Sₙ by left and right multiplication.

*Remark.* Choosing an operator chooses a symmetry group. The more symmetric the operator, the larger the orbit on which its performance is the same. The less symmetric, the more structure it assumes.

## 5. Search Distributions and the Information-Geometric Flow

Many algorithms keep no population, only a family of distributions P_θ on X, and move θ towards where the good samples fell.

**The IGO flow** (Ollivier, Arnold, Auger and Hansen 2017).
- **Quantile weights.** Let W_θ^f(x) = w(P_θ{y : f(y) ≺ f(x)}) for a non-increasing weight w on [0, 1]. This is a rewriting of f by its quantiles, so it is invariant under §3's strictly increasing maps.
- **The flow.** dθ/dt = Ĩ(θ)⁻¹ ∫ W_θ^f(x) ∇_θ ln P_θ(x) P_θ(dx), the natural gradient of the expected weight in the Fisher metric. It is invariant under reparametrisations of θ as well as under §3's maps.
- **Discretisation.** With λ samples ranked by f, weights ŵᵢ, and a step δt: θ ← θ + δt Σ ŵᵢ Ĩ⁻¹ ∇_θ ln P_θ(xᵢ).

**Proposition 5.1.** For an exponential family in its expectation parameters T̄ = E_θ[T(x)], one has Ĩ⁻¹ ∇_θ ln P_θ(x) = T(x) − T̄, so the step is T̄ ← T̄ + δt Σ ŵᵢ (T(xᵢ) − T̄) (Ollivier et al. 2017).

**Corollary 5.2.**
1. **Bernoulli.** For the product Bernoulli family on {0,1}ⁿ, T(x) = x and T̄ = p, so p ← p + δt Σ ŵᵢ(xᵢ − p). With Σ ŵᵢ = 1 this is PBIL's update (Baluja 1994) with learning rate δt.
2. **Gaussian.** For Gaussians in (m, C), T(x) = (x, xxᵀ). The step gives m ← m + δt Σ ŵᵢ(xᵢ − m) and C ← C + δt Σ ŵᵢ((xᵢ − m)(xᵢ − m)ᵀ − C). This is CMA-ES's rank-μ update without evolution paths or step-size control (Hansen 2016).

The checks of `distributions` verify both to rounding.

**Proposition 5.3** (quantum-inspired algorithms are product distributions). Let n Q-bits be unentangled, each a state αᵢ|0⟩ + βᵢ|1⟩. Observing them in the computational basis draws x with P(x) = Πᵢ |βᵢ|^{2xᵢ} |αᵢ|^{2(1−xᵢ)}, a product Bernoulli law with pᵢ = |βᵢ|² = sin²θᵢ. Hence the quantum-inspired evolutionary algorithm of Han and Kim (2002) is a search distribution of the product Bernoulli family. Its rotation gate is an update of θ, and no quantum effect enters.

*Proof.* The joint state is the tensor product ⊗ᵢ(αᵢ|0⟩ + βᵢ|1⟩), whose amplitude on |x⟩ is Πᵢ αᵢ^{1−xᵢ} βᵢ^{xᵢ}. Born's rule squares its modulus. ∎

**Samplers.** A sampler is the device that draws from P_θ. The CPU's sampler defines the law. A quantum device, an annealer or a probabilistic-bit machine samples a law of its own, and its correctness is that law (decision 0003 §2.8).

## 6. Many Objectives

**Definition 6.1.** On ℝᵐ, a *dominates* b, written a ≺ b, when aᵢ ≤ bᵢ for every i and a ≠ b. This is a strict partial order.
- The *Pareto set* is the set of points not dominated by any other, and its image is the *Pareto front*.
- *Constrained dominance* (Deb, Pratap, Agarwal and Meyarivan 2002) composes this order with Deb's order of §1: feasibility, then violation, then Pareto dominance.

**Ranks are heights.** A finite set P with ≺ is a finite poset. Its *non-dominated sorting* is the partition into the antichains F₁ = min P, F₂ = min(P ∖ F₁), …, so the rank of a point is its height in the poset. Fast non-dominated sorting computes it in O(mN²) (Deb et al. 2002).

**Hypervolume.** For a finite A ⊂ ℝᵐ and a reference point r, HV(A) = λ(∪_{a∈A, a≤r} [a, r]), the Lebesgue measure of the region A dominates below r.

**Theorem 6.2** (strict Pareto compliance; Zitzler, Thiele, Laumanns, Fonseca and da Fonseca 2003; Fleischer 2003). If every point of B is weakly dominated by a point of A, and some point of A dominates a point of B that A does not contain, then HV(A) > HV(B) for any r that both sets dominate somewhere.

Maximising the hypervolume over sets of μ points therefore places them on the front. The checks hold the library's three algorithms to one another and to inclusion–exclusion:
- a sweep in two objectives;
- a sweep in three (Beume et al. 2009);
- WFG for any m (While, Bradstreet and Barone 2012).

**The fronts' exact hypervolumes.** With r = (1, …, 1) and the fronts of `benchmarks`, a point y ∈ [0,1]ᵐ is dominated by the front exactly when it lies above it, so HV is the measure of that region.
- **DTLZ1** (Σfᵢ = ½). y is dominated exactly when Σyᵢ ≥ ½: if so, f = y/(2Σyᵢ) ≤ y lies on the front; if not, every f ≤ y has Σfᵢ < ½. So HV = 1 − 2⁻ᵐ/m!.
- **DTLZ2–4** (‖f‖ = 1). HV = 1 − π^{m/2}/(2ᵐ Γ(m/2 + 1)).
- **ZDT1** (f₂ = 1 − √f₁). HV = ∫₀¹ √t dt = 2/3.
- **ZDT2** (f₂ = 1 − f₁²). HV = ∫₀¹ t² dt = 1/3.

## 7. Runtimes

**The (1+1) EA.** It mutates its point by standard bit mutation at rate 1/n and keeps the child when it is no worse.

**Theorem 7.1** (fitness levels; Wegener 2002). Partition X into levels A₁ <_f … <_f A_k, the last holding the optima, and let sᵢ be a lower bound on the probability of leaving Aᵢ for a higher level. Then E[T] ≤ Σᵢ 1/sᵢ.

**Consequences.**
- **OneMax.** sᵢ ≥ (n − i)/(en), so E[T] ≤ en H_n ≈ en ln n. The bound is tight: E[T] = en ln n − c₁n + O(log n), with c₁ ≈ 1.89 (Doerr, Fouz and Witt 2011; Hwang, Panholzer, Rolin, Tsai and Chen 2018).
- **LeadingOnes.** E[T] = ((e − 1)/2) n² (1 + o(1)) (Böttcher, Doerr and Neumann 2010).
- **Jump_k.** E[T] = Θ(nᵏ) (Droste, Jansen and Wegener 2002). Heavy-tailed mutation reduces this by a factor k^{Θ(k)} (Doerr, Le, Makhmara and Nguyen 2017).

*Remark.* For the (1+1) EA on OneMax the number of ones is itself a Markov chain on {0, …, n}. Its expected hitting time can be computed exactly from the transition probabilities, which are sums of products of binomials. The checks of `genetic` compare runs with this exact value, not only with the leading term.

**The (1+(λ,λ)) GA** solves OneMax in expected O(n) evaluations with its self-adjusting λ, the first algorithm with crossover proved faster than every mutation-based one on it (Doerr, Doerr and Ebel 2015; Doerr and Doerr 2018).

## 8. Diversity and Illumination

A repertoire search maps a partition {Cⱼ} of the descriptor space D to elites: for each cell, the best point found whose descriptor falls in it (MAP-Elites; Mouret and Clune 2015).
- **Coverage** is the fraction of cells filled.
- **QD-score** is the sum over filled cells of the elites' qualities, offset to be positive. It is monotone non-decreasing along a run, since an elite is replaced only by a better point of its cell.
- **Cells.** On a grid the cells are boxes. In high-dimensional descriptor spaces they are the Voronoi cells of centroids chosen to tile D evenly (Vassiliades, Chatzilygeroudis and Mouret 2018).

*Rationale.* For discovery the question is often "what is possible here, and how well can each kind of solution do?" rather than "what is best?". A repertoire answers the first; an optimum answers only the second.

## 9. Annealing and Sampling

**The Boltzmann law.** For an energy f on a finite X, π_T(x) = e^{−f(x)/T}/Z(T) concentrates on the minimisers as T → 0.

**Metropolis chains.** Let the proposal kernel q be symmetric. Accepting a proposal y from x with probability min(1, e^{−(f(y)−f(x))/T}) satisfies detailed balance with π_T. If q is irreducible and the chain aperiodic, π_T is its unique stationary law (Metropolis et al. 1953; Hastings 1970).

**Annealing schedules.** Lowering T slowly enough finds a minimiser with probability tending to 1. The logarithmic schedule T_t = c/ln(t + 1) suffices exactly when c is at least the depth of the deepest local minimum that is not global (Hajek 1988).

**Parallel tempering.** It runs replicas at temperatures T₁ < … < T_K and proposes swaps of neighbouring replicas, accepted with probability min(1, e^{(1/Tᵢ − 1/Tᵢ₊₁)(fᵢ − fᵢ₊₁)}). This keeps the product law Πₖ π_{Tₖ} stationary (Geyer 1991; Hukushima and Nemoto 1996).

**Population annealing.** Resampling a population from T to T′ with weights e^{−(1/T′ − 1/T) f} makes the mean weight an unbiased estimator of Z(T′)/Z(T) (Hukushima and Iba 2003; Machta 2010). The product of the mean weights over a schedule then estimates ln Z. The checks compare it with ln Z computed by enumeration.

## 10. Quantum Search, and What a Device's Law Is

Let N items be searched by an oracle that marks the t items with f(x) < τ, and put sin²θ = t/N.
- **The invariant plane.** Grover's iteration G (the oracle's reflection, then the reflection about the uniform state) maps the plane spanned by |good⟩ (the uniform state on the marked items) and |bad⟩ (the uniform state on the rest) into itself as a rotation by 2θ.
- **The law after r iterations.** From the uniform state, at angle θ from |bad⟩, r iterations give the state at angle (2r + 1)θ.

**Proposition 10.1.** After r iterations, measuring gives a marked item with probability sin²((2r + 1)θ), uniformly among the marked items. Otherwise it gives an unmarked item, uniformly among those (Boyer, Brassard, Høyer and Tapp 1998).

This is the *law of the device*. For heuresis it is all that matters about the device, and it can be emulated exactly by a classical sampler that knows the marked set (at a classical cost of N evaluations, reported apart from the oracle calls).
- **Unknown t.** When t is unknown, BBHT draws r uniformly below a bound m that grows by a factor 6/5 after each failure, and finds a marked item in O(√(N/t)) expected oracle calls.
- **Minimum finding.** Dürr and Høyer (1996) lower τ to each improvement found, and find the minimum in expected fewer than 22.5√N oracle calls.
- **Global optimisation.** Grover adaptive search (Baritompa, Bulger and Wood 2005) is the same scheme for global optimisation, with its own rules for choosing r.

*Caveat.* The quadratic speed-up is in oracle calls, against the N/t expected calls of classical random search for the same threshold. It is not a speed-up in wall time on a classical emulator, and it holds only when the oracle is a quantum circuit.

## 11. Programs as Terms

**Signatures and terms.** A *signature* Σ has a set of sorts S and operation symbols σ: s₁ × … × s_k → s. Its *terms* of sort s are the trees built from the symbols with matching sorts: the term algebra T_Σ, the initial Σ-algebra. A program is a term, and an interpreter is the unique homomorphism T_Σ → A to the algebra of meanings. Strongly typed genetic programming generates and varies only well-sorted terms (Montana 1995). Subtree crossover exchanges subterms of equal sort, so it maps well-sorted terms to well-sorted terms.

**Dimensions.** Physical dimensions form the free abelian group ℤ⁷ over the SI base units (physis `units`). Give each variable its dimension, multiplication and division the group's operation and its inverse, addition the requirement of equal dimensions, and transcendental functions dimensionless arguments. The dimension of a term is then a partial homomorphism into ℤ⁷, defined exactly on the dimensionally consistent terms. Generating terms of a prescribed dimension makes symbolic regression search only formulas that could be physical laws.

**Grammars.** A context-free grammar generates derivation trees, and grammar-guided programming varies those trees (Whigham 1995).
- Grammatical evolution's genotype is a sequence of integers (codons). The phenotype is the leftmost derivation that rewrites the first nonterminal by its rule numbered (codon mod number of rules), wrapping the sequence when it runs out (Ryan, Collins and O'Neill 1998).
- The map is total on derivations but not injective, and it is not local: one codon's change can rewrite the whole remainder.

## 12. Distributed Search

**Islands.** An island model runs sub-populations joined by a migration topology, a directed graph, and exchanges individuals every τ generations.
- **Takeover time.** Under selection alone, takeover is the time for the best individual to fill the population. It measures how fast information crosses the topology. On a cellular algorithm with deterministic best-of-neighbourhood replacement it equals the eccentricity of the starting cell in the neighbourhood graph (Rudolph 2000; Giacobini, Tomassini, Tettamanzi and Alba 2005).
- **Diversity.** Sparse topologies keep islands diverse for longer.

**Asynchronous evaluation.** Generational algorithms wait for their slowest evaluation; asynchronous ones do not, at the price of updating on partial information. heuresis's simulated workers measure that trade-off in simulated wall time before a search is spread over real machines (Scott and De Jong 2015).

## 13. Surrogates

A Gaussian process prior on f, conditioned on the evaluations, gives at each x a posterior mean μ(x) and standard deviation σ(x) (tyche `spatial`).

**Proposition 13.1.** With the best value f*, improvement I = max(0, f* − f(x)) and z = (f* − μ)/σ, the *expected improvement* is EI(x) = (f* − μ)Φ(z) + σφ(z), where Φ and φ are the standard normal distribution and density (Jones, Schonlau and Welch 1998).

Maximising EI balances a low mean against a high uncertainty. Bayesian optimisation evaluates its maximiser and conditions again, which suits evaluations that are expensive simulations.

## 14. References

- Baluja, S. (1994), Population-based incremental learning, Technical Report CMU-CS-94-163, Carnegie Mellon University.
- Baritompa, W. P., Bulger, D. W. and Wood, G. R. (2005), Grover's quantum algorithm applied to global optimization, *SIAM J. Optim.* 15, 1170–1184.
- Beume, N., Fonseca, C. M., López-Ibáñez, M., Paquete, L. and Vahrenhold, J. (2009), On the complexity of computing the hypervolume indicator, *IEEE Trans. Evol. Comput.* 13, 1075–1082.
- Böttcher, S., Doerr, B. and Neumann, F. (2010), Optimal fixed and adaptive mutation rates for the LeadingOnes problem, *PPSN XI*, 1–10.
- Boyer, M., Brassard, G., Høyer, P. and Tapp, A. (1998), Tight bounds on quantum searching, *Fortschr. Phys.* 46, 493–505.
- Deb, K. (2000), An efficient constraint handling method for genetic algorithms, *Comput. Methods Appl. Mech. Engrg.* 186, 311–338.
- Deb, K., Pratap, A., Agarwal, S. and Meyarivan, T. (2002), A fast and elitist multiobjective genetic algorithm: NSGA-II, *IEEE Trans. Evol. Comput.* 6, 182–197.
- Doerr, B., Doerr, C. and Ebel, F. (2015), From black-box complexity to designing new genetic algorithms, *Theor. Comput. Sci.* 567, 87–104.
- Doerr, B., Fouz, M. and Witt, C. (2011), Sharp bounds by probability-generating functions and variable drift, *GECCO '11*, 2083–2090.
- Doerr, B., Le, H. P., Makhmara, R. and Nguyen, T. D. (2017), Fast genetic algorithms, *GECCO '17*, 777–784.
- Droste, S., Jansen, T. and Wegener, I. (2002), On the analysis of the (1+1) evolutionary algorithm, *Theor. Comput. Sci.* 276, 51–81.
- Dürr, C. and Høyer, P. (1996), A quantum algorithm for finding the minimum, arXiv:quant-ph/9607014.
- Fleischer, M. (2003), The measure of Pareto optima: applications to multi-objective metaheuristics, *EMO 2003*, LNCS 2632, 519–533.
- Geyer, C. J. (1991), Markov chain Monte Carlo maximum likelihood, in *Computing Science and Statistics*, 156–163.
- Giacobini, M., Tomassini, M., Tettamanzi, A. G. B. and Alba, E. (2005), Selection intensity in cellular evolutionary algorithms for regular lattices, *IEEE Trans. Evol. Comput.* 9, 489–505.
- Hajek, B. (1988), Cooling schedules for optimal annealing, *Math. Oper. Res.* 13, 311–329.
- Hastings, W. K. (1970), Monte Carlo sampling methods using Markov chains and their applications, *Biometrika* 57, 97–109.
- Han, K.-H. and Kim, J.-H. (2002), Quantum-inspired evolutionary algorithm for a class of combinatorial optimization, *IEEE Trans. Evol. Comput.* 6, 580–593.
- Hansen, N. (2016), The CMA evolution strategy: a tutorial, arXiv:1604.00772.
- Hansen, N., Ros, R., Mauny, N., Schoenauer, M. and Auger, A. (2011), Impacts of invariance in search: when CMA-ES and PSO face ill-conditioned and non-separable problems, *Appl. Soft Comput.* 11, 5755–5769.
- Hukushima, K. and Iba, Y. (2003), Population annealing and its application to a spin glass, *AIP Conf. Proc.* 690, 200–206.
- Hukushima, K. and Nemoto, K. (1996), Exchange Monte Carlo method and application to spin glass simulations, *J. Phys. Soc. Jpn.* 65, 1604–1608.
- Hwang, H.-K., Panholzer, A., Rolin, N., Tsai, T.-H. and Chen, W.-M. (2018), Probabilistic analysis of the (1+1)-evolutionary algorithm, *Evol. Comput.* 26, 299–345.
- Jones, D. R., Schonlau, M. and Welch, W. J. (1998), Efficient global optimization of expensive black-box functions, *J. Global Optim.* 13, 455–492.
- Machta, J. (2010), Population annealing with weighted averages: a Monte Carlo method for rough free-energy landscapes, *Phys. Rev. E* 82, 026704.
- Metropolis, N., Rosenbluth, A. W., Rosenbluth, M. N., Teller, A. H. and Teller, E. (1953), Equation of state calculations by fast computing machines, *J. Chem. Phys.* 21, 1087–1092.
- Montana, D. J. (1995), Strongly typed genetic programming, *Evol. Comput.* 3, 199–230.
- Mouret, J.-B. and Clune, J. (2015), Illuminating search spaces by mapping elites, arXiv:1504.04909.
- Ollivier, Y., Arnold, L., Auger, A. and Hansen, N. (2017), Information-geometric optimization algorithms: a unifying picture via invariance principles, *J. Mach. Learn. Res.* 18, 1–65.
- Rudolph, G. (2000), On takeover times in spatially structured populations: array and ring, *Proc. 2nd Asia-Pacific Conf. Genetic Algorithms and Applications*, 144–151.
- Ryan, C., Collins, J. J. and O'Neill, M. (1998), Grammatical evolution: evolving programs for an arbitrary language, *EuroGP '98*, LNCS 1391, 83–96.
- Schumacher, C., Vose, M. D. and Whitley, L. D. (2001), The no free lunch and problem description length, *GECCO '01*, 565–570.
- Scott, E. O. and De Jong, K. A. (2015), Understanding simulated asynchronous evolutionary algorithms, *FOGA '15*, 85–92.
- Vassiliades, V., Chatzilygeroudis, K. and Mouret, J.-B. (2018), Using centroidal Voronoi tessellations to scale up the multidimensional archive of phenotypic elites algorithm, *IEEE Trans. Evol. Comput.* 22, 623–630.
- Wegener, I. (2002), Methods for the analysis of evolutionary algorithms on pseudo-Boolean functions, in *Evolutionary Optimization*, Kluwer, 349–369.
- Whigham, P. A. (1995), Grammatically-based genetic programming, *Proc. Workshop on Genetic Programming: From Theory to Real-World Applications*, 33–41.
- While, L., Bradstreet, L. and Barone, L. (2012), A fast way of calculating exact hypervolumes, *IEEE Trans. Evol. Comput.* 16, 86–95.
- Wolpert, D. H. and Macready, W. G. (1997), No free lunch theorems for optimization, *IEEE Trans. Evol. Comput.* 1, 67–82.
- Zitzler, E., Thiele, L., Laumanns, M., Fonseca, C. M. and da Fonseca, V. G. (2003), Performance assessment of multiobjective optimizers: an analysis and review, *IEEE Trans. Evol. Comput.* 7, 117–132.
