# landscapes: The Structure of Fitness Landscapes

*Status: draft (2026-11-05): random walks and their correlations, fitness–distance correlation, and, on small hypercubes, exact correlations, elementary landscapes, local optima, basins and neutrality ([decision 0004](../development/decisions/0004-the-first-modules.md)).* Walsh spectra wait for calx's fast Walsh–Hadamard transform.

## 1. Concepts

A landscape is a function on the vertices of a graph: the space with the neighbourhood a search moves in. Whether a search can exploit it depends on its structure, which these tools measure. The autocorrelation ρ(s) of values along a random walk says how fast the landscape decorrelates with distance, and its correlation length −1/ln|ρ(1)| how far one can see. The fitness–distance correlation says whether values fall towards the optimum (r near +1 with objectives minimised) or mislead (r ≤ 0); it is a summary, not a proof of difficulty.

The Laplacian of the graph organises all of this. An elementary landscape is an eigenfunction of it up to a constant: Δf = −λ(f − f̄), Grover's wave equation, and then ρ(s) = (1 − λ/d)ˢ exactly. On the hypercube under one-bit flips the Walsh functions of order q are the eigenfunctions with λ = 2q, so OneMax (order 1) and an Ising energy without fields (order 2) are elementary, and any landscape is a sum of elementary ones whose ρ(1) is the variance-weighted mean of 1 − 2q/n. A random QUBO mixes orders 1 and 2 and is not elementary.

On small hypercubes everything can be counted: local minima, their basins under steepest descent (the vertices of a local optima network), and neutrality, the fraction of a point's neighbours of equal value and the components of equal value, the neutral networks along which a search drifts without selection.

## 2. Content

| Header | Content |
|---|---|
| `walks.hpp` | `random_walk(start, f, step, length, draws)`; `autocorrelation(walk, max_lag)` (Weinberger's estimator); `correlation_length(ρ₁)`; `fitness_distance_correlation(values, distances)` and over a sample with a distance to the optimum |
| `enumerated.hpp` | `Hypercube::tabulate(n, f)`, `point`, `index`; `one_step_correlation(h)`; `elementary(h)` → `Elementary{lambda, residual}`; `local_minima(h)`; `basins(h)` → `Basins{optima, sizes, attractor}`; `neutral_degree(h, m)`; `neutral_networks(h)` |
| `landscapes.hpp` | everything above |

## 3. Checks

`check_heuresis_landscapes` (18):
- OneMax elementary with λ = 2 (residual 4 × 10⁻¹⁴), ρ(1) = 1 − 2/n exactly, one local optimum; a walk of 4 × 10⁵ steps whose ρ(1…5) lie within 0.02 of (1 − 2/n)ˢ, and its correlation length within 10%; its fitness–distance correlation 1 over all 2¹⁰ points;
- an Ising energy without fields elementary with λ = 4; with fields, ρ(1) = 1 − (2Σh² + 4ΣJ²)/(n(Σh² + ΣJ²)) to 10⁻¹⁵; a random QUBO not elementary;
- NK at k = 0 elementary with λ = 2; at k = 3 the mean ρ(1) of 30 instances within 0.05 of Weinberger's 1 − (k + 1)/n (measured −0.030, of which about −0.022 is the finite tables' 2ᵐ/(2ᵐ − 1));
- Trap of two blocks of four: exactly four local optima, basins of 121, 55, 55 and 25 points as derived from the blocks' independence (a block with three or four ones completes, one with fewer empties), and a negative fitness–distance correlation;
- LeadingOnes: a point with i leading ones has n − i − 1 neutral neighbours, and there is one neutral network for each of the n + 1 values.

## 4. Dependencies

`search`, `variation`; tyche `descriptive` (means, variances, correlations); in the checks, `benchmarks`.

## 5. References

- Altenberg, L. (1997), Fitness distance correlation analysis: an instructive counterexample, *ICGA 7*, 57–64.
- Grover, L. K. (1992), Local search and the local structure of NP-complete problems, *Oper. Res. Lett.* 12, 235–243.
- Jones, T. and Forrest, S. (1995), Fitness distance correlation as a measure of problem difficulty for genetic algorithms, *ICGA 6*, 184–192.
- Ochoa, G., Tomassini, M., Vérel, S. and Darabos, C. (2008), A study of NK landscapes' basins and local optima networks, *GECCO*, 555–562.
- Reidys, C. M. and Stadler, P. F. (2001), Neutrality in fitness landscapes, *Appl. Math. Comput.* 117, 321–350.
- Stadler, P. F. (1996), Landscapes and their correlation functions, *J. Math. Chem.* 20, 1–45.
- Stadler, P. F. (2002), Fitness landscapes, in *Biological Evolution and Statistical Physics*, Springer, 183–204.
- Weinberger, E. D. (1990), Correlated and uncorrelated fitness landscapes and how to tell the difference, *Biol. Cybern.* 63, 325–336.
- Weinberger, E. D. (1991), Local properties of Kauffman's N-k model: a tunably rugged energy landscape, *Phys. Rev. A* 44, 6399–6413.

## 6. Open Questions

1. Walsh spectra (the variance of each order) by the fast Walsh–Hadamard transform, which belongs in calx's `transforms` (requested), and Fourier spectra of landscapes on Sₙ by morphe's transforms on finite groups.
2. Local optima networks with their edges (escape probabilities between basins), and sampled rather than enumerated versions for large spaces.
3. Landscapes under other neighbourhoods than one flip (the walk tools already take any step).
