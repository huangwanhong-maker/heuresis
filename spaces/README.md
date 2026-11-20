# spaces: Typed Spaces and Their Operators

*Status: draft (2026-11-20; [decision 0008](../development/decisions/0008-typed-spaces.md)).*

## 1. Concepts

A search moves among values of some kind: amounts on a line, units shared among parts, an order of items, a subset, an orientation in space. Each kind has its own geometry, and a good operator respects it. Here each kind is a **space**: it supplies a way to draw values, to move a value within a ball of its metric, and to measure distances, with a declared **unit** of distance. It may supply a recombination (and say whether it is geometric, pure and respectful), a repair, a chart for continuous and curved spaces, a standardised embedding in ℝᵏ, a floor on the probability that a mutation moves, a bound on its distances, and, when it is finite, its values.

Variation is defined by the metric (Moraglio 2007). A mutation draws from a ball about its parent; a geometric crossover draws from the segment between its parents, the values c with d(x, c) + d(c, y) = d(x, y). On a product of spaces, crossing each part geometrically is geometric for the weighted sum of the parts' distances (Moraglio's product theorem), so a genotype built of many kinds keeps the property its parts have. Each part's distance is divided by its unit before the sum, so that a wide part does not drown a narrow one.

Every value, of every space, is one tree (`Value`): a real or integer vector, a category, a permutation, a set, a sequence, a record, a choice, a point of a manifold. heuresis's evolutionary loop takes any copyable genotype, so it runs on every space through one adapter.

Every space is held to the same **laws**, written once (`laws.hpp`) and run on any space, a person's as well as the built-in ones. A law that fails is reported with its worst case, not refused: an operator that is not geometric can still serve a search, but a search should know.

## 2. Content

| Header | What it holds |
|---|---|
| `value.hpp` | the value tree; `key`, exact (reals in hexadecimal), so that equal values share a key and only they do; `words` |
| `space.hpp` | the contract: required and optional operations, `Properties`, `Refusal` |
| `base.hpp` | `RealBox` (Euclidean; Gaussian steps; line recombination); `IntegerBox` and `multiset_space` (ℓ¹; Rudolph's geometric steps; box recombination); `Categorical` (discrete metric); `Ordinal` (path metric); `Simplex` (Aitchison's geometry in ilr coordinates, from tyche's `compositions`); `PermutationSpace` under Cayley's, Hamming's or Kendall's metric (cycle crossover; a point of a bubble-sort path); `Subset` with bounds on its size (symmetric difference); `Composition`, N identical units among D parts with caps (transport distance); `SequenceSpace` of any space's values (edit distance; homologous recombination along an optimal alignment) |
| `manifolds.hpp` | `Sphere` Sⁿ⁻¹ and `Rotations` SO(3) (physis's rotations): geodesic distance, exponential map and logarithm within the injectivity radius, slerp |
| `constructors.hpp` | `Product` of named, weighted parts; `Choice`, a tagged union with a metric across branches; `Constrained`, a subspace by a predicate with repair or counted rejection |
| `laws.hpp` | `check_laws`: validity, the metric axioms, the diameter, the ball and floor of mutation, purity, the segment, repair, the chart's identities, the embedding, the unit, the elements, the keys |
| `diagnostics.hpp` | Raidl and Gottlieb's measures of a representation through any decoder (mutation innovation, redundancy, crossover innovation, the excess over the phenotypic segment); fitness–distance correlation; χ² uniformity of a finite sampler, given size if asked |
| `adapters.hpp` | `operators(space)`: the maker, mutations at several scales, recombination, the distance in units and the key, as `evolution::Operators<Value>` |
| `spaces.hpp` | all of the above |

### 2.1 Units

| Space | Metric | Unit | Exact because |
|---|---|---|---|
| real box | Euclidean | √(Σwᵢ²/6), root-mean-square | E(X − Y)² = w²/6 on an interval |
| integer box, multisets | ℓ¹ | Σ(nᵢ² − 1)/(3nᵢ), mean | the mean |X − Y| of two uniform integers among n |
| categories | discrete | 1 − 1/k | P(X ≠ Y) |
| ordered levels | |i − j| | (k² − 1)/(3k) | as an integer box |
| simplex | Aitchison | π√((D − 1)/3), root-mean-square | Var(log E) = π²/6 for E exponential |
| permutations | Cayley, Hamming, Kendall | n − Hₙ, n − 1, n(n − 1)/4 | mean cycles Hₙ; mean fixed points 1; half the pairs inverted |
| subsets | symmetric difference | 2E[K](1 − E[K]/n) | E|A ∩ B| = E[K]²/n for independent uniform sets |
| compositions | transport | from the uniform marginal (estimated with caps) | P(Xᵢ = j) = C(N − j + D − 2, D − 2)/C(N + D − 1, D − 1) |
| sequences | edit | estimated | — |
| sphere | geodesic | π/2 | y ↦ −y maps θ to π − θ |
| SO(3) | geodesic | π/2 + 2/π | the angle has density (1 − cos θ)/π |
| product | Σ wᵢ dᵢ/uᵢ | Σ wᵢ | exact when every part's unit is its mean |

### 2.2 Using a Space

```cpp
#include <heuresis/spaces/spaces.hpp>
namespace hs = heuresis::spaces;

auto genotype = std::make_shared<hs::Product>(std::vector<hs::Part>{
    {"shares", std::make_shared<hs::Simplex>(4, 1e6), 1},            // a budget of 10⁶ in four shares
    {"order", std::make_shared<hs::PermutationSpace>(6), 1},          // the order of six grants
    {"chosen", std::make_shared<hs::Subset>(10, 2, 4), 1}});         // two to four sites of ten
heuresis::evolution::Evolution<hs::Value> e(hs::operators(genotype), strategy);
```

## 3. Checks

`tests/check_spaces.cpp`, 83 checks:

- every law on 26 spaces: every base space, two manifolds, a product of five spaces, two choices and two constrained subspaces;
- the laws find four faults planted in a broken space and pass what is sound in it;
- units and diameters equal the mean and greatest distance over every pair of ten finite spaces, the product's Σwᵢ among them;
- the product theorem on 30 random products of geometric spaces (3,000 pairs);
- a choice's distance is a metric on all 3,375 triples of a finite choice;
- samplers uniform by χ² over every value: subsets of bounded size, compositions with and without caps, permutations, sequences given their length, a choice by size, a product;
- the sphere's and SO(3)'s retractions invert their logarithms to 10⁻¹⁵; the trust radius π is sharp; SO(3)'s values have one sign; the simplex's distance is Aitchison's from clr coordinates computed apart, and its ilr chart a bijection;
- line recombination of random keys is geometric under Kendall's metric on the decoded orders (each pair both parents order alike the child orders alike); random keys at a small scale leave the order unchanged in nine mutations of ten, a direct permutation in none;
- an evolution over a product of seven spaces finds a known optimum, its discrete parts exactly;
- refusals.

## 4. Dependencies

heuresis's `variation` (permutations, Rudolph's step, distances) and `evolution` (the adapter); physis's `geometry` (rotations); tyche's `compositions` (ilr coordinates); calx's random streams.

## 5. References

- Moraglio, A. (2007). *Towards a Geometric Unification of Evolutionary Algorithms.* PhD thesis, University of Essex.
- Moraglio, A., Poli, R. (2005). Topological crossover for the permutation representation. *GECCO workshops.*
- Radcliffe, N. J. (1994). The algebra of genetic algorithms. *Annals of Mathematics and Artificial Intelligence* 10, 339–384.
- Rudolph, G. (1994). An evolutionary algorithm for integer programming. *PPSN III*, 139–148.
- Aitchison, J. (1986). *The Statistical Analysis of Compositional Data.* Chapman and Hall. Egozcue, J. J., et al. (2003). Isometric logratio transformations for compositional data analysis. *Mathematical Geology* 35, 279–300.
- Absil, P.-A., Mahony, R., Sepulchre, R. (2008). *Optimization Algorithms on Matrix Manifolds.* Princeton University Press.
- Shoemake, K. (1985). Animating rotation with quaternion curves. *SIGGRAPH '85*, 245–254. Shoemake, K. (1992). Uniform random rotations. *Graphics Gems III*, 124–132.
- Duchon, P., Flajolet, P., Louchard, G., Schaeffer, G. (2004). Boltzmann samplers for the random generation of combinatorial structures. *Combinatorics, Probability and Computing* 13, 577–625.
- Bentley, J., Floyd, R. (1987). A sample of brilliance. *Communications of the ACM* 30, 754–757.
- Raidl, G. R., Gottlieb, J. (2005). Empirical analysis of locality, heritability and heuristic bias in evolutionary algorithms. *Evolutionary Computation* 13(4), 441–475.
- Jones, T., Forrest, S. (1995). Fitness distance correlation as a measure of problem difficulty for genetic algorithms. *ICGA*, 184–192.

## 6. Open Questions

- **Not yet:** graphs (the planned `graphs`); a dependent product, whose parts' shapes depend on one another; transport of tangent vectors and a basis of the chart, which CMA-ES on a manifold would need; shrinkers for the laws' counterexamples; a joint covariance across spaces with no common embedding (an open problem the survey names).
- **Respectfulness** is declared by each space but not checked generically: what two values share has no meaning independent of a space (the simplex declares none, since it depends on the basis).
