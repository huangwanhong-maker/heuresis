# variation: Spaces and Their Kernels

*Status: draft (2026-11-05): real boxes, bit strings, integer boxes and permutations, with their repairs, mutations, crossovers and distances ([decision 0004](../development/decisions/0004-the-first-modules.md)).*

## 1. Concepts

A representation is a space with a way to sample it, kernels that vary its points, repairs that bring points back into it, and distances. A mutation is a Markov kernel from one parent to one child. A crossover takes two parents to two children; for the position-wise operators the second child is the complement of the first. A kernel draws from the stream it is given and from nothing else.

**Symmetry.** Each kernel commutes with a group, and choosing an operator means stating which symmetries the problem has.
- The hypercube {0,1}ⁿ is the Cayley graph of (ℤ/2)ⁿ, and its automorphism group is the hyperoctahedral group (ℤ/2)ⁿ ⋊ Sₙ.
- Standard bit mutation and uniform crossover commute with all of it.
- k-point crossover commutes with the translations x ↦ x ⊕ m but, among permutations of positions, only with reversal: it assumes that neighbouring positions belong together.
- On Sₙ, a mutation is a step of a random walk on a Cayley graph: swaps are transpositions, inversions reverse a segment (the 2-opt move of a tour), insertions move an element. The generators to choose are those under which the problem is smooth.
- The permutation crossovers each keep something different: CX every position, PMX a segment's positions, OX1 a segment and the relative order of the rest.

**Bounds.** A point that leaves a box is repaired, and the repair is part of the operator's law. Clipping puts mass on the walls, reflection folds it back, wrapping treats the box as a torus, and resampling redraws the coordinates that left. A search with an adaptive step size should rather search ℝⁿ through Hansen's linear–quadratic map onto the box (`BoxTransform`): a repaired sample against a wall lets the step grow without bound when the optimum lies on a corner, as the checks of `strategies` show.

**Heavy tails.** Standard bit mutation flips k bits with probability that falls exponentially in k. Heavy-tailed mutation draws the rate α/n from a power law, so that jumps of size k cost only polynomially. On Jump_k this is faster by a factor k^Θ(k) (Doerr et al. 2017).

## 2. Content

| Header | Content |
|---|---|
| `real.hpp` | `Real`, `Box` (`cube`, `unbounded`, `contains`), `Repair{clip, reflect, wrap, resample}`, `repair`, `uniform_point`; `gaussian_mutation`, `cauchy_mutation`, `polynomial_mutation` (bounded, Deb and Goyal 1996); `simulated_binary_crossover` (bounded, Deb and Agrawal 1995), `blend_crossover` (BLX-α), `intermediate_recombination`, `iso_line` (Vassiliades and Mouret 2018) |
| `box_transform.hpp` | `BoxTransform(box)`: Hansen's linear–quadratic map of ℝⁿ onto a box (pycma's `BoxConstraintsLinQuadTransformation`), `forward`, `inverse` | (moved here from `strategies`, where it was written for CMA-ES)
| `binary.hpp` | `Bits`, `random_bits`, `ones`; `standard_bit_mutation` (by geometric jumps), `standard_bit_mutation_positive`, `flip_exactly` (Floyd's sampling), `PowerLaw` and `heavy_tailed_mutation` (fmut_β) |
| `integer.hpp` | `Integers`, `IntegerBox`, `uniform_integers`; `reset_mutation`, `creep_mutation`, `geometric_mutation` with `rudolph_p` (Rudolph 1994) |
| `permutation.hpp` | `Permutation`, `identity_permutation`, `is_permutation`, `shuffle`, `random_permutation`, `inverse`; `swap_mutation`, `inversion_mutation`, `insertion_mutation`, `scramble_mutation`; `partially_mapped_crossover`, `order_crossover`, `cycle_crossover` |
| `generic.hpp` | `uniform_crossover`, `k_point_crossover`, `one_point_crossover`, `two_point_crossover` on any sequences |
| `distance.hpp` | `hamming`, `euclidean`, `kendall_distance` (Fenwick tree, O(n log n)), `cayley_distance` (n − cycles) |
| `variation.hpp` | all of the above |

## 3. Checks

`check_heuresis_variation` (33):
- Reflections and wraps against their closed forms.
- Standard bit mutation's number of flips against Bin(n, p), and each bit's frequency against p.
- The positive variant against the binomial conditioned on at least one flip.
- k flips exact on both of Floyd's branches, with uniform positions.
- The power law's normalisation, ratio and frequencies.
- Bit mutation and uniform crossover commuting exactly with x ↦ x ⊕ m under the same stream.
- k-point crossover giving complementary children with exactly k switches.
- SBX away from the bounds keeping the parents' sum to rounding. Bounded SBX and polynomial mutation staying in the box, BLX-α in its interval, iso+line without isotropic noise on the parents' line.
- Intermediate recombination exact. Gaussian mutation's mean and variance.
- Rudolph's mean ℓ¹ step equal to σ.
- Every permutation operator giving a permutation.
- CX keeping every position, PMX taking positions from its parents, OX1 keeping the other parent's cyclic order.
- Kendall's distance against counting discordant pairs, and Cayley's against breadth-first search on S₆.

## 4. Dependencies

`search` (the streams' types); calx `random`.

## 5. References

- Bentley, J. and Floyd, R. W. (1987), A sample of brilliance, *CACM* 30, 754–757.
- Biedrzycki, R., Arabas, J. and Jagodziński, D. (2019), Bound constraints handling in differential evolution: an experimental study, *Swarm Evol. Comput.* 50, 100453.
- Carvalho Pinto, E. and Doerr, C. (2018), A simple proof for the usefulness of crossover in black-box optimization, *PPSN XV*, 29–41.
- Davis, L. (1985), Applying adaptive algorithms to epistatic domains, *IJCAI* 1, 162–164.
- Deb, K. and Agrawal, R. B. (1995), Simulated binary crossover for continuous search space, *Complex Systems* 9, 115–148.
- Deb, K. and Goyal, M. (1996), A combined genetic adaptive search (GeneAS) for engineering design, *Comput. Sci. Informatics* 26, 30–45.
- Devroye, L. (1986), *Non-Uniform Random Variate Generation*, Springer.
- Diaconis, P. (1988), *Group Representations in Probability and Statistics*, IMS Lecture Notes 11.
- Doerr, B., Le, H. P., Makhmara, R. and Nguyen, T. D. (2017), Fast genetic algorithms, *GECCO '17*, 777–784.
- Eshelman, L. J. and Schaffer, J. D. (1993), Real-coded genetic algorithms and interval-schemata, *FOGA 2*, 187–202.
- Goldberg, D. E. and Lingle, R. (1985), Alleles, loci, and the travelling salesman problem, *ICGA 1*, 154–159.
- Oliver, I. M., Smith, D. J. and Holland, J. R. C. (1987), A study of permutation crossover operators on the TSP, *ICGA 2*, 224–230.
- Rudolph, G. (1994), An evolutionary algorithm for integer programming, *PPSN III*, 139–148.
- Syswerda, G. (1989), Uniform crossover in genetic algorithms, *ICGA 3*, 2–9.
- Vassiliades, V. and Mouret, J.-B. (2018), Discovering the elite hypervolume by leveraging interspecies correlation, *GECCO '18*, 149–156.
- Yao, X., Liu, Y. and Lin, G. (1999), Evolutionary programming made faster, *IEEE Trans. Evol. Comput.* 3, 82–102.

## 6. Open Questions

1. **Trees and graphs** as representations are in `programs` and planned in `graphs` (morphe's rewriting); whether a common interface for structured genotypes is worth having waits for the second of them (the infrastructure's rule of two).
2. **Mixed spaces.** Products of the spaces here (a design with real, integer and categorical parts) are composed by hand today; a product representation is open.
