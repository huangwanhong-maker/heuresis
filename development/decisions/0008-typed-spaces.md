# 0008. Typed Spaces

- **Status:** Accepted, provisionally (2026-11-20)
- **Date:** 2026-11-20
- **Decided by:** Claude, at the infrastructure's request for a universal evolutionary framework for allocating heterogeneous resources, in which "the representative algebraic space of the resource" can be declared (ℝᴺ, a set, a manifold), after its survey *Evolutionary heterogeneous resource allocation* (Part III) and the design `development/ideas/2026-11-20-evolutionary-allocation-of-heterogeneous-resources.md` (its Stage 1).

## 1. Context

heuresis has representations and kernels for real boxes, bit strings, integer boxes, permutations and programs (`variation`, `programs`), each with its own genotype type, and an evolutionary loop over any copyable genotype (`evolution`, decision 0006). An allocation of heterogeneous resources needs a genotype made of many kinds at once (shares of a divisible pool, units among recipients, an order of grants, a choice of sites, an orientation), whose operators respect each kind's geometry and compose. The survey's Part III found the theory for it in Moraglio's geometric framework: operators defined by a metric, and the product theorem, which makes a product's part-wise crossover geometric for the weighted sum of distances. It found the hazard too: parts of different widths, summed without a unit, let the widest part decide.

## 2. Decision

A module `spaces`:

- **One value tree** for every genotype (real and integer vectors, categories, permutations, sets, sequences, records, choices, points of manifolds), with an exact key, so that `evolution` runs on every space through one adapter.
- **A contract of a space.** Required: sample, mutate within a ball of the metric at a standardised scale, distance, a declared unit (its mean or root-mean-square distance, or its diameter, said which, and whether exact), validity. Optional, each declared present or absent: recombination with whether it is geometric, pure and respectful; repair; a chart (log, retract, trust radius); a standardised embedding; a floor on moving; a bound on distances; the elements of a finite space.
- **Base spaces:** real and integer boxes, multisets, categories, ordered levels, the simplex in Aitchison's geometry (tyche's ilr coordinates), permutations under Cayley's, Hamming's or Kendall's metric, subsets with bounded size, compositions of identical units with caps, sequences of any space's values under an edit distance, the sphere and SO(3) (physis's rotations). Every unit is a closed form where one exists (table in the module's README), else an estimate the space says is one.
- **Constructors:** the product (distance Σ wᵢ dᵢ/uᵢ; recombination part by part), the choice (a jump across branches large enough for the triangle inequality, truncation said to cost geometry where a branch is unbounded), the constrained subspace (retry within a bound and count; repair if declared).
- **Laws for every space,** run by one function on any space, reported and not refused.
- **Diagnostics of a representation** through a decoder (Raidl and Gottlieb's measures), fitness–distance correlation, and χ² uniformity of samplers, given size if asked.

No dependency changes: physis and tyche were heuresis's already.

## 3. Alternatives Considered

- **A genotype type per space, composed by templates.** Faster, and checked by the compiler; but a product's type would grow with every part, and a genotype declared at run time (by a domain's data, as an allocation problem is) could not be built. One value tree, read by its space, serves both.
- **Normalising every space into ℝᵏ** (random keys) and searching there. General and simple, but the survey's Part I found redundancy and lost locality in decoders; the checks show it (random keys leave the order unchanged in nine small mutations of ten). Keys remain available as each space's embedding, for CMA-ES and surrogates.
- **Ranges of coordinates as units.** Easier to state, but not what a typical distance is; the mean distance makes a part's weight mean what it says (Σwᵢ is then the product's own unit, exactly).

## 4. Consequences

- 83 checks in `check_heuresis_spaces`; heuresis 675.
- The allocation domain's planned `evolution` (the design's Stage 5) assembles its genotype from resources' spaces; kategoria's `resources` (its decision 0004) says what a resource is, and a resource's carrier space is a space here.
- **Not yet:** graphs; the dependent product; transport and bases on charts; shrinkers; joint covariance across spaces with no common embedding.
