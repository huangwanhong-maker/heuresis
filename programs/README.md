# programs: Programs as Terms, Typed by Dimension

*Status: draft (2026-11-05): terms of a signature typed by kinds and physical dimensions, their generation, variation and interpretation; symbolic regression; genetic programming on the ask–tell protocol; derivations of context-free grammars and grammatical evolution ([decision 0004](../development/decisions/0004-the-first-modules.md)).* Local fitting of constants, linear scaling, semantic operators and graph genotypes are planned.

## 1. Concepts

A program is a term of a many-sorted signature: function symbols with sorted arguments and results, constants and variables of given sorts, so that the programs of a sort are a carrier of the term algebra (Goguen, Thatcher, Wagner and Wright 1977). Genetic programming searches that carrier by varying terms (Koza 1992); strongly typed genetic programming keeps every term well-sorted (Montana 1995).

Here a sort is a kind (quantities, truth values, others the caller declares) and, for a quantity, a dimension in the free abelian group ℤ⁷ over the SI base units (physis's `Dimension`). The arithmetic symbols are polymorphic in the dimension: addition needs equal dimensions, a product adds them, a quotient subtracts them, a power p/q multiplies them by p/q where that is integral, and sin, exp and log take and give dimensionless quantities. The dimension of a term is therefore computed by a partial homomorphism from its subterms, and a term is dimensionally consistent exactly when that homomorphism is defined on it. Such a term is covariant under a change of units: scaling the base units scales its value by λ^{dim t}, the scaling behind Buckingham's Π theorem. Symbolic regression typed this way generates only formulas that could be physical laws, and the space it searches is far smaller: on the checks' data, d = ½ a t² is found in the initial population on every seed, where the untyped search finds it on three seeds of ten.

Generation needs to know which sorts can be reached: G₀ is the terminals' sorts, G_d adds the sorts of every application to sorts of G_{d−1}, with exponents bounded so that the sets are finite and reach a fixpoint. To grow a term of sort s within depth d, only heads whose argument sorts lie in G_{d−1} are chosen, so generation never fails on a sort of G_d, and every term it makes is well-sorted. Variation replaces a subterm by another of the same sort (subtree crossover, subtree, point, hoist and shrink mutation), so it keeps every node's sort. Depth and size limits and lexicographic parsimony pressure (Luke and Panait 2002) restrain the growth of terms.

A context-free grammar gives another space of programs, its language. A derivation tree may be searched directly, crossover exchanging subtrees rooted at one nonterminal and mutation regrowing one (Whigham 1995), and every yield is a word of the language. Or a sequence of codons may be read through the leftmost derivation, each codon choosing a rule modulo the number of rules of the leftmost nonterminal, wrapping when the codons run out (grammatical evolution; Ryan, Collins and O'Neill 1998; O'Neill and Ryan 2001). The second map is many-to-one, so the codon space has large neutral networks.

## 2. Content

| Header | Content |
|---|---|
| `signature.hpp` | `Sort{kind, dimension}`; `Rule{fixed, same, product, quotient, power}`; `Primitive`, `Variable`, `Constant` (fixed or ephemeral); `Signature`: symbols numbered primitives, variables, constants; `synthesise(f, argument sorts)` (the homomorphism), `generable(d)` and `generable(sort, d)` (the sets G_d, cached), `argument_options(f, target, d)`, `terminals(sort)`; `library::` add, subtract, multiply, divide, square, cube, root, sine, cosine, exponential, logarithm |
| `trees.hpp` | `Node{symbol, value, sort}`, `Tree` (prefix order); `subtree_end`, `node_depths`, `depth`, `well_sorted`, `interpret`, `to_string` |
| `generation.hpp` | `generate(sig, sort, d, full, stream)`, `grow`, `full`; `ramp(k, d_min, d_max)` and `ramped(...)`, ramped half-and-half (Koza 1992, §6.2) |
| `operators.hpp` | `Limits{depth, size, mutation_depth}`; `subtree_crossover` (90/10 rule), `subtree_mutation`, `point_mutation`, `hoist_mutation`, `shrink_mutation`; `lexicographic_parsimony(evaluations, sizes)` |
| `regression.hpp` | `Cases{inputs, targets}`; `case_errors`; `regression(sig, tree, cases)`: the mean squared error, with the errors case by case as descriptors; a failure where a value is not finite |
| `genetic_programming.hpp` | `GpOptions`; `GeneticProgramming`: generational, ask–tell, elitist, parents by tournament (under parsimony) or ε-lexicase on the descriptors; injectable and migratory |
| `grammar.hpp` | `DerivationNode`, `Derivation`; `GrammarSpace(grammar)`: least depths, `grow`, `levels`, `depth`, `yield`, `crossover`, `mutate`; `map_codons(grammar, codons, max_wraps)` → `Mapping{valid, word, derivation, codons_read, wraps}` |

## 3. Checks

`check_heuresis_programs` (27):

- 3000 terms generated for seven sorts (lengths, velocities, accelerations, forces, areas, m kg⁻¹ and dimensionless) by grow and full: every one well-sorted, of its sort, within its depth; the square cannot head a term of dimension m and √ cannot take one.
- **Covariance under a change of units, bit for bit.** For 2346 generated terms whose value is finite, halving the kilogram, quartering the metre and doubling the second (powers of two, so every rounding commutes with the scaling) scales the value by exactly 2^{e_kg + 2e_m − e_s} of the term's dimension.
- Ramped half-and-half over depths 2 to 6: 120 terms at each depth; every full term a complete binary tree of size 2^{d+1} − 1; every grown term within its depth.
- Crossover and the four mutations keep every term well-sorted and of its sort over 3000 trials; a child beyond the limits is its parent; point mutation keeps the shape.
- Lexicographic parsimony orders by fitness, then by size.
- Grammatical evolution on E → E O E | V, O → + | *, V → x | y: the codons (4, 5, 3, 2, 9) map to y + x, reading six codons with one wrap, along the leftmost derivation worked by hand; without wrapping they map to no word.
- Derivations grown, crossed and mutated, and the valid mappings of random codons (685 of 1000), all yield words that morphe's Earley parser accepts.
- **Exact rediscovery.** From 20 cases of (a, t, d) with a in m s⁻², t in s and d in m, genetic programming typed by dimension finds d = ½ a t² exactly, and on ten seeds it finds it every time within 20 generations, where the same search without dimensions finds it three times (3000 evaluations against 51571). From 30 cases of (m, v, g, h), with initial terms too shallow to hold the law, it breeds E = ½ m v² + m g h exactly, of the sort of an energy, in generation 6.
- Telling every generation in reverse gives the same populations.

## 4. Dependencies

heuresis `search`, `variation`, `selection`; physis `units` (dimensions); morphe `formal` (grammars; Earley's parser in the checks); calx `random`.

## 5. References

- Fortin, F.-A., De Rainville, F.-M., Gardner, M.-A., Parizeau, M. and Gagné, C. (2012), DEAP: evolutionary algorithms made easy, *JMLR* 13, 2171–2175.
- Goguen, J. A., Thatcher, J. W., Wagner, E. G. and Wright, J. B. (1977), Initial algebra semantics and continuous algebras, *J. ACM* 24, 68–95.
- Kinnear, K. E. (1994), Fitness landscapes and difficulty in genetic programming, *IEEE World Congress on Computational Intelligence*, 142–147.
- Koza, J. R. (1992), *Genetic Programming: On the Programming of Computers by Means of Natural Selection*, MIT Press.
- La Cava, W., Spector, L. and Danai, K. (2016), Epsilon-lexicase selection for regression, *GECCO*, 741–748.
- Luke, S. and Panait, L. (2002), Lexicographic parsimony pressure, *GECCO*, 829–836.
- Montana, D. J. (1995), Strongly typed genetic programming, *Evol. Comput.* 3, 199–230.
- O'Neill, M. and Ryan, C. (2001), Grammatical evolution, *IEEE Trans. Evol. Comput.* 5, 349–358.
- Poli, R. and Langdon, W. B. (1998), Schema theory for genetic programming with one-point crossover and point mutation, *Evol. Comput.* 6, 231–252.
- Ryan, C., Collins, J. J. and O'Neill, M. (1998), Grammatical evolution: evolving programs for an arbitrary language, *EuroGP*, LNCS 1391, 83–96.
- Whigham, P. A. (1995), Grammatically-based genetic programming, *Workshop on Genetic Programming: From Theory to Real-World Applications*, ML'95, 33–41.
- Barenblatt, G. I. (1996), *Scaling, Self-Similarity, and Intermediate Asymptotics*, Cambridge.

## 6. Open Questions

1. **Constants.** Ephemeral constants are drawn and mutated, never fitted. Fitting them locally (by calx's Nelder–Mead or BFGS) or by linear scaling (Keijzer 2003, a least-squares slope and intercept, which tyche's `descriptive` holds) needs a library this module does not yet link; with dimensions, the fitted slope and intercept carry dimensions of their own.
2. **Bloat.** Lexicographic parsimony slows growth but does not remove redundancy: the energy law above was bred in a form many times the size of ½ m v² + m g h. Simplification by algebraic rewriting (the term algebra modulo the field axioms) would belong to morphe.
3. **Typed grammars.** Dimensions could be carried as attributes of a grammar (an attribute grammar whose synthesised attribute is the dimension), joining the two halves of this module.
4. **Semantic variation and graph genotypes** (morphe's double-pushout rewriting) are planned.
