# quantum: Quantum and Quantum-Inspired Search

*Status: draft (2026-11-05): the quantum-inspired evolutionary algorithm of Han and Kim, shown to be a search distribution of the product-Bernoulli family; Grover adaptive search (Dürr and Høyer; Baritompa, Bulger and Wood) on any device that keeps the law of amplitude amplification, with that law's exact emulation as the reference ([decision 0004](../development/decisions/0004-the-first-modules.md)).* Variational quantum algorithms (QAOA, whose outer loop is any optimiser of the library) and devices of registers and gates are planned, the latter in physis's `quantum`.

## 1. Concepts

**Quantum-inspired algorithms are search distributions.** Han and Kim's algorithm keeps each individual as Q-bits cos φᵢ|0⟩ + sin φᵢ|1⟩, observes them, and turns their angles towards the best strings seen by a lookup table. The register is a product state, and measuring it gives the product Bernoulli law with pᵢ = sin² φᵢ; no entangling gate is ever applied, so the algorithm never leaves that family. It is an estimation-of-distribution algorithm whose rule rotates angles instead of moving frequencies, and it needs no quantum device. (An entangled register's law is no product law: the Bell state has P(00)P(11) − P(01)P(10) = ¼, every product law 0.)

**Grover's law is all a search sees of a device.** With t of N points below a threshold, r Grover iterations leave the register in the plane of the uniform states on the marked and the unmarked points, so a measurement returns a marked point with probability sin²((2r + 1)θ), sin²θ = t/N, uniformly within each set. `EmulatedGrover` draws from that law exactly; to know the marked set it evaluates every point once, classically, a cost it reports apart from the oracle calls a device would spend. It emulates the law of a quantum computer, not a quantum computer, and it is the reference against which a device is tested, as calx's interpreter is for its backends ([decision 0003](../development/decisions/0003-the-search-process-and-its-seams.md) §2.8).

**Grover adaptive search** takes a uniformly measured point as its threshold, then repeatedly measures after r iterations, r uniform below m, and evaluates the outcome classically: an improvement becomes the threshold and resets m = 1, a failure multiplies m by λ = 6/5 up to √N. Dürr and Høyer bound the expected oracle calls before the minimum is held by (45/4)√N + (7/10) lg² N, against Θ(N) for any classical search. It is an asynchronous ask–tell algorithm offering one measured point at a time.

## 2. Content

| Header | Content |
|---|---|
| `inspired.hpp` | `QuantumInspired(n, QuantumInspiredOptions{individuals, table, global_period}, draws)`, generational, observing each Q-bit by the product family's comparison U < sin²φ from the keyed stream; `observation_probabilities` |
| `grover.hpp` | `Measurement`; the concept `GroverDevice` (`size()`, `measure(threshold, r, stream)`); `grover_success(t, N, r)`; `EmulatedGrover(N, f)` (`marked`, `measure`, `classical_evaluations`, `minimum`); `GroverAdaptiveSearch<D>(device, GroverSearchOptions{lambda, cutoff}, draws)` (`oracle_calls`, `measurements`); `dh_cutoff`, `dh_expected_bound` |

## 3. Checks

`check_heuresis_quantum` (15):
- a product register's measured law equal to the product Bernoulli law to 10⁻¹⁶; the Bell state's law not a product law;
- QIEA's observations equal, bit for bit, to the product family's reference draws from the same streams, after its angles have moved; QIEA solving OneMax (n = 40);
- the law against a direct simulation of the amplitudes on 64 points: P(marked) = sin²((2r + 1)θ) for r = 0..12 to 10⁻¹⁵, the amplitudes uniform within each set; the emulator's frequency against the law;
- Dürr and Høyer's bound on the mean oracle calls at N = 256, 1024, 4096 (about 20, 51 and 120, against bounds of 225, 430 and 821), growing as N^0.65 over that range (½ asymptotically; 1 for a classical search);
- Grover adaptive search reaching the minima of a QUBO and an NK landscape on 2¹⁰ points found by enumeration;
- the same search, unchanged, on a second device that simulates the amplitudes, through the seam.

## 4. Dependencies

heuresis `distributions` (and through it `search`, `variation`); calx `random`.

## 5. References

- Baritompa, W. P., Bulger, D. W. and Wood, G. R. (2005), Grover's quantum algorithm applied to global optimization, *SIAM J. Optim.* 15, 1170–1184.
- Boyer, M., Brassard, G., Høyer, P. and Tapp, A. (1998), Tight bounds on quantum searching, *Fortschr. Phys.* 46, 493–505.
- Dürr, C. and Høyer, P. (1996), A quantum algorithm for finding the minimum, arXiv:quant-ph/9607014.
- Grover, L. K. (1996), A fast quantum mechanical algorithm for database search, STOC, 212–219.
- Han, K.-H. and Kim, J.-H. (2002), Quantum-inspired evolutionary algorithm for a class of combinatorial optimization, *IEEE Trans. Evol. Comput.* 6, 580–593.

## 6. Open Questions

1. Registers, gates and measurement as such (statevector and density-matrix simulators, noise) belong to physis's `quantum`; when it holds them, a device built on them is a `GroverDevice` and a sampler of the distributions module, and its law is checked against `EmulatedGrover`.
2. Variational algorithms (QAOA, VQE) need nothing new here: their parameters are searched by `strategies` or `models` with an evaluator that runs the circuit. Which of them to state as problems, and where, is open.
3. Annealers and probabilistic-bit machines as samplers of Boltzmann laws, tested against `annealing`'s exact laws on small spaces.
4. The emulation enumerates the space (N classical evaluations); a law-preserving emulation for structured oracles (counting below a threshold without enumeration) would let it reach larger N.
