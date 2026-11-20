# sequential: Orders of Grants

*Status: draft (2026-11-20; [decision 0009](../development/decisions/0009-orders-of-grants.md)).*

## 1. Concepts

When resources are given out one grant at a time, the order can matter: a stock runs out, a price rises after each sale, a budget binds. An allocation made so is a **word** over an alphabet of grants acting partially on a state, a **grant system**. Where two grants commute, their order does not matter, and a search that treats their two orders as different wastes its evaluations; where they do not, the order is part of what is chosen.

Two grants **commute** at a state when, applied in either order and each skipped where it is not allowed, they reach the same state, and either both orders grant both or neither does. Grants competing for the last unit of a stock do not commute, though both orders are infeasible: one order gives the unit to the first, the other to the second. Independence is known two ways, kept apart: **declared**, from what each grant reads and writes (morphe's footprints), and **tested**, by trying both orders at states reached by random walks. A declared independence that a test refutes is reported with the word that reached the witness state.

Words that differ only by exchanging adjacent commuting grants form one **trace** (Mazurkiewicz). A search keys its evaluations by the trace of the word granted (morphe's lexicographic normal form), so each trace costs one evaluation.

A word is read as an allocation by a **decoder**, and every decoder says what it can reach, since a decoder can exclude every optimum. Variation of orders is chosen by the **feature** the objective depends on: precedence, adjacency or position (Starkweather et al. 1991). Where outcomes are random and observed as they happen, the **adaptivity gap** says whether a policy could do better than a fixed sequence.

## 2. Content

| Header | What it holds |
|---|---|
| `grants.hpp` | `GrantSystem<S>` (the partial action, optional footprints, a step of time); `run`, `feasible`; `commute_at` and `independence` (declared and tested, witnesses, contradictions); decoders `serial` (skip, defer, stop) and `stepped` (with an optional separator letter), each with its `reach` |
| `canonical.hpp` | `trace_key`; `keyed_by_trace`, the evolutionary loop's operators keyed by the trace of the decoded word; `TraceCache`; `redundancy` and `redundancy_of_orders` |
| `operators.hpp` | crossovers by feature: precedence-preserving (PPX), order-based (OX2), position-based (POS), non-wrapping order (NWOX), edge recombination with shared edges kept; feature distances; what each keeps, as predicates; `order_operators` for the evolutionary loop; `feature_probe` |
| `words.hpp` | `WordSpace`: words with fixed letter counts (picking sequences, serial orders with quotas), a space of `spaces` under the least number of adjacent exchanges, with its unit in closed form |
| `mallows.hpp` | Mallows's laws under Kendall's (repeated insertion), Cayley's (Ewens by the Chinese restaurant) and Hamming's (fixed points, then a derangement) metrics, sampled exactly, with their normalisers and expected distances; `fit_mallows` (Borda centre, dispersion by its expected distance) |
| `adaptivity.hpp` | `adaptivity_bracket`: the best common sequence and clairvoyance over scenarios, which bracket the best policy |
| `policies.hpp` | Policies against sequences ([decision 0011](../development/decisions/0011-calibrated-surrogates-and-policies-against-sequences.md)): `Instance<S>`; `follow` (the allowed grant of greatest score, step by step); `linear`, `program`; `worth`, `mean_worth`; `train_linear` (CMA-ES), `train_program` (genetic programming); `search_sequence` (CMA-ES over random keys, the deferring rule), `best_order` (every order of a small instance) |
| `sequential.hpp` | all of the above but `policies.hpp`, which brings in CMA-ES and genetic programming and is included by name |

### 2.1 Decoders and Their Reach

| Decoder | Outputs | Reach |
|---|---|---|
| serial, skipping | feasible words | every maximal feasible word; an output need not be maximal |
| serial, deferring | feasible, maximal words | exactly the maximal feasible words of distinct grants, over all orders |
| serial, stopping | feasible prefixes | the feasible prefixes of the word |
| stepped | grants by step of time | never waits while a grant is allowed; with a separator letter, every timed plan |

### 2.2 What Each Crossover Keeps

| Operator | Kept exactly |
|---|---|
| PPX | every precedence both parents share |
| edge recombination (shared edges kept) | every edge both parents share |
| cycle crossover, PMX (`variation`) | every position both parents share |
| OX2, POS | items they move appear in the second parent's order |
| NWOX | a segment of the second parent at its own positions, the first parent's order around it |
| a point of a bubble-sort path (`spaces`) | the segment of Kendall's metric |

## 3. Checks

`tests/check_sequential.cpp`, 48 checks:

- commutator tests equal exact commutation over every reachable state of a system of stocks and holdings; a rising price makes order matter, with its witness; a footprint that forgets the price is reported as a contradiction;
- on 100 random systems whose grants consume and produce, against brute force: every output feasible; the deferring decoder's outputs over all orders exactly the maximal feasible words; the skipping decoder reaching every maximal word and stopping short of one on 1,817 of 12,000 orders; the stepped decoder unable to wait for a falling price without a separator, and able to with one;
- six grants on three stocks: 720 orders in 8 traces of 90 words each, as morphe's linearisations count; a cache and an evolution keyed by trace evaluate each trace once;
- each crossover keeps what it claims on 500 pairs of related parents;
- Starkweather's effect: on a depletion system (weighted completion from a stock that runs out) the probe finds precedence and precedence-keeping operators reach a mean best of 15,194 against edge recombination's 16,281; on setups between consecutive items the probe finds adjacency, and edge recombination reaches 5.90 against 8.30;
- Mallows's laws: probabilities summing to one, expected distances as enumerated, samples by χ² against the law under each metric; a law fitted to 5,000 of its samples recovers its centre and its dispersion within 0.001;
- words with fixed letter counts: every law of `spaces`; the distance as breadth-first search finds it on all 900 pairs; the unit as the mean over every pair; an evolution of a picking sequence finds the brute-force optimum of the least utility;
- on 50 random stochastic knapsacks, best sequence ≤ best adaptive policy (by dynamic programming) ≤ clairvoyance, and the adaptivity gap at most 4;
- refusals.

`tests/check_policies.cpp`, 10 checks: following a policy decodes a feasible word; a constant score follows the index order; a threshold grants nothing; on knapsacks whose price rises with every grant, no policy or searched sequence above each held-out instance's enumerated best; sequences searched per instance reaching 93.5% to 98.9% of the best with 10 to 300 evaluations, a linear policy learned once 95.5% and a program 95.3% on instances they never saw, generalising within half a per cent of their training share: the policy beaten by a sequence searched with 30 evaluations per instance, and beating one searched with 10.

## 4. Dependencies

heuresis's `spaces` (values, the space contract, permutations under Kendall's metric), `variation` (permutations and their operators) and `evolution` (the loop); morphe's `formal` (traces, their normal forms: its decision 0021); calx's random streams.

## 5. References

- Mazurkiewicz, A. (1977). Concurrent program schemes and their interpretations. DAIMI PB-78. Diekert, V., Rozenberg, G. (eds.) (1995). *The Book of Traces.* World Scientific.
- Bednarczyk, M. A. (1987). *Categories of Asynchronous Systems.* PhD thesis, University of Sussex.
- Kolisch, R. (1996). Serial and parallel resource-constrained project scheduling methods revisited. *European Journal of Operational Research* 90, 320–333.
- Starkweather, T., McDaniel, S., Mathias, K., Whitley, D., Whitley, C. (1991). A comparison of genetic sequencing operators. *ICGA*, 69–76.
- Whitley, D., Starkweather, T., Fuquay, D. (1989). Scheduling problems and traveling salesmen: the genetic edge recombination operator. *ICGA*, 133–140.
- Syswerda, G. (1991). Schedule optimization using genetic algorithms. In Davis, L. (ed.), *Handbook of Genetic Algorithms*, 332–349.
- Bierwirth, C., Mattfeld, D. C., Kopfer, H. (1996). On permutation representations for scheduling problems. *PPSN IV*, 310–318.
- Cicirello, V. A. (2006). Non-wrapping order crossover: an order preserving crossover operator that respects absolute position. *GECCO*, 1125–1132.
- Mallows, C. L. (1957). Non-null ranking models. *Biometrika* 44, 114–130. Doignon, J.-P., Pekeč, A., Regenwetter, M. (2004). The repeated insertion model for rankings. *Psychometrika* 69, 33–54. Arratia, R., Barbour, A. D., Tavaré, S. (2003). *Logarithmic Combinatorial Structures.* EMS. Ceberio, J., Irurozki, E., Mendiburu, A., Lozano, J. A. (2014). A distance-based ranking model estimation of distribution algorithm for the flowshop scheduling problem. *Progress in Artificial Intelligence* 2, 103–117.
- Diaconis, P. (1988). *Group Representations in Probability and Statistics.* IMS.
- Dean, B. C., Goemans, M. X., Vondrák, J. (2008). Approximating the stochastic knapsack problem: the benefit of adaptivity. *Mathematics of Operations Research* 33(4), 945–964.

## 6. Open Questions

- **Not yet:** policies as priority functions evolved as heuresis's programs or proposed by a model; the biased random-key GA; Plackett–Luce beside Mallows behind one interface (heuresis's `distributions` holds Plackett–Luce); mechanisms run on the words (the allocation domain and nomos run them; here they are genotypes); independence decided exactly for finite systems is kategoria's `resources`, not called here, since heuresis does not depend on kategoria.
