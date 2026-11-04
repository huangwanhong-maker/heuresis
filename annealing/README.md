# annealing: Searches by the Metropolis Rule

*Status: draft (2026-11-05): the Metropolis rule and its schedules (constant, geometric, logarithmic, modified Lam); simulated annealing, parallel tempering and population annealing on the ask–tell protocol, the last with its estimate of the free energy ([decision 0004](../development/decisions/0004-the-first-modules.md)).* Adaptive tempering ladders, weighted averages over families and annealing on devices that sample a Boltzmann law (decision 0003 §2.8) are planned.

## 1. Concepts

A proposal y from the state x, drawn by a symmetric kernel, is accepted with probability min(1, exp(−ΔE/T)) (Metropolis et al. 1953). The chain satisfies detailed balance with respect to the Boltzmann law π_T(x) ∝ exp(−E(x)/T), its stationary law on a finite space when the kernel is irreducible. Annealing lowers T so that the law concentrates on the minima (Kirkpatrick, Gelatt and Vecchi 1983). Only a logarithmic schedule T_k = c / ln(k + k₀) is guaranteed to converge to the global minima in probability, exactly when c is at least the greatest depth of a non-global local minimum (Hajek 1988); geometric and adaptive schedules are faster and carry no such guarantee. The modified Lam schedule adapts T step by step so that the acceptance rate follows a target, 0.44 through the middle of the run (Lam and Delosme 1988; Swartz 1993; Boyan 1998).

The energy is the first objective. A failure has energy +∞, and so does an infeasible point unless a finite penalty coefficient is given: the rule needs a number, and which number is the caller's choice.

Parallel tempering runs R chains at increasing temperatures and lets neighbours exchange their states with probability min(1, exp((βᵢ − β_{i+1})(Eᵢ − E_{i+1}))), which keeps the product of the Boltzmann laws stationary. The hot chains cross barriers and hand their states down to the cold ones (Swendsen and Wang 1986; Geyer 1991; Hukushima and Nemoto 1996).

Population annealing carries a population through increasing inverse temperatures. At each step it reweights the population by exp(−Δβ E), resamples it and equilibrates it by Metropolis sweeps. The mean weight at each step estimates the ratio of successive partition functions without bias, so the run estimates ln Z and the free energy −ln Z / β (Hukushima and Iba 2003; Machta 2010). Its R proposals at each step are evaluated together, so it spreads over workers as a single chain cannot.

## 2. Content

| Header | Content |
|---|---|
| `schedule.hpp` | `Schedule::constant(T)`, `geometric(T₀, α, steps per level)`, `logarithmic(c, k₀)`, `modified_lam(T₀, steps)`; `temperature()`, `advance(accepted)`, `acceptance()`, `lam_target(f)` |
| `metropolis.hpp` | `energy(evaluation, penalty)`; `metropolis(current, proposed, T, u)` |
| `simulated_annealing.hpp` | `SimulatedAnnealing<G>(sampler, proposal, schedule, draws, Options{penalty, min_temperature})`: one candidate at a time (breadth 1), the Metropolis rule at each tell; `state()`, `current_energy()`, `accepted()`; injectable and migratory |
| `tempering.hpp` | `ParallelTempering<G>(sampler, proposal, temperatures, draws, penalty)`: one generation of R proposals, then even or odd exchanges; `states()`, `energies()`, `exchange_rate(i)` |
| `population_annealing.hpp` | `PopulationAnnealing<G>(sampler, proposal, Options{population, betas, sweeps, penalty}, draws)`: reweighting, stochastic universal resampling, sweeps; `log_partition_ratio()`, `log_partition(ln Z₀)`, `mean_energy()`, `beta()` |

## 3. Checks

`check_heuresis_annealing` (19):

- The schedules against their closed forms, and the Metropolis rule's edge cases (downhill always; infinite energy never accepted from a finite one, and always left).
- **The Boltzmann law, exactly.** On a QUBO of 8 bits at T = 2, a million steps of the chain are within total variation 0.008 of the Boltzmann law computed by enumerating the 256 states.
- The ask–tell chain equals the Metropolis chain driven by hand with the same streams, step by step.
- Parallel tempering on a QUBO of 6 bits at four temperatures: each temperature's 250,000 states are within total variation 0.006 of its own Boltzmann law; exchanges are accepted between every pair of neighbours; on a pool of threads, with generations cut into parts, the run is the same.
- **Population annealing's free energy.** On a QUBO of 10 bits carried from β = 0 to 1.5 in 30 steps with 2000 states, four independent runs estimate ln Z with a mean error of 0.48 standard errors from the exact value by enumeration. Each final population's mean energy is within 0.023 of the exact Boltzmann mean.
- Geometric annealing finds the ground state of a 16-bit QUBO, found by enumerating its 65,536 states; the same seed gives the same run. The modified Lam schedule holds the acceptance at 0.4395 from 20% to 60% of the run, against its target 0.44, and ends at the ground state.

## 4. Dependencies

heuresis `search`, `selection` (stochastic universal sampling for resampling); calx `random`; in the checks, heuresis `benchmarks` (QUBO) and `variation` (bit flips).

## 5. References

- Boyan, J. A. (1998), *Learning Evaluation Functions for Global Optimization*, PhD thesis, Carnegie Mellon University.
- Geyer, C. J. (1991), Markov chain Monte Carlo maximum likelihood, *Computing Science and Statistics* 23, 156–163.
- Hajek, B. (1988), Cooling schedules for optimal annealing, *Math. Oper. Res.* 13, 311–329.
- Hastings, W. K. (1970), Monte Carlo sampling methods using Markov chains and their applications, *Biometrika* 57, 97–109.
- Hukushima, K. and Iba, Y. (2003), Population annealing and its application to a spin glass, *AIP Conf. Proc.* 690, 200–206.
- Hukushima, K. and Nemoto, K. (1996), Exchange Monte Carlo method and application to spin glass simulations, *J. Phys. Soc. Japan* 65, 1604–1608.
- Kirkpatrick, S., Gelatt, C. D. and Vecchi, M. P. (1983), Optimization by simulated annealing, *Science* 220, 671–680.
- Lam, J. and Delosme, J.-M. (1988), *Performance of a new annealing schedule*, Design Automation Conference, 306–311.
- Machta, J. (2010), Population annealing with weighted averages: a Monte Carlo method for rough free-energy landscapes, *Phys. Rev. E* 82, 026704.
- Metropolis, N., Rosenbluth, A. W., Rosenbluth, M. N., Teller, A. H. and Teller, E. (1953), Equation of state calculations by fast computing machines, *J. Chem. Phys.* 21, 1087–1092.
- Swartz, W. P. (1993), *Automatic Synthesis and Placement of Standard Cells*, PhD thesis, Yale University.
- Swendsen, R. H. and Wang, J.-S. (1986), Replica Monte Carlo simulation of spin-glasses, *Phys. Rev. Lett.* 57, 2607–2609.

## 6. Open Questions

1. **Tempering ladders.** The temperatures are the caller's; ladders adapted to equalise exchange rates, or to maximise round trips (Katzgraber, Trebst, Huse and Troyer 2006), are planned.
2. **Hajek's constant.** The depth d* that makes the logarithmic schedule convergent can be computed exactly on small spaces (a minimax path problem over the neighbourhood graph); a check of convergence in probability needs many long chains and is not yet made.
3. **Devices.** Annealers and probabilistic-bit machines sample Boltzmann laws of QUBOs in hardware; as samplers (decision 0003 §2.8) they would be checked against these chains' laws.
4. **Physis.** The spin models of physis's statistical mechanics are energies of this kind; the Metropolis kernel here is the general one, which a model's own (Glauber, cluster) updates may replace.
