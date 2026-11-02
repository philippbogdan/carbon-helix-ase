# What the comparison measures

Carbon-Helix R8 gives an estimated 24.4x reduction in memory error per second relative to adaptive C4 at $`p=0.00079`$. Its own advantage over full Carbon is extraction cost: 61.5% fewer two-qubit gates and 51.0% shorter rounds. Carbon's error estimates are partly extrapolated.

All rows are X-basis memories with one full preparation round, four subsequent rounds and two logical qubits, under the Helios-like model in [helios.py](src/hase/helios.py).

| Quantity | C4 adaptive | Carbon full | Carbon R8 |
| --- | ---: | ---: | ---: |
| Estimated error per logical qubit per second | 1.55e-4 | 6.61e-6 | 6.33e-6 |
| Conditional 95% interval, per second | 1.33e-4 to 1.73e-4 | 4.83e-6 to 1.01e-5 | 4.40e-6 to 1.05e-5 |
| Estimated error per logical qubit per round | 2.49e-5 | 6.51e-6 | 3.70e-6 |
| Two-qubit gates per round | 54.1 | 560 | 215.8 |
| Modelled round time, ms | 100.6 | 788.4 | 386.5 |
| Modelled preparation time, ms | 241.2 | 788.4 | 788.4 |
| Peak qubits | 52 | 92 | 92 |
| Share from observed fault strata | 99.1% | 38.2% | 41.4% |
| Stratified shots | 480,000 | 64,000 | 64,000 |
| Failures across sampled strata | 163 | 153 | 80 |

[headline.csv](figures/data/headline.csv) holds unrounded values. The failure totals are from biased fault-count sampling, so failures divided by shots are not the displayed error probabilities. Carbon uses 60 data qubits and 32 reused ancillas. R8's central error rate is 4.2% below full Carbon's, with almost entirely overlapping intervals; this does not establish a rate improvement.

## Definitions

A block failure means either logical prediction is wrong. The reported per-logical-qubit error $`e`$ is half the effective block error per round, rather than a separately measured marginal error of each logical qubit. For $`r=4`$ rounds, preparation time $`t_0`$ and round time $`t`$ in seconds:

```math
\lambda = \frac{-r\log(1-2e)}{2(t_0+rt)}.
```

The rate includes preparation; gate and round-time averages exclude it. Interval endpoints use the same conversion with time fixed. The ratio is 6.7x per round, rather than 24.4x per second, because the experiments last different lengths. Neither quantity establishes a long-time lifetime or a logical-gate speedup.

The [estimator](analysis/strat.py) weights failure probabilities conditioned on fault count by the model's fault-count distribution. It uses observed fractions for strata with at least three failures, a polynomial elsewhere, and minimum orders of six for Carbon and three for C4. A "measured" label requires at least 90% observed-stratum contribution. The 95% intervals use smoothed bootstrap refits and exclude fit misspecification and device uncertainty. [Methods](docs/methods.md) explain the conditioning approximation.

At $`p=0.0003`$, full Carbon and R8 have only 0.8% and 1.7% observed-stratum contribution. These almost entirely extrapolated rows scale every noise component together; they are a model sweep, not a device forecast.

## Device assumptions

Helios's memory error during a compiled extraction round is unpublished. This assumption changes the estimated error-per-second ratio most:

| Model variation | C4 / Carbon full, estimated | C4 / Carbon R4, estimated |
| --- | ---: | ---: |
| Double idle-error rate | 4.8x | 5.1x |
| Halve idle-error rate | 63.1x, heavily extrapolated | 330.1x, heavily extrapolated |
| Measurement/reset batches: 4 ms instead of 2 ms | 30.6x | 24.1x |

These are [saved estimates](figures/data/sensitivity.json) comparing adaptive C4 with full Carbon and R4; R8 was not tested. At half idle error, only about 1% of either Carbon estimate comes from observed strata. Every ratio includes preparation and uses its variant's [timing model](src/hase/timing.py). With longer measurement/reset batches, rounds take 828.4 ms for full Carbon, 528.9 ms for R4, 253.2 ms for full C4 and 105.5 ms for adaptive C4; preparation uses the corresponding full-extraction time.

There is a separate timing/noise approximation: simultaneously needed block checks take longer than the individual-block noise schedules. The [timing audit](results/timing_consistency.json) estimates 7.90 ms of missing idle exposure per R8 round at the headline noise. This is an exposure estimate, not a correction or bound on logical error.

Transport and leakage are approximated by Pauli noise; compilation, leakage dynamics and hardware execution remain untested. The [evidence guide](EVIDENCE.md) separates circuit bounds, decoder errors and statistical checks. [REPRODUCE.md](REPRODUCE.md) maps each calculation to its command.
