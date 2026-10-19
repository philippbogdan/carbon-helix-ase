# Results you can check

R8 removes 61.46% of full Carbon's two-qubit gates per round and 50.97% of its modelled round time at p = 0.00079. Its effective memory error rate is 24.39x lower than adaptive C4's central estimate. Full Carbon already supplies nearly all that rate advantage: R8 improves its central rate by only 4.3%.

## Find the evidence

| Claim | Saved evidence | Recompute |
| --- | --- | --- |
| All table and figure values | `figures/data/headline.csv`, six rows | `./run.sh figures` |
| Source chain for all five figures | `figures/data/DATA.md`, `figures/data/provenance.json` | `scripts/figure_data.py --check` |
| Error fits and gate/time averages | `results/final/c4_strat_helios/` and `results/final/ch_helios/` | `./run.sh reproduce` |
| Sampling and decoding work | Fresh `build/check.json` and `build/samples_*.npz` | `./run.sh` |
| Supplied circuits rebuild exactly | `circuits/` | `./run.sh` |
| Halfway parity reads expose the cancelling fault pair | `build/check.json`, `halfway_parities` | `./run.sh` |

Every plotted row has a key such as `Carbon-Helix|l2=on_event,l3=ambiguous,rot=8|r4`. Look it up under `value`, then `helios|0.00079` or `helios|0.0003`, then `rows` in `results/summary.json`.

`./run.sh reproduce` pools the raw samples, refits the estimator, repeats the 50 bootstrap fits and resamples 4,000 executed paths per configuration and noise point. It compares the results with the saved values and fails on disagreement. It writes the fresh values to `build/reproduction.json`. It does not rerun the original rare-event simulation campaigns.

## What was estimated

All displayed memories encode two logical qubits. Each has one full preparation round followed by four memory rounds in the X basis. Error per logical qubit per round is half the block error per round. A block failure means at least one logical prediction is wrong.

The samples are stratified by total fault count W. For each stratum, the analysis estimates P(fail | W), then weights it by P(W) at the requested noise. It uses the observed failure fraction where at least three failures were seen. Elsewhere it uses a polynomial fit. C4 assumes a lowest order of three faults, Carbon six. A saved estimate is labelled measured when at least 90% of its value comes from the observed fractions. This does not mean every shot was sampled directly at the plotted p.

At p = 0.00079, the sampled shares are 99.1% for adaptive C4, 38.2% for full Carbon and 41.4% for R8. Their source campaigns contain 480,000, 64,000 and 64,000 shots respectively, with 163, 153 and 80 failures across the sampled fault counts. Those totals are not direct failure frequencies at the plotted noise.

The 95% intervals come from a smoothed parametric bootstrap with refitting. They are conditional on the chosen fit and do not include model or device uncertainty. In particular, low-noise fits do not independently demonstrate six-fault scaling for R8. Whole-circuit decomposition bounds for periodic and escalating rules do not cover rotations. Restricted end-gadget checks for R8 are shown separately in the distance figure. The bundled distance and decoder audits retain the evidence and known limits of the original run.

## Error per second

For per-logical-qubit error e, r memory rounds, preparation time t0 and round time t, the comparison CSV uses:

```text
rate = -r * log(1 - 2e) / (2 * (t0 + r*t))
```

Times are in seconds. Preparation takes 241.2 ms for C4 and 788.4 ms for Carbon. The headline exporter computes those times from the protocol rather than storing another input constant. Interval endpoints use the same transformation, holding time fixed. Round times and gate counts exclude preparation; the rate includes it.

A longer round stores the state for longer, so comparing memory rates by elapsed time is useful. These short experiments have different durations. This calculation is not a measured lifetime or a forecast of logical computation throughput.

## Porting changes the result

Helios-like timing and noise are in `src/hase/helios.py`. Idle errors, transport, leakage and measurement timing include approximations. Memory error during a compiled extraction round is an important unmeasured input. At p = 0.0003 every noise component is reduced together, which is a model sweep rather than a prediction for a device upgrade.

The result worth testing on another stack is the extraction saving at an acceptable logical error rate. Run both full and adaptive extraction with your compiled timing and noise. These memory estimates do not establish a logical-gate speedup.

## Circuit order and full-measurement spacing

The free-order fits support slopes near six at the fault densities sampled. These differ from the imposed order-six extrapolation used by the headline estimator. The circuit decomposition covers full, escalating and periodic schedules at the lengths listed in [the evidence guide](EVIDENCE.md), but does not prove a bound for R8. The same guide gives the six-fault window construction and the periodic ratios. `./run.sh verify` refits those results from raw counts and replays the constructions.
