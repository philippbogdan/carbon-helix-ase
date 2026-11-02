# Fault-count samples

`ch_helios/` and `c4_strat_helios/` supply the headline. Their uniform-noise counterparts are `ch_main/` and `c4_strat_uniform/`. `ch_interval/` varies full-check spacing, `ch_direct/` checks the estimator with direct sampling, and the `*_helios_sens/` folders vary device assumptions.

`c4_repro/` checks full-extraction C4 against the authors' Fig. 11a values: uniform noise, 12 memory rounds, 100,000 stratified shots per basis, with 392 X and 113 Z failures across strata. Decoder priors use $`p=0.0009`$; the estimator evaluates three noise points. `./run.sh baseline` refits the [comparison](../../EVIDENCE.md#c4-helix-baseline); `campaigns.py commands c4_repro` prints its sampling commands.

Files record the protocol, rule, noise, basis, rounds, seed, shot counts and failures. Stratified files group by total fault count; `analysis/strat.py` pools them and reweights to the requested noise. Do not divide their pooled failures by shots to obtain a headline rate. [The campaign registry](../../scripts/campaigns.py) prints the exact sampling commands; [RESULTS.md](../../RESULTS.md) explains the fit.
