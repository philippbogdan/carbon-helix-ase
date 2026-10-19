# From samples to plotted points

The four supplied inputs are exported from `results/summary.json`, research commit `ac806e6cbf4c314a89176771197e76047498fb34`. The summary is a saved result, not a fresh simulation. `scripts/figure_data.py --check` checks every input value, including all six CSV rows, and hashes the source files in `provenance.json`. Run it without `--check` to rebuild the pack. JSON spacing may change; values do not.

## headline.csv: error and extraction cost

`scripts/headline.py:rows` selects `value["helios|0.00079"]` and `value["helios|0.0003"]`, then these keys under `rows`:

- `C4-Helix|mode=ase|r4`
- `Carbon-Helix|full|r4`
- `Carbon-Helix|l2=on_event,l3=ambiguous,rot=8|r4`

It copies error, intervals, sampled share, status, gate/time means, peak qubits and counts. It computes preparation time from the protocol and effective rate as `-r*log(1-2*eps)/(2*(prep_s+r*round_s))`. These are four-round X memories under Helios-like noise. Both codes encode two logical qubits. The future point scales all noise components together.

Raw counts are `results/final/c4_strat_helios/*/*/*/*.json` and `results/final/ch_helios/*/*/*/*.json`, selected by code, rule, noise, X basis and four rounds. `analysis/strat.py:load` pools each configuration by fault count. `analysis/value.py:evaluate` fits the hybrid estimator, with order three for C4 and six for Carbon. It uses observed strata with at least three failures and fits the rest. The 95% intervals use 50 smoothed bootstrap refits, seed 3. The measured label requires 90% sampled share. `path_resources` uses 4000 directly sampled paths, seed 0, to estimate gates and time, excluding preparation. `./run.sh reproduce` repeats all six estimates and checks them against the summary.

Figure 1 plots `eps_LQ`, `ci_low`, `ci_high`, and `status`; its fold labels divide C4 by R8. The horizontal offsets separate overlapping points. Figure 2 uses `twoq_per_round` and `peak_qubits` with the level split below.

## circuits.json: level costs

This is the summary's complete `circuits` section. `analysis/make_all.py:sec_circuits` rebuilds it from `src/hase/` protocols, `analysis/resources.py:peak_live` and modelled timing. `scripts/figure_data.py --check` requires exact equality with that rebuilt record. No statistical samples are involved in the part counts.

Figure 2 stacks Carbon `L1_a.two_q`, `L2_all.two_q` and `L3.two_q`: 120 + 200 + 240 = 560. Peak qubits in the comparison CSV are the maximum over the sampled campaign's executed paths. The 98-qubit reference is the Helios capacity cited in the main source guide.

## distance.json: saved solver results

This is the summary's complete `distance` section. `analysis/make_all.py:sec_distance` aggregates `results/distance/*.json` and adds `results/gadget_gain.json`. The data check rebuilds this aggregation exactly; it does not rerun the searches.

`end__<design>__<rule>__<basis>__c*.json` records each logical class for restricted end-gadget mechanisms. For each case, the aggregation takes the minimum weight and lower bound, checks all solver statuses and retains the logical combinations. Figure 3 uses the 16 `k4cyc|R8_0` through `R8_7` X/Z cases and five other `k4cyc` cases. All three classes must be present to fill the beads. Quiet X lacks class `01` and stays open.

`design__*.json` supplies the alternative gadgets. `whole__*.json` supplies the unrestricted search's best weight, lower bound and elapsed time. Filled beads reach the lower bound; open beads reach the best found weight. These end-gadget results alone do not bound the whole R8 circuit. Saved decomposition proofs in `results/relaxed_distance/` are separate evidence, with scope in `EVIDENCE.md`.

## interval.json: escalation and schedule error

This is the complete summary `interval` section. Raw fault-count samples are `results/final/ch_interval/*/*/*/*.json`. `analysis/interval.py:models` pools them by rule, T and basis. `boot_all` uses the order-six hybrid estimator and 100 smoothed bootstrap refits, seed 0 plus 101 times each configuration's index in sorted order. The full baseline is shared. `study` divides per-round errors by that baseline and takes the 2.5th and 97.5th percentiles of bootstrap ratios. `ratio_driven` requires at least 50% sampled share in both estimates.

For family R, `T` denotes the rotation period (`rot=4` or `rot=8`), not a forced full-measurement cap. Figure 5 plots every `ratio[T]` and `ratio_ci95[T]` entry for B, C, R and P, marking untested T values. Within each T, p decreases from 0.005 to 0.0005. The study uses X memories with uniform noise, eight memory rounds except for periodic T = 16, which uses sixteen. `./run.sh verify` refits all families, errors, ratios, intervals and status classifications from the raw counts.

For Figure 4, `study` calls `analysis/strat.py:path_resources` on the B and C rules with no forced cap. Under uniform noise, it directly samples 8000 eight-round paths at each of seven p values, with RNG seed `[0, p_index, 9177]`, and subtracts the full preparation round. These path samples were not saved individually. Their parameters, deterministic sampler and summary `escalation_scan` values supply the reproducible source chain. `./run.sh verify` repeats both scans and checks every frequency. The figure's error bars are `1.96*sqrt(f*(1-f)/8000)`, treating shots as units. Its Helios line is a reference p, not an additional sampled point.

## Export

`../make_figures.py` reads only this pack. Plotted values are unchanged. It adds the documented ratios, offsets, bead counts and intervals, then exports PNG and SVG through `../_export/figure_export.py` and drops the PNG alpha channel after checking opacity. `../reproduction.json` records the clean-copy comparison with the supplied originals.
