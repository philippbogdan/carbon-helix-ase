# Samples and saved evidence

The headline comparison uses 480,000 adaptive C4 shots, 64,000 full Carbon shots and 64,000 Carbon R8 shots. These sample different fault counts. Their failure totals cannot be divided by shots to obtain the plotted low-noise error rates.

`summary.json` is the saved numerical record from research commit `ac806e6cbf4c314a89176771197e76047498fb34`. Its generated prose is historical source text. The repository explanations are written separately.

| Input | What it supports |
| --- | --- |
| `final/c4_strat_helios/`, `final/ch_helios/` | Six headline estimates and their cost averages |
| `final/ch_interval/` | All four schedule families and deterministic escalation replay |
| `distance/` | Restricted searches, alternative gadgets and unfinished whole-circuit searches |
| `relaxed_distance/`, `inner_window/` | Decomposition proofs and finite-window checks |
| `minimal/`, `redecode/` | Failure constructions, cores and stronger-decoder checks |
| `circuits/` | Historical exports and research variants |
| `timing_consistency.json`, `time_budget.json` | Timing checks and resource costs |

Use `final/` for the analysed samples. `raw/` overlaps it and includes older batches; pooling both would double-count data. Original provenance stamps remain in the JSON files.

[Figure provenance](../figures/data/DATA.md) traces each plotted point to these inputs. [Methods](../RESULTS.md) give the estimator. [Evidence](../EVIDENCE.md) states the bound coverage and the limits of the selected failure constructions.
