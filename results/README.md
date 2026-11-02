# Saved samples and evidence

The headline uses 480,000 adaptive C4 shots and 64,000 shots for each Carbon configuration. These are stratified by fault count; their failure totals cannot be read as direct low-noise failure frequencies.

`summary.json` gathers the numerical results. [final/](final/) contains the sample histograms; [distance/](distance/), [relaxed_distance/](relaxed_distance/) and [inner_window/](inner_window/) contain solver evidence. [minimal/](minimal/) holds failure constructions and reduced fault sets; [redecode/](redecode/) holds stronger-decoder checks. [reference/](reference/) credits the authors' Fig. 11a values used in the `c4_uniform` baseline comparison.

`analysis/aggregate.py` rebuilds the summary without a previous summary, using the samples, saved audits and reference CSV; `--check` compares against an existing summary and `--only` updates named sections. `./run.sh baseline` refits and checks `c4_uniform`. Sensitivity round times use each variant's model. [The per-second comparison](../RESULTS.md#device-assumptions) also includes preparation.

`checks.json` records code parameters and detector-model comparisons. `timing_consistency.json` estimates omitted idle exposure. `decoder_benchmark.json` compares decoder settings; `gadget_gain.json` bounds fault propagation. [EVIDENCE.md](../EVIDENCE.md) states the scope; [REPRODUCE.md](../REPRODUCE.md) distinguishes checking these outputs from rerunning their calculations.
