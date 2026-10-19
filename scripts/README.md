# Local commands

Start at the repository root with `./run.sh`. It rebuilds the supplied circuits exactly, checks noiseless X and Z memories, tests the cancelling cat-fault pair and decodes a noisy sample. Fresh outputs go to `build/`.

| Command | What it checks |
| --- | --- |
| `./run.sh figures` | All four figure inputs, circuit costs, solver aggregation, saved bound coverage and witness; regenerates five figures |
| `./run.sh reproduce` | Six headline estimates refitted from raw counts, including gate and time sampling |
| `./run.sh verify` | Twelve free-order fits, every plotted schedule ratio, both escalation scans and constructed six-fault failures |

`figure_data.py` exports the data pack; `--check` compares it without rewriting its values. `headline.py` selects the six comparison rows and converts error per round to error per second. `../figures/make_figures.py` is the only figure renderer. `evidence.py` checks the saved bounds and exports the numerical evidence tables.

`build.py` exports full memories and reusable extraction parts. `simulate.py` runs adaptive paths locally. `job_protocol.py` samples directly; `job_strat.py` samples by fault count. Each accepts `--help`.

`build_native.py` builds the bundled Frontier engine with path-prior updates when decoding first needs it. `build_decoder.py` builds the unchanged upstream engine separately.

The distance drivers retain the solver entry points. Full searches may take hours. [Evidence](../EVIDENCE.md) gives small local checks and states what the saved proofs cover. `campaigns.py` preserves historical seeds and job arguments; its cluster submission commands are not needed for local use.
