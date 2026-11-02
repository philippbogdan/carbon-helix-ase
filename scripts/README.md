# Entry points

`./run.sh headline` refits and checks the headline. `baseline` runs `c4_baseline.py` to refit full-extraction C4 under uniform noise, compare with the authors' Fig. 11a readings and check the saved summary section. `check` rebuilds and samples circuits; `evidence` checks saved bounds; `figures` renders the data; `verify` repeats the longer analyses. Run them from the repository root. [REPRODUCE.md](../REPRODUCE.md) gives dependencies, costs and output paths.

`simulate.py` runs live adaptive memories; `build.py` exports full circuits. `sample_strata.py` samples by fault count and `sample_direct.py` samples directly. `campaigns.py list` lists configurations; `commands` prints their local invocations and seeds.

`path_distance.py` searches one path; `distance_suite.py` enumerates restricted end cases; `relaxed_distance.py` proves decomposition bounds; `inner_window_distance.py` checks block histories. `structure_checks.py` checks code structure and engine/Stim agreement. These searches can take hours.

`figure_data.py` exports saved summary sections, including `c4_uniform.json`; `--check` compares without writing. `evidence.py` checks bounds and exports exponent/interval tables. `r8_paths.py` samples the hero's executed parts without decoding; rendering uses the supplied JSON. `build_decoder.py` compiles unchanged Frontier; `build_native.py` adds adaptive prior updates in a separate build.
