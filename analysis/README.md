# Estimates and cross-checks

`strat.py` pools fault-count samples, fits rare-event probabilities and samples paths for resource means. `value.py` supplies the headline comparison; `carbon.py` adds free-order fits, held-out checks and decoder audits. The headline extrapolation assumes six faults for Carbon; the free-order fits test local slopes independently.

`interval.py` compares full-check spacing. `window_failures.py`, `minimal_fails.py` and `core_census.py` expose failure mechanisms. `decoder_audit.py` and `redecode.py` test stronger decoding; `timing_consistency.py` measures omitted idle exposure. `resources.py`, `gadget_gain.py` and `distance_witness.py` inspect circuit costs and fault propagation.

`aggregate.py` builds a new summary from samples, stored audits and the [paper reference CSV](../results/reference/) without needing an existing summary. Its `c4_uniform` section compares full-extraction C4 with the authors' Fig. 11a readings. `--check` compares with an existing summary; `--only sensitivity` or `--only c4_uniform` updates that section in place. A full summary check took about 5.5 CPU minutes; it does not rerun the audits. [REPRODUCE.md](../REPRODUCE.md) gives the commands; [EVIDENCE.md](../EVIDENCE.md) states their scope.
