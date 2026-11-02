# Block errors between full measurements

`T*_X.json` and `T*_Z.json` give exact inner fault distance four for all 276 weight-6 schedules tested at windows of 2, 4 and 8 rounds. `structure.json` checks block-local faults and the absence of outer logical operators supported on at most two blocks.

These finite checks support the [window argument](../../docs/interval_theory.md); they do not prove every window length or bound outer-gadget faults. `scripts/inner_window_distance.py` repeats the searches, while `./run.sh evidence` checks the saved outputs.
