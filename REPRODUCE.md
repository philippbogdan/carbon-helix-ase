# Check, sample, rebuild

From the repository root, with Python 3.12 or newer:

```sh
./run.sh headline
```

This installs the pinned [requirements](requirements.txt) in `.venv`, refits the six headline estimates from included fault-count samples, and compares them with `results/summary.json` and `figures/data/headline.csv`. It also resamples paths for resource averages, prints the comparison and writes `build/headline.json`. Expected time after installation: about 10 seconds. The sensitivity values come from saved estimates.

For a quick comparison using only the saved estimates, without refitting or sampling:

```sh
.venv/bin/python scripts/headline.py --saved
```

| Command | Work and output |
| --- | --- |
| `./run.sh baseline` | Refits the full-extraction C4 uniform-noise samples and compares the mean X/Z block error per round with the authors' Fig. 11a readings at three noise points. Checks `results/summary.json` section `c4_uniform`, prints the comparison and writes `build/c4_baseline.json`. About 3 seconds after installation. |
| `./run.sh check` | Rebuilds circuits into `build/circuits/` and requires exact agreement; checks noiseless memories and the halfway cat-parity fault pair; decodes noisy X and Z samples. About a minute plus compilation. |
| `./run.sh evidence` | Checks the figure JSON against the summary, saved bound coverage, inner-window results and the explicit logical-error witness; exports exponent and interval CSVs. Writes `build/evidence-check.json`. Under a second is expected. |
| `./run.sh figures` | Renders the visuals from `figures/data/` through `figures/make_figures.py`; no sampling or decoding. About 51 seconds in the measured run. |
| `./run.sh verify` | Refit stage (12 exponents, all schedule ratios and intervals, both escalation scans): matched in under 10 minutes on the build Mac. Constructed-failure replay (`scripts/evidence.py --windows`) matched exactly: 20/20 complementary six-fault pairs had at least one failure, 0/20 selected five-fault subsets failed, and the window constructions matched the saved results. Replay took about 93 minutes wall clock on 2 Linux cores, including environment creation and decoder compilation. Every summary section matched; the summary check took about 5.5 CPU minutes. |

Decoding requires a C++17 compiler, selected with `CXX` if needed. Figure video export requires `ffmpeg` with H.264 encoding. `baseline` refits saved samples without fresh sampling or decoding; [its scope](EVIDENCE.md#c4-helix-baseline) is full extraction under uniform noise. `evidence` checks saved solver outputs; it does not repeat the distance searches. `verify` reanalyses samples and replays constructions; it does not repeat the full sampling campaigns. Reproduction depends on the pinned numerical packages and seeds; elapsed times vary by machine.

## Fresh memories

```sh
.venv/bin/python scripts/simulate.py --shots 128 --out build/r8.json
.venv/bin/python scripts/simulate.py --rule full --shots 128 --out build/full.json
.venv/bin/python scripts/build.py --rounds 8 --p 0.001 --out build/my-circuits
```

Choose unused sample output paths. The first commands simulate and decode short Helios-like memories; small direct samples cannot resolve the headline rare-event rates. `build.py` exports full-extraction circuits under uniform noise. [USE.md](USE.md) covers rules, mappings and model changes.

## Full calculations

The included circuits and component noise model are the public inputs. [campaigns.py](scripts/campaigns.py) specifies sample counts, decoder settings, output paths and seeds:

```sh
.venv/bin/python scripts/campaigns.py list
.venv/bin/python scripts/campaigns.py commands all > build/campaign-commands.sh
```

The commands write into `results/` and usually skip existing outputs. To rerun everything, use a separate working copy, move its supplied `results/` to `build/reference-results/`, create an empty `results/`, copy `build/reference-results/reference/` back to `results/reference/`, then execute `sh build/campaign-commands.sh`. Keep the reference copy for comparison. `commands carbon,c4_strat` selects the main comparisons; `commands c4_repro` selects the full-extraction C4 baseline; `--missing` prints only commands whose output path is absent. Expected total work from the recorded results is about 1,300 CPU-hours, including about 206 CPU-hours for the headline sampling campaigns.

The registry covers sampling and distance searches. Regenerate the supporting audits separately:

| Entry point, using `.venv/bin/python` | Output |
| --- | --- |
| `scripts/structure_checks.py` | `results/checks.json` |
| `analysis/gadget_gain.py --out results/gadget_gain.json` | Gadget fault propagation |
| `analysis/decoder_benchmark.py` | Decoder comparison |
| `analysis/timing_consistency.py` | Idle-exposure estimate |
| `analysis/window_failures.py` | Constructed preparation and window failures |
| `analysis/redecode.py CAMPAIGN TAG BASIS` | Stronger-decoder audit for that configuration |
| `analysis/minimal_fails.py --campaign CAMPAIGN TAG BASIS LIMIT PART/PARTS` | Reduced failures for that configuration and partition |

For the last commands, use each saved audit's configuration; for reduced failures also preserve its `n`, `part`, `nparts` and `selection`. The supplied reductions use `lowest_w`. Fault lists are recreated by the sampling commands. These analyses can require `PYTHONPATH=src:third_party/frontier` when invoked directly.

Run `.venv/bin/python analysis/aggregate.py` to write a new `results/summary.json` from the samples, audit files and [paper reference CSV](results/reference/); no previous summary is needed. For comparison, first restore the reference summary from `build/reference-results/summary.json`, then use `--check`, which leaves it unchanged. `--only sensitivity` or `--only c4_uniform` rebuilds just that section in an existing summary; combine it with `--check` to compare without writing. The aggregator reads the saved audits rather than rerunning them.

Numerical comparisons and solver statuses matter; elapsed times, dependency-version fields and time-limited solver outcomes need not reproduce exactly. [EVIDENCE.md](EVIDENCE.md) identifies each result's scope.
