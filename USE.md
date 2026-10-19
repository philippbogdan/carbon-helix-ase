# Run or port the extraction

Start with full extraction to check your simulator. Then compare R8 under your noise and timing model. The two encode the same two logical qubits in 60 data qubits; both use at most 92 physical qubits in the saved campaigns.

## Full extraction

Run `./run.sh`, then change the number of memory rounds or noise:

```sh
.venv/bin/python scripts/build.py --rounds 8 --p 0.001 --out build/my-circuits
```

`--rounds` excludes one full preparation round. The complete X and Z circuits include resets, preparation, memory, destructive readout, detectors and two logical observables. No shots are discarded.

```python
import stim
circuit = stim.Circuit.from_file("build/my-circuits/memory_X.stim")
detectors, observables = circuit.compile_detector_sampler(seed=7).sample(
    1000, separate_observables=True
)
model = circuit.detector_error_model()
```

The builder's uniform model uses two-qubit depolarisation, reset and measurement flips at p, and single-qubit and idle Z errors at p/10. Override them with `--p2`, `--p1`, `--pr`, `--pm`, `--pidle`.

## Adaptive extraction

```sh
.venv/bin/python scripts/simulate.py --shots 128 --out build/r8.json
.venv/bin/python scripts/simulate.py --rule full --shots 128 --out build/full.json
.venv/bin/python scripts/simulate.py --noise uniform --p 0.001 --rounds 8 --shots 128 --out build/uniform.json
```

The default is R8, four memory rounds, X basis and Helios-like noise at p = 0.00079. Output includes the observed failure count and executed parts. These direct samples do not refit the rare-event estimates. Use `--basis Z` to check the other memory basis. Seeds and decoder settings are written into each output.

| Rule string | Decisions after preparation |
| --- | --- |
| `full` | Run every check every round |
| `l2=on_event,l3=ambiguous,rot=8` | R8: run weight-6 checks on blocks with inner events; run all outer checks on ambiguous changes or flags; otherwise rotate one outer check per round |
| `l2=on_event,l3=ambiguous,T3=4` | B4: escalate on ambiguity and force full checks every fourth round |

R8 cycles through X0, Z0, X1, Z1, X2, Z2, X3, Z3. Preparation is always full. The exact round indexing and trigger are in `CarbonHelixRotating.plan` in `src/hase/variants.py` and `CarbonHelixProtocol.plan` in `src/hase/carbonhelix.py`. The simulator keeps the executed path and changes decoder priors to match it. Concatenating the possible parts into one fixed Stim circuit would lose those decisions.

## Connect the parts to your stack

| Part | Two-qubit gates | Ancillas | Purpose |
| --- | ---: | ---: | --- |
| L1_a or L1_b | 120 | 30 | Inner C4 checks, alternating schedules |
| L2_all | 200 | 30 | All weight-6 Carbon checks |
| L3 | 240 | 32 | All weight-16 outer checks |

The part files in `circuits/` contain extraction operations and local feedback. Data enter at indices 0 to 59. L1 changes where some data live: use `data_output` in `circuits/manifest.json` when wiring the next part. Preserve measurement order, bases and `rec[...]` feedback. The manifest gives each measurement's key and role. `code.json` supplies stabilisers and logical operators.

Outer cats are checked halfway through their data interactions. The fault-pair check in `scripts/check_parities.py` shows why: two cat faults can cancel before the final flags but still damage data.

For hardware timing and noise, edit `src/hase/helios.py` and the layer accounting in `src/hase/timing.py`, or add a model through `src/hase/protocols.py`. The supplied model approximates transport and leakage as Pauli noise. Replace those assumptions with your compiled schedule and measured idle behaviour before comparing hardware performance.

For stratified sampling, `scripts/job_strat.py --help` exposes the local campaign worker. It needs no cluster service. Retained source files outside these examples support the broader research run; only the commands documented here form the checked entry points.
