# Use the circuits or port the schedule

Start with full extraction, then compare R8 under the same noise and timing. Both protect two logical qubits in 60 data qubits and use at most 92 qubits in the supplied comparison. This is a simulated resource count; a hardware schedule still needs compilation.

## Load a full memory

After the installation step in [REPRODUCE.md](REPRODUCE.md):

```python
import stim

circuit = stim.Circuit.from_file("circuits/memory_X.stim")
model = circuit.detector_error_model()
```

The supplied files have one preparation round and two memory rounds under uniform noise at $`p=0.001`$. They are reusable examples; the headline uses four memory rounds and Helios-like noise. `memory_Z.stim` changes the basis; `_noiseless.stim` removes noise. [build.py](scripts/build.py) changes the round count and uniform-model probabilities through `--p`, `--p2`, `--p1`, `--pr`, `--pm` and `--pidle`.

## Follow the adaptive decisions

```sh
.venv/bin/python scripts/simulate.py --shots 128 --out build/r8.json
```

The default is X memory, four rounds after preparation, R8 and Helios-like noise at $`p=0.00079`$. Use an unused output path. This small direct sample exercises the schedule and decoder; it does not measure the rare-event headline rate.

| Rule string | Checks after preparation |
| --- | --- |
| `full` | Every check every round |
| `l2=on_event,l3=ambiguous,rot=8` | R8: block checks on inner events; full outer checks on ambiguity or flags; otherwise one rotating outer check |
| `l2=on_event,l3=ambiguous,T3=4` | B4: the same escalation trigger, plus forced outer checks every fourth round |

Preparation always runs all checks. The rotation orders X0, Z0, X1, Z1, X2, Z2, X3, Z3 and indexes by round modulo eight, so the first post-preparation round uses Z0 when it does not escalate. [CarbonHelixRotating.plan](src/hase/variants.py) and [CarbonHelixProtocol.plan](src/hase/carbonhelix.py) define the decisions and syndrome references. A fixed Stim circuit of all possible parts cannot reproduce live branching. [The decoder](src/hase/runner.py) uses the path actually executed.

## Connect the parts

| Part | Two-qubit gates | Ancillas |
| --- | ---: | ---: |
| `L1_a` or `L1_b` | 120 | 30 |
| `L2_all` | 200 | 30 |
| `L3` | 240 | 32 |

The [manifest](circuits/manifest.json) records each part's measurement order, roles and data maps. Input data occupy indices 0 to 59. L1 refreshes data into ancillas, so connect the next part through `data_output`; preserve measurement bases, order and `rec[...]` feedback. [code.json](circuits/code.json) supplies stabilisers and paired logical operators. The outer checks' halfway cat-parity measurements catch fault pairs that final flags alone miss.

Replace [helios.py](src/hase/helios.py)'s noise parameters with your device inputs. [timing.py](src/hase/timing.py) uses the protocol's timing parameters, including preparation; models without durations use default Helios timing. Adapt its scheduling to your compiled circuit and check the [block idle-noise approximation](RESULTS.md#device-assumptions). Measure both full and adaptive extraction with those changes before drawing a hardware conclusion.
