# Numbers behind the figures

| File | Contents and units |
| --- | --- |
| `headline.csv` | Error per logical qubit per round (`eps_LQ`) and per second (`rate`), conditional intervals, gate counts, times in ms, qubits and observed-stratum shares |
| `r8_paths.json` | Real simulated R8 part lists; virtual-qubit gate pairs and whole-part durations in ms |
| `circuits.json` | Part gate counts, layers, ancillas, peak qubits and modelled durations in ms |
| `exponents.csv` | Free local error slopes and profile-likelihood intervals, with each fit's `p_star` |
| `intervals.csv` | Periodic/full per-round error ratios, bootstrap intervals and observed-stratum shares |
| `interval.json` | All interval families, ratios and directly sampled escalation fractions |
| `distance.json` | Restricted end-gadget searches, whole-path searches and gadget checks |
| `sensitivity.json` | Saved errors per round and resource means, with round times evaluated under each device variant |
| `direct.json` | Direct full-Carbon sampling versus stratified estimates under uniform noise |
| `c4_uniform.json` | Full-extraction C4 mean X/Z block error per round, conditional intervals, observed-stratum shares and comparison with the authors' Fig. 11a readings |

`headline.csv` selects `results/summary.json` → `value` → `helios|0.00079` or `helios|0.0003` → `rows`; `scripts/headline.py` adds preparation time and converts rates. Carbon's low-noise values are partly or almost entirely extrapolated. `scripts/figure_data.py` copies the matching JSON sections; `scripts/evidence.py` exports the exponent and interval CSVs.

Sensitivity ratios also include preparation, evaluated with the same variant as the round times. `scripts/headline.py --saved` prints these per-second comparisons; R8 was not tested in this sensitivity data.

`c4_uniform.json` copies the summary's `c4_uniform` section. Its `points` contain `eps_avg`, `ci95`, `measured_share_X`, `measured_share_Z`, `paper`, `ratio` and `paper_inside_ci`; [reference/](../../results/reference/) records the authors' values and reading uncertainty. These are full-extraction block rates over 12 memory rounds, averaged across bases; the conditional bootstrap intervals exclude reading uncertainty. The [evidence table](../../EVIDENCE.md#c4-helix-baseline) uses these values; no figure is rendered from them.

`r8_paths.json` contains 2,000 shots at $`p=0.00079`$, seed 2026, X memory and four rounds after preparation. Its mean is 214.0 gates per memory round, with full-outer escalation in 16.3% of those rounds. The headline's 215.8-gate mean comes from a separate 4,000-path sample. Local ancilla names are reused between parts, and `gate_layers` omits non-two-qubit operations; it is not a physical schedule.

The [results guide](../../RESULTS.md), [evidence guide](../../EVIDENCE.md) and [reproduction commands](../../REPRODUCE.md) define the normalisations, uncertainty and checks. No plotted error interval includes device uncertainty.
