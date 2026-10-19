# Circuit, scaling and interval evidence

The extraction fits 92 qubits. Six-fault suppression has circuit bounds for the schedules below and free-order fits near six. Silent errors in two blocks give a reason to cap the full-measurement interval even at lower noise. The saved proof outputs, samples and failure constructions are included here.

## What the circuit bounds cover

The decomposition retains flags and inner detectors and relaxes the outer detector constraints. Any undetected logical error of the full circuit must survive this relaxation. It tabulates cheapest block histories and flag-silent outer-gadget effects, then searches their combinations. Optional weight-6 checks are minimised over run and skip, so each computation covers every such schedule.

All 12 saved cases exclude undetected logical errors of at most ten faults in every nontrivial logical class:

| Case | Preparation plus memory rounds | Basis | Outer-check order |
| --- | --- | --- | --- |
| `X_XZ_full1`, `Z_XZ_full1` | 1 + 1 | X, Z | X then Z |
| `Z_ZX_full1` | 1 + 1 | Z | Z then X |
| `X_XZ_full8`, `Z_XZ_full8` | 1 + 8 | X, Z | X then Z |
| `X_XZ_any4`, `Z_XZ_any4` | 1 + 4 | X, Z | X then Z |
| `X_XZ_any8` | 1 + 8 | X | X then Z |
| `X_XZ_P2x4` | 1 + 8, four windows of 2 | X | X then Z |
| `X_XZ_P4x2` | 1 + 8, two windows of 4 | X | X then Z |
| `X_XZ_P8x1` | 1 + 8, one window of 8 | X | X then Z |
| `X_XZ_P16x1` | 1 + 16, one window of 16 | X | X then Z |

`any` includes each block's weight-6 checks as optional and a possible full outer-check part in every round. It bounds escalating and periodic paths. It does not bound rotating individual outer checks (R4, R8, R8F), arbitrary experiment lengths or logical gates.

Raw tasks are in `results/relaxed_distance/`. Their subset coverage, all three logical classes and the no-gadget cases are checked by `scripts/evidence.py`. That script also verifies an 11-fault witness from `results/distance/whole__k4cyc__ZX__Z__c1.json` directly against the rebuilt detector matrix: zero syndrome and a nonzero logical flip. The bound and witness establish distance exactly 11 for that two-round Z memory with X checks last. Distance at least 11 means an ideal minimum-fault decoder can correct up to five faults. A finite-beam probabilistic decoder can make extra errors; this distance statement alone does not certify Frontier's error order.

Outside outer-check gadgets, faults act within one Carbon block. Exact integer programmes give inner fault distance four on all 276 weight-6 schedules for windows T = 2, 4, 8, in both bases. No logical operator is supported on two blocks or fewer, so block-only faults need at least 12 for an undetected outer logical. The whole-circuit decomposition accounts for the outer gadget faults that can reduce this to 11. `results/inner_window/` contains all paths and the structural check. Those finite-window checks support the argument; they are not a proof for every T.

## What the exponent fits show

`figures/exponents.csv` records all six rules under both noise models. The fit uses binomial likelihood on the sampled fault-count strata with a free real order k in a falling-factorial model. It reports the implied slope d ln(error)/d ln(p) at p*, where the expected number of faults equals the observed failures' mean count. Profile-likelihood 95% intervals describe that fit. This does not impose six as a minimum order. It is still a model of the observed strata, not a direct measurement of asymptotic low-p behaviour.

The input is `results/final/ch_main/` for uniform noise and `results/final/ch_helios/` for Helios-like noise. Each four-round X configuration has 64,000 stratified shots. The main value estimator uses orders 6 to 9 in sparse strata. Slopes derived from that extrapolation cannot test the six-fault premise.

Six preparation faults on half the support of a weight-12 logical share their syndrome with faults on the complementary half. At least one member must fail under a deterministic decoder. The construction in `analysis/window_failures.py` checks 20 such pairs: every pair has a failure. It also checks 20 five-fault subsets with no failures. That small five-fault sample is not a lower-bound proof.

Frontier's stronger-decoder audit removes 20/146 full, 22/169 any-event, 18/123 B4 and 25/155 R8 failures under uniform noise. Under Helios-like noise it removes 13/153 full and 6/80 R8 failures. The campaign estimates count those failures. `results/redecode/`, `analysis/redecode.py` and `analysis/decoder_audit.py` retain the audit. Cores confirmed by exact or stronger decoding reach six faults for full, C, B4 and R4; the smallest confirmed sampled cores for R8 and R8F have seven faults. The general six-fault tie construction does not replace a whole-circuit lower bound for those rotations.

## Why a fixed full-measurement interval

An inner weight-four logical restricted to three of its qubits has the syndrome of the remaining single error. A decoder may correct that single error confidently and leave an inner logical. Two such three-fault events in different blocks can fail inside one outer-measurement window. `results/minimal/constructed.json` records the exact placements and outer-measurement paths: full corrects all eight; B4 fails four, B8 six, C two and R4 six. The B8 construction forces both L2 and L3 every eighth round; B4 forces L3 every fourth. Do not treat this selected set as an empirical probability or claim every same-window placement fails.

For fixed T in the analysed setting, both the in-round floor and the window failures start at p^6. A fault-counting expansion has the form

```text
error(p,T) = p^6 [B' + C1(T-1) + C2(T-1)^2 + ...] + higher orders.
```

Two events have more placements as the window grows. Their contribution per round has a nonzero window-length penalty at the same order as the floor. Holding their ratio to the full-extraction error fixed therefore requires a bounded T as p falls. The 1/p^4 suggestion would let that penalty grow; it does not retain a constant-factor comparison. This is an asymptotic fault-counting argument supported by finite circuits and explicit failures. The expansion is not justified uniformly for windows growing without limit. It cannot assign an exact exponent to a 1/p^4 rule, or a best T on hardware.

The periodic study isolates window length by never escalating on events: L1 runs every round, L2 on events and at window ends, and L3 only at the ends. It uses eight memory rounds for T = 2, 4, 8 and sixteen for T = 16. Ratios compare per-round error with full extraction. `figures/intervals.csv` keeps all ten saved p values, the 95% bootstrap intervals, sampled shares and campaign counts. At p = 0.004, 0.003, 0.0025, at least half of every compared periodic value comes from sampled strata with at least three failures. The schedule figure also shows all lower-share points as open markers, and includes the B, C and R families. This threshold is weaker than the 90% rule used to label the headline estimates measured. Uncertainty is conditional on the order-six hybrid model, with 100 smoothed parametric bootstrap fits.

The central periodic ratios through T = 16 are below 1.5 at those three p. Their confidence intervals do not establish a universal optimal interval. Below the sampled region the fitted ratios cannot independently show that the interval is constant. Escalating rules also obscure the window penalty at the sampled p because they often measure the full syndrome before T. Direct path sampling without forcing gives ambiguous-event escalation in about 12% of rounds at p = 0.001 and 0.92% at p = 0.0001. This explains why a forced cap matters more at lower p, rather than establishing a long-time memory lifetime.

## Reproduce and inspect

Run these from the root:

```sh
./run.sh figures
./run.sh verify
```

`figures` checks the data pack against the saved summary, checks circuit costs and saved distance aggregation, regenerates all five plots, checks complete saved bound coverage and verifies the 11-fault witness. The construction replay decodes eight-round circuits with stronger decoding and can take more than 15 minutes. `verify` refits all 12 free exponents from raw counts and the ratios, intervals and sampled-share classifications for all four schedule families at all ten p with the original 100-fit bootstrap seed allocation. It also repeats the two 8000-shot escalation scans and reruns all constructed failures. Fresh checks go to `build/evidence-check.json` and `build/window-replay/`; saved results remain unchanged. Neither command reruns the expensive distance searches or original sampling campaigns.

The distance algorithms and their local drivers are included. A small fresh solver case and a witness description are:

```sh
.venv/bin/python scripts/inner_window_distance.py --T 2 --basis X --out build/inner-T2-X.json
.venv/bin/python analysis/distance_witness.py results/distance/whole__k4cyc__ZX__Z__c1.json
```

The full decomposition can take hours. `.venv/bin/python scripts/relaxed_distance.py --help` lists its options. Kind `P8x1` searches one eight-round periodic window. The driver defaults to one local task. `scripts/job_distance.py` and `scripts/distance_suite.py` retain the earlier unrestricted and end-gadget searches. `scripts/campaigns.py` retains the original seeds and job arguments as a research registry; its cluster submission interface is not needed by these local commands.
