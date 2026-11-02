# Definitions behind the estimates

Carbon-Helix concatenates a $`[[10,2,3]]`$ outer code with five $`[[12,2,4]]`$ Carbon blocks, giving $`[[60,2,12]]`$. A stabiliser check measures a parity without reading out the encoded information. A detection event is a change from the check's previous result; a flag is an auxiliary measurement that detects dangerous fault propagation. [code.json](../circuits/code.json) and [checks.json](../results/checks.json) give the matrices and structural checks.

## Extraction and paths

L1 measures the inner C4 checks. L2 measures weight-6 Carbon checks. L3 measures the eight weight-16 outer checks. R8 runs L2 on blocks with inner events, escalates to L3 on an ambiguous syndrome change or a flag, and otherwise runs one rotating outer check. A change is ambiguous when it does not match a single-qubit Pauli error on a block.

The outer-check gadget reads cat-qubit parities halfway through the data interactions. A cat error can appear and disappear before the final flags while leaving data errors; the halfway read makes that fault pair visible. [check_parities.py](../scripts/check_parities.py) constructs it.

The adaptive engine records executed parts. Its superset detector model gives skipped parts zero fault prior and replaces skipped measurements with erasure mechanisms. Frontier then decodes each path with its own priors. The [porting guide](../USE.md) covers the physical data maps, which differ from the virtual slot names used internally.

## Fault-count estimation

For total fault count $`W`$ over all possible part instances, the estimate is

```math
P_{\mathrm{fail}}(p)=\sum_w P_p(W=w)P(\mathrm{fail}\mid W=w).
```

Sampling with multiplied fault odds makes rare counts accessible. Reusing conditional failure fractions across scaled noise values is a small-noise approximation: component probabilities scale together, while their conditional odds need not stay exactly proportional. [strat.py](../analysis/strat.py) combines observed fractions and fitted fractions and propagates counting uncertainty by refitting bootstrap samples. [RESULTS.md](../RESULTS.md) defines the per-round and per-second normalisations and the observed-stratum threshold.

The headline's minimum Carbon fit order is six. Free-order fits instead estimate a real falling-factorial order from the observed counts, then report the local slope at the noise whose expected fault count matches that of the observed failures. Held-out checks predict lower-count strata from higher-count ones; a non-rejection cannot establish the extrapolated low-noise limit. Locally reduced failure sets can miss smaller failing subsets.

## Circuit distance by decomposition

The [distance algorithm](../src/hase/relaxed_distance.py) keeps flags and inner detectors, but combines each outer detector chain into a final parity. It tabulates flag-silent outer-gadget effects and cheapest block histories. Optional block checks are minimised over execution and omission. Removing constraints can only lower the minimum undetected logical fault weight, so proving a lower bound in this relaxed model bounds every covered path.

The [coverage table](../EVIDENCE.md#circuit-bounds) lists the finite cases. Minimum-fault distance bounds do not certify the behaviour of approximate probabilistic decoding, and the saved bounds do not cover R8's individual rotating checks.
