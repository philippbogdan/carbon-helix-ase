# Stronger-decoder checks

These files replay sampled failures with wider Frontier searches and also test a random sample of successes. Under Helios-like noise, stronger decoding corrects 13 of 153 full-Carbon failures and 6 of 80 R8 failures. The headline estimates still count those failures.

`per_w` records transitions by fault count; `totals` summarises the audited sample. Wider approximate decoding can change decisions in either direction and is not an exact oracle. `analysis/redecode.py` repeats the comparison; [EVIDENCE.md](../../EVIDENCE.md#scaling-and-decoding) explains its scope.
