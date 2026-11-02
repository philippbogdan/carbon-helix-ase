# Explicit failure mechanisms

`constructed.json` tests complementary preparation errors and selected pairs of three-fault block events within a measurement window. Other files greedily reduce sampled failures and classify the remaining sets with a most-likely-error ILP or stronger approximate decoding.

A locally minimal set need not be the smallest possible failure. The ILP wrapper accepts returned solutions without requiring proven optimality; its labels are not exact-decoder certificates. Constructed placements demonstrate existence, not frequency. [EVIDENCE.md](../../EVIDENCE.md) separates these witnesses from lower bounds.
