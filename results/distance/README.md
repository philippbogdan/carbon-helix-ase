# Path and end-gadget searches

`end__*` files restrict faults to end-of-circuit mechanisms. `design__*` compares outer-check gadgets. An optimum in either restricted search is not a whole-circuit lower bound. `whole__*` searches complete fixed paths and keeps the best weight and solver lower bound separately.

`whole__k4cyc__ZX__Z__c1.json` supplies an explicit 11-fault undetected logical error. Together with the matching [decomposition bound](../relaxed_distance/), it establishes distance exactly 11 for that finite Z memory. R8's restricted end checks do not establish its whole-circuit distance. [EVIDENCE.md](../../EVIDENCE.md#circuit-bounds) gives the coverage.
