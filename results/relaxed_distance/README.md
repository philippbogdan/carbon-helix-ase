# Whole-circuit lower bounds

These 12 cases cover full, escalating and periodic Carbon schedules and exclude undetected logical errors of at most ten faults. The case name records basis, outer-check order and path family; `task*.json` partitions the active-gadget subsets. Every logical class and the no-gadget case must be covered.

`./run.sh evidence` checks coverage and saved outcomes without repeating the searches. [EVIDENCE.md](../../EVIDENCE.md#circuit-bounds) lists the finite lengths and connects the bound to an explicit witness. Rotating individual outer checks, including R8, are outside these bounds.
