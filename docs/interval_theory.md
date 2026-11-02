# Why the measurement interval stays bounded

A three-fault error on a Carbon block can look like a single-qubit error while leaving an inner logical error after correction. Two such events in different blocks can defeat correction before the next full outer measurement. [The constructed placements](../results/minimal/constructed.json) demonstrate this mechanism without estimating its probability.

For a fixed interval $`T`$, the fault-counting picture is

```math
\epsilon(p,T)=p^6\bigl[B+C_1(T-1)+C_2(T-1)^2+\cdots\bigr]
+\text{higher orders in }p.
```

The window penalty and the full-extraction floor start at the same fault order. More time between measurements gives the events more placements, so maintaining a fixed relative error penalty requires a bounded maximum interval as $`p`$ falls. This argument does not justify extending the expansion uniformly to unbounded windows, assign an exact error exponent to $`T\propto1/p^4`$, or choose a universal best interval.

The periodic data use eight memory rounds for $`T=2,4,8`$ and sixteen for $`T=16`$. Ratios share the eight-round full-extraction baseline, so the longest-window comparison also changes experiment length. At $`p=0.004,0.003,0.0025`$, at least half of both estimates in every ratio comes from observed strata. This is weaker than the headline's 90% "measured" threshold. Lower-noise fitted ratios cannot independently establish the scaling law.

Ambiguous-event B and any-event C can end a window early. Periodic P never escalates, exposing the full window length. For rotation R, the interval is the spacing of each individual outer check, not a forced full-check cap. The finite R8 memories do not establish that its rotation replaces a cap in arbitrarily long storage. [EVIDENCE.md](../EVIDENCE.md#why-a-fixed-full-measurement-interval) connects these distinctions to the results.
