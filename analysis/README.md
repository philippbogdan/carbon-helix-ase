# Estimation and checks

The central result is an extraction saving at similar estimated memory error per second. The main comparison assumes order six for sparse Carbon strata. Separate free-order fits and circuit bounds test that assumption within their stated scope.

`strat.py` pools fault-count samples, fits error estimates and intervals, and samples executed paths for cost. `value.py` compares Carbon and C4. `interval.py` studies forced full-check spacing and event-trigger frequency. These supply the four inputs in [the figure pack](../figures/data/DATA.md).

Run `./run.sh reproduce` from the repository root to refit the headline comparison. `./run.sh verify` refits all plotted schedule families and free exponents, repeats the escalation scans and replays the six-fault constructions. The commands compare fresh calculations with saved results.

`window_failures.py` builds the selected failing placements. `decoder_audit.py` and `redecode.py` check approximate decoding. The distance modules search for failing fault sets or prove lower bounds. [Evidence](../EVIDENCE.md) distinguishes whole-circuit proofs from restricted end-gadget searches.

`make_all.py` contains the original summary aggregators. The focused data check calls its circuit and distance aggregators directly. Its complete report build and `render_docs.py` also need historical templates and paper images, so they are retained as research source. Other modules preserve the wider comparisons and audits.
