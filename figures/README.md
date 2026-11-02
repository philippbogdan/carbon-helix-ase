# Visual evidence

The [R8 animation](r8_paths.gif) shows selected real simulated paths beside full extraction; [MP4](r8_paths.mp4) preserves the same sequence. Ancillas occupy a polar grid keyed to part-local names: a check's apparent location near a block does not identify its physical ancillas. Playback shows gate order, with pauses between rounds and shots, rather than elapsed time.

`memory_rate.png` compares estimated error per second, `extraction_cost.png` compares gates and modelled time, `fault_order.png` shows free-order fits, and `interval_cost.png` shows the cost of longer measurement windows. Each static figure also has an SVG. Captions and caveats live in [the main README](../README.md) and [evidence guide](../EVIDENCE.md).

`./run.sh figures` renders from [data/](data/) through `make_figures.py`. It does not sample, decode or fit. The [data guide](data/README.md) identifies every input and its units.
