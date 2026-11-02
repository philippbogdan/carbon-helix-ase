# Circuits, adaptive paths and decoding

`protocols.make_protocol` builds a memory from the code, noise, basis, rule and round count, adding full preparation. `carbonhelix.py` defines checks and event triggers; `variants.py` adds R8's rotating outer checks. `codes.py` holds the stabilisers and logical operators.

`engine.py` samples the parts each shot actually needs and exports fixed paths to Stim. `runner.py` decodes with path-dependent priors; `decoding.py` wraps Frontier and `native.py` loads its prior-update extension. `helios.py` supplies component noise. `timing.py` uses each protocol's durations, including preparation, with default Helios timing for models without durations. The [block idle-noise approximation](../../RESULTS.md#device-assumptions) remains. [USE.md](../../USE.md) connects these modules to reusable circuits.
