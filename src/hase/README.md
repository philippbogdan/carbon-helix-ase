# Circuit and simulation code

This code follows the actual syndrome-dependent extraction path. Full, escalating, periodic and rotating rules share the same Carbon-Helix data code.

`protocols.make_protocol` builds an experiment from code, p, basis, memory rounds, rule and noise model. It adds one full preparation round. `carbonhelix.py` defines extraction parts and triggers; `variants.py` adds rotations such as R8.

`engine.py` samples faults and exports fixed paths to Stim. `runner.py` groups shots by path and decodes each group with matching priors. `decoding.py` wraps Frontier; `native.py` builds the engine with adaptive prior updates.

`helios.py` and `timing.py` hold the device-like noise and elapsed-time assumptions. Replace those with compiled timing and measured noise before using the comparison to choose a hardware schedule. [Usage](../../USE.md) gives the checked commands.
