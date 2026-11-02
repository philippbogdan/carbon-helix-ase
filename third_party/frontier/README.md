# Frontier decoder

Frontier approximates logical maximum-likelihood decoding. Here its fault priors follow the checks each memory path actually executed.

The bundled Python and C++ sources are from [Leverrier and Urbanke's implementation](https://github.com/aleverrier/frontier), commit `5d5a60968182eb17cdc12c5ac1949732ead0905b`, accompanying [arXiv:2606.20513](https://arxiv.org/abs/2606.20513).

The supplied fixed-circuit checks and adaptive decoder call `src/hase/native.py`, which uses `scripts/build_native.py` to add `update_priors` and compile into `build/adaptive_native/`. The patch refreshes probabilities and pruning weights when paths change. `scripts/build_decoder.py` can separately compile the unchanged `_frontier_native.cpp` into `build/native/` for upstream API use. Both leave this directory's source unchanged and require a C++17 compiler.

[AUTHORS.md](AUTHORS.md), [LICENSE](LICENSE) and [NOTICE](NOTICE) retain upstream attribution and Apache-2.0 terms. The surrounding project's own code, data, figures and text use [CC0 1.0](../../LICENSE); that dedication does not replace Frontier's licence.
