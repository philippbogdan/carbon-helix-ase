# Frontier decoder

Frontier approximates logical maximum-likelihood decoding. Here it decodes each executed extraction path with priors that account for skipped checks.

The bundled Python and C++ sources come from `aleverrier/frontier` commit `5d5a60968182eb17cdc12c5ac1949732ead0905b`. `_frontier_native.cpp` and `_frontier_native.upstream.cpp` contain the same upstream source.

The checked commands use `scripts/build_native.py` to apply the local prior-update patch and compile a separate engine. The patch updates fault probabilities and pruning weights when paths change. `scripts/build_decoder.py` can build the unchanged engine. No binaries are required in the source copy.

[AUTHORS.md](AUTHORS.md), [LICENSE](LICENSE) and [NOTICE](NOTICE) preserve upstream attribution and Apache 2.0 terms. Philipp Bogdan's own code, data, figures and text in the surrounding project are dedicated to the public domain under [CC0 1.0](../../LICENSE). Frontier remains under Apache-2.0.
