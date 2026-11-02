# References and reuse

- [C4-Helix, arXiv:2609.03194](https://arxiv.org/abs/2609.03194): outer code, C4 extraction and comparison conventions. The authors' Fig. 11a supplies the [reference readings](../results/reference/) for our full-extraction uniform-noise check; we read the points from the published figure, with estimated 3% relative reading uncertainty.
- [Carbon, arXiv:2404.02280](https://arxiv.org/abs/2404.02280): Carbon stabilisers and logical operators.
- [Adaptive syndrome extraction, arXiv:2502.14835](https://arxiv.org/abs/2502.14835) and its [reference implementation](https://github.com/noahberthusen/adaptive_qec): adaptive-check methodology.
- [Helios, arXiv:2511.05465](https://arxiv.org/abs/2511.05465): device benchmarks used in [the model](../src/hase/helios.py); assumed timings and Pauli approximations are identified there.
- [Frontier, arXiv:2606.20513](https://arxiv.org/abs/2606.20513), Anthony Leverrier and Rüdiger Urbanke: approximate logical maximum-likelihood decoding. The [bundled sources](../third_party/frontier/) retain their Apache-2.0 licence and attribution.

Our code, data, figures and text use [CC0 1.0](../LICENSE); the source publication behind the reference readings is the C4-Helix authors' work, not covered by our dedication. Installed dependencies retain their own terms: Stim, Apache-2.0; NumPy and SciPy, BSD-3-Clause; Matplotlib, its BSD-style licence; Pillow, MIT-CMU. The local native-decoder patch is applied at build time, leaving the bundled upstream source unchanged.
