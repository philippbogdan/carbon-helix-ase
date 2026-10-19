# Sources and provenance

The data and scientific implementation come from Philipp Bogdan's `helix-ase-frontier` run at `ac806e6cbf4c314a89176771197e76047498fb34`. This repository adds a focused public interface, new explanations, result plots and local reproduction commands. The original numerical summary and raw JSON records retain their provenance stamps.

`source-manifest.json` records SHA-256 hashes of the imported scientific source and results at assembly. The main changes to that source are local Frontier imports and native-build paths. The circuit exporter and check scripts come from the earlier circuit package and now use the complete adaptive engine. Public figures and documentation are newly written.

## Prior work

- [C4-Helix, arXiv:2609.03194](https://arxiv.org/abs/2609.03194): the concatenated construction, inner extraction and C4 comparison.
- [Carbon, arXiv:2404.02280](https://arxiv.org/abs/2404.02280): the inner code's stabilisers and logical operators.
- [Adaptive syndrome extraction, arXiv:2502.14835](https://arxiv.org/abs/2502.14835) and [reference implementation](https://github.com/noahberthusen/adaptive_qec): prior work on selective extraction.
- [Helios, arXiv:2511.05465](https://arxiv.org/abs/2511.05465): component data used by the device-like model. `src/hase/helios.py` records each input and its modelling assumptions.
- [Frontier, arXiv:2606.20513](https://arxiv.org/abs/2606.20513), by Anthony Leverrier and Ruediger Urbanke: the decoder. Bundled source revision: `5d5a60968182eb17cdc12c5ac1949732ead0905b`.

Paper figures and manuscripts are not redistributed. The plotted comparisons here use this run's samples, not digitised paper points. Original digitised reference values remain in the research archive for audit.

Philipp Bogdan dedicates his code, data, figures and text to the public domain under [CC0 1.0](LICENSE), with no attribution or other conditions required. This covers his work only. Frontier keeps its [Apache 2.0 licence](third_party/frontier/LICENSE), [authors](third_party/frontier/AUTHORS.md) and [notice](third_party/frontier/NOTICE).

## Figure preparation

The five supplied figures are kept unchanged. `figures/make_figures.py` retains their plotting code with local data and exporter paths. `_export/figure_export.py` and `_export/_common.py` are copied from the scientific-visualization skill supplied with this project. Exporter code is unchanged. [Data provenance](figures/data/DATA.md) records the full source chain. `figures/data/provenance.json` hashes the pack and its inputs; `source-manifest.json` remains the original import record.

The figure review followed Kassis, T., Agarwal, V., He, Y., Patel, D., and Brueckner, A. M. (2026), [Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://doi.org/10.48550/arXiv.2609.00065). This credits guidance on source data, uncertainty and export checks. It is not a source for the quantum-error-correction results.

The 98-qubit capacity was checked against [Quantinuum's Helios specifications](https://www.quantinuum.com/products-solutions/quantinuum-systems/helios) on 19 October 2026. The added local distance drivers come from the same research revision as the existing analysis; their hashes are in `source-manifest.json`.
