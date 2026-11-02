# C4-Helix paper reference

[c4helix_fig11a.csv](c4helix_fig11a.csv) contains three values read from Fig. 11a of [C4-Helix, arXiv:2609.03194](https://arxiv.org/abs/2609.03194). The results belong to the C4-Helix authors; our contribution is reading the published figure using gridline calibration and marker-centre matching. The readings are approximate, with estimated 3% relative uncertainty, and are not an author-supplied numerical table.

| Column | Meaning |
| --- | --- |
| `p` | Uniform circuit-noise parameter |
| `error_per_round` | Mean X/Z block error per round for full extraction over 12 memory rounds, at forced-gap threshold gamma = 0 (no postselection) |
| `relative_reading_uncertainty` | Estimated fractional uncertainty from reading the figure: 0.03 |
| `source` | Paper, figure and threshold |

For each basis, block failure probability $`p_L`$ becomes $`1-(1-p_L)^{1/12}`$ before averaging. These are block rates, not the headline's per-logical-qubit rates. Reading uncertainty is not included in our simulation's bootstrap intervals or in the check of whether each reference value lies inside them.

`analysis/aggregate.py` reads the noise points and rates into `c4_uniform`; `./run.sh baseline` refits our samples and checks that saved section. [The comparison and limits](../../EVIDENCE.md#c4-helix-baseline) explain what this supports. Credit for the reference results remains with the paper's authors; the repository's CC0 dedication covers our work, not the source publication.
