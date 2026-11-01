US$0.1 million to US$10 million

My estimate of the additional value to the quantum-computing industry over the next ten years, in 2026 US dollars, is centred around **US$1 million**. This is a range for expected value; realised gains could be zero.

The useful contribution is an implemented Carbon-Helix extraction circuit, adaptive measurement schedules and reusable evidence about their failure modes. At the principal simulated noise point, R8 reduces two-qubit gates from 560 to about 216 per round and modelled round time from 788 to 387 milliseconds, using 92 physical qubits for two logical qubits. Its central memory-error rate per second is similar to full extraction. The headline 24.4-fold advantage compares against C4-Helix; R8 improves on full Carbon's central rate by only about 4%. These figures support a possible saving in error-correction resources. [Repository results](RESULTS.md), [comparison data](figures/data/headline.csv).

I use two additive benefit channels. The probabilities and conditional gains below are my judgements, rather than measured adoption rates.

| Benefit | Chance of realisation | Industry gain if realised | Expected contribution |
| --- | ---: | ---: | ---: |
| Reuse saves research and validation effort | 25% to 50% | US$0.2m to US$1m | US$0.05m to US$0.5m |
| Hardware adoption delivers additional operating savings | 1% to 10% | US$5m to US$100m | US$0.05m to US$10m |

The first channel assumes one to four researcher-years saved across adopting teams, net of learning and integration effort, at an assumed US$200,000 to US$250,000 per year including overheads. As a salary benchmark, the US median for computer research scientists was US$140,300 in May 2025. The benefit is future effort avoided by others. [US Bureau of Labor Statistics](https://www.bls.gov/ooh/computer-and-information-technology/computer-and-information-research-scientists.htm).

The second assumes US$100m to US$1bn of relevant hardware and operating expenditure across adopters over a decade, with a net 5% to 10% saving attributable to this work after implementation costs and competing improvements. These are scenario assumptions. For scale, Quantinuum reported US$165m of net research and development expenditure in 2025, only part of which could be affected by extraction improvements. This channel excludes the research labour counted above. [Quantinuum prospectus](https://www.sec.gov/Archives/edgar/data/2110105/000162828026041003/quantinuum-424b4.htm). Adding the probability-weighted contributions gives approximately the opening range; a central calculation is 30% × US$0.5m + 5% × US$20m = US$1.15m.

The low deployment probability reflects substantial remaining uncertainty. The headline uses short memory simulations, only 41% of R8's error estimate comes from observed fault strata, and R8 lacks a whole-circuit distance bound. Hardware compilation, transport, leakage and useful logical computation remain unvalidated for this implementation. [Evidence and limits](EVIDENCE.md), [noise model](src/hase/helios.py). Existing [adaptive extraction research](https://arxiv.org/abs/2502.14835) and [hardware colour-code results](https://www.quantinuum.com/fault-tolerance-by-2029) also limit the additional benefit attributable to this particular work. Its value depends on saving effort or improving deployed systems beyond those alternatives.
