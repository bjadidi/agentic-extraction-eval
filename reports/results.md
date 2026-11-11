| Strategy | Auto-accepted accuracy | Silent errors | Automation | Straight-through docs | Final accuracy | Model $/doc | Total $/doc | p50 latency (s) |
|---|---|---|---|---|---|---|---|---|
| small model only | 90.0% | 10.00% | 100.0% | 100% | 90.0% | 0.00020 | 0.000 | 1.2 |
| large model only | 98.8% | 1.25% | 100.0% | 100% | 98.8% | 0.00240 | 0.002 | 3.9 |
| small + validation gate -> human | 99.1% | 0.83% | 90.0% | 40% | 99.2% | 0.00020 | 0.320 | 61.2 |
| cascade without grounding check | 97.3% | 2.71% | 98.5% | 88% | 97.3% | 0.00066 | 0.047 | 3.8 |
| cascade: small -> large -> human | 99.2% | 0.83% | 98.5% | 88% | 99.2% | 0.00074 | 0.047 | 3.8 |

Primary-model calibration (cascade run):

| Confidence | n | Accuracy |
|---|---|---|
| 0.2-0.4 | 10 | 10.0% |
| 0.4-0.6 | 11 | 18.2% |
| 0.6-0.8 | 44 | 88.6% |
| 0.8-1.0 | 415 | 94.0% |

Per-layout final accuracy (cascade): layout 0: 100.0%, layout 1: 97.5%, layout 2: 99.2%, layout 3: 100.0%
