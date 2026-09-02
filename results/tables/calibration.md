| Entailment cut | Fitted on dev (kappa) | Reported on held-out (kappa) |
|---|---|---|
| default, entail > 0.5 | 0.673 | 0.737 |
| calibrated, entail > 0.95 | 0.740 | 0.771 |

Fitted on 40 development labels; reported on 40 held-out labels the sweep never saw.

The contradiction threshold is not swept: a claim counts as unsupported when it is not entailed, so that threshold only decides whether such a claim is reported as contradicted or not-addressed and cannot change any label.

Identification: of 24 threshold-sensitive dev traces 2 change label between the two cuts, and of 24 held-out traces 1 do. The kappa difference therefore rests on a handful of traces, so the fitted cut is reported as a sensitivity analysis and NOT adopted as the default: the direction of the finding is robust to the threshold, its magnitude is not pinned down by this many annotations.
