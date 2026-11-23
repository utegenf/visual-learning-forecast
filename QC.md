# Quality control and post-hoc sensitivity analyses

Everything here was added after the pre-registered analysis, in response to a methodological review. The pre-registered results (`prereg/prereg.md`, `results/tables/h1_forecast.csv`, `h2_extinction.csv`) are unchanged.

## Plane-level z-drift QC (src/06_qc.py)
| Status | Planes (121 task sessions × 8) |
|---|---|
| Pass | 672 |
| Fail | 80 (in 32 sessions) |
| NA (no QC metric for that processing generation) | 216 (27 sessions have no QC at all) |

NA is never treated as a pass. Three versions are analysed: **all planes**, **pass only** (Fail and NA dropped) and **exclude failed** (Pass + NA kept).

## Decoder variants (src/07_features_v2.py)
- **Input A (raw, the pre-registered definition):** each cell's mean ΔF/F in 0–0.5 s after the flash.
- **Input B (baseline-subtracted):** the same minus the 0.5 s before the flash. This isolates the evoked response.
- **Variants run for each input:**
  - random 5-fold CV (pre-registered scoring)
  - temporal-block CV (5 contiguous time blocks)
  - lick-free flashes only (no lick within 0.75 s, so missed changes vs lick-free non-changes)
  - behaviour-only decoder (lick in window, running speed)
- The re-run uses a new random subsample, so `decode_acc` correlates r = 0.89 with the pre-registered values (mean absolute difference 0.02).

## H2: OPHYS_6 minus OPHYS_4 change-decoding accuracy, per mouse (6 mice, exact two-sided sign-flip test)
| Decoder | Input A (raw) | Input B (baseline-subtracted) |
|---|---|---|
| Random CV | −0.071, 6/6 lower, p = 0.031 | −0.067, 6/6, p = 0.031 |
| Temporal-block CV | −0.043, 6/6, p = 0.031 | −0.058, 6/6, p = 0.031 |
| **Lick-free flashes** | −0.070, 6/6, p = 0.031 | **−0.028, 4/6, p = 0.22** |
| Behaviour only | −0.134, 6/6, p = 0.031 | (same) |

These figures are for all planes. Pass-only and exclude-failed differ by at most 0.006 and give the same directions. See `results/tables/sensitivity_results_all.csv`.

**Reading:**
- **Robust:** the phase-associated decline survives plane QC, temporal-block CV and the removal of lick flashes when decoding raw activity.
- **Weak:** when the decoder sees only baseline-corrected evoked responses on lick-free flashes, the decline shrinks to −0.028 and is not consistent across mice. Evidence that the stimulus-evoked change signal itself weakens is therefore weak.
- **Behaviour changes most:** behaviour alone separates change from non-change far less in OPHYS_6 (−0.134), consistent with reduced engagement.

## H1 refinements (Input A)
| QC version | Pooled ΔMAE [95% CI] | Per-mouse mean ΔMAE | Mice improved | Same-stage-only ΔMAE [95% CI] |
|---|---|---|---|---|
| All planes | −0.022 [−0.048, 0.005] | −0.024 | 5/6 | 0.007 [−0.040, 0.082] |
| Pass only | −0.019 [−0.044, 0.008] | −0.017 | 5/6 | −0.037 [−0.077, 0.004] |
| Exclude failed | −0.027 [−0.050, −0.001] | −0.028 | 5/6 | 0.001 [−0.046, 0.067] |

The error reduction is small and similar in every version. One of six post-hoc intervals excludes zero. Given the number of variants, this is not treated as evidence, and H1 remains inconclusive.

## Not yet done
- **Tracked neurons.** All analyses use each session's whole recorded population, not the same neurons across sessions. The cross-session neuron identities (`unique_roicat_id`, in each mouse's `CTL-coreg-id-mapping` asset) refer to an earlier processing version of each session. Using them needs that version's traces, and coverage is incomplete (for example, one of mouse 782149's two OPHYS_4 sessions is not in its mapping).
