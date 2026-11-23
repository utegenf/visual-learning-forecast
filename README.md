# Learning-related changes in visual cortical population activity: a longitudinal analysis of acquisition and reward omission

Do population measurements of inhibitory neurons in mouse visual cortex improve prediction of next-session learning? And does change-related population activity differ between the rewarded and reward-omission phases? This is a pre-registered analysis of the Allen Institute Visual Learning dataset (6 mice, 121 task sessions, two-photon imaging across training, novel images and extinction).

## Results
| | Result |
|---|---|
| Learning curves (Fig. 1) | d′ rises over training with large day-to-day variability, and falls when the reward is removed in 5 of 6 mice. |
| **H1** (pre-registered): do neural features improve the next-session forecast of Δd′? | **Inconclusive (rule not met).** Leave-one-mouse-out mean absolute error 0.442 → 0.417, a 5.7% relative reduction (ΔMAE −0.025, 95% CI −0.051 to +0.001). Lower error in 5 of 6 held-out mice. |
| **H2** (pre-registered): does change decoding differ between the rewarded (OPHYS_4) and reward-omission (OPHYS_6) phases? | **Supported by the pre-registered rule.** Decoding accuracy was lower in the omission phase in all 6 mice (mean −0.070, exact two-sided sign-flip p = 0.031). Read this as a consistent phase-associated change, not evidence that reward removal caused it. |
| Exploratory | The drop exceeds each mouse's prior trend (6/6). But novelty loss and behavioural changes are confounded with reward removal, so the design cannot separate them. |

Figures: `results/figures/fig1_learning_curves.png`, `fig2_neural_across_sessions.png`, `fig3_h1_h2.png`. The plan is in `prereg/prereg.md`, committed before any neural data was loaded.

## Post-hoc sensitivity analyses (see QC.md)
These were added after a methodological review, and the pre-registered results above are unchanged.
- **Plane QC:** 80 of 968 planes failed z-drift QC and 216 have no QC. Excluding them leaves H1 and H2 essentially unchanged.
- **H2 holds** with temporal-block cross-validation, and with lick-free flashes when decoding raw activity (6/6 mice in each).
- **H2 is weak in the strictest test:** baseline-corrected evoked responses on lick-free flashes give −0.028, 4/6 mice, p = 0.22. The decline is best described as a change in population activity during the omission phase. Evidence that the evoked change signal itself weakens is weak.
- **H1** stays small and inconclusive in every version (pooled ΔMAE −0.019 to −0.027, with 5/6 mice improved).
- **Not yet done:** analysis of the same tracked neurons across sessions.

## Possible next analysis
Test whether population representations change gradually across acquisition, not only between phases, for a more direct link to experience-dependent learning.

## Pipeline
```
src/01_sessions.py         session table (public AIND metadata API + S3 listing)
src/02_behavior.py         per-session hit/false-alarm rates, d' (log-linear correction), Fig. 1
src/03_neural_features.py  streams processed NWB (zarr) per session, soma ROIs only, per-plane features
src/04_analysis.py         H1 (LOMO forecast, cluster bootstrap) and H2 (exact sign-flip test), Figs. 2-3
src/05-09                  exploratory and post-hoc: drift check, plane QC, sensitivity features, sensitivity tests, annotated figures
```
Full definitions are in METHODS.md, and the reproduction commands are at its end.
Data is streamed anonymously from `s3://aind-open-data` and not stored here.

## Limitations
- Calcium imaging gives an indirect proxy of neural activity. These are learning-related changes in activity, not direct evidence of synaptic plasticity or changes in connectivity.
- Reward omission always followed novel-image exposure, so reward removal, habituation to the images and session order cannot be separated.
- 6 mice. Mice are the independent units, and cells and planes are nested within session features.
- All results are correlational.
- Only inhibitory neurons are labelled in this dataset. Subclass analyses are out of scope.
- The dataset is pre-release, and the exact asset names used are in `results/tables/sessions.csv`.
- Training stages use different stimuli (gratings, then images), so d′ is not strictly comparable across stages.

## Data and credit
Allen Institute for Neural Dynamics, Visual Learning dataset (SWDB 2026 Data Book). The data was accessed in November 2026 from https://registry.opendata.aws/allen-nd-open-data (CC-BY-4.0). The code here is released under Apache-2.0.
