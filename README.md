# visual-learning-forecast

Can population activity of inhibitory neurons in mouse visual cortex forecast learning? And does it change when a learned reward is removed? This is a pre-registered analysis of the Allen Institute Visual Learning dataset (6 mice, 121 task sessions, two-photon imaging across training, novel images and extinction).

## Results
| | Result |
|---|---|
| Learning curves (Fig. 1) | d′ rises over training with large day-to-day variability, and falls when the reward is removed in 5 of 6 mice. |
| **H1** (pre-registered): do neural features improve the next-session forecast of Δd′? | **Not supported.** Leave-one-mouse-out MAE 0.442 → 0.417 (ΔMAE −0.025, 95% CI −0.051 to +0.001). Lower error in 5 of 6 held-out mice. |
| **H2** (pre-registered): does change decoding differ with vs without reward? | **Supported.** Decoding accuracy fell in all 6 mice (mean −0.070, exact p = 0.031). |
| Exploratory | The drop exceeds each mouse's prior trend (6/6). But novelty loss and behavioural changes are confounded with reward removal, so the design cannot separate them. |

Figures: `results/figures/fig1_learning_curves.png`, `fig2_neural_across_sessions.png`, `fig3_h1_h2.png`. The plan is in `prereg/prereg.md`, committed before any neural data was loaded.

## Pipeline
```
src/01_sessions.py         session table (public AIND metadata API + S3 listing)
src/02_behavior.py         per-session hit/false-alarm rates, d' (log-linear correction), Fig. 1
src/03_neural_features.py  streams processed NWB (zarr) per session, soma ROIs only, per-plane features
src/04_analysis.py         H1 (LOMO forecast, cluster bootstrap) and H2 (exact sign-flip test), Figs. 2-3
```
Data is streamed anonymously from `s3://aind-open-data` and not stored here.

## Limitations
- 6 mice. Mice are the independent units, and cells and planes are nested within session features.
- Calcium imaging measures activity, not synaptic change. All results are correlational.
- Only inhibitory neurons are labelled in this dataset. Subclass analyses are out of scope.
- The dataset is pre-release, and the exact asset names used are in `results/tables/sessions.csv`.
- Training stages use different stimuli (gratings, then images), so d′ is not strictly comparable across stages.

## Data and credit
Allen Institute for Neural Dynamics, Visual Learning dataset (SWDB 2026 Data Book). The data was accessed in November 2026 from https://registry.opendata.aws/allen-nd-open-data (CC-BY-4.0). The code here is released under Apache-2.0.
