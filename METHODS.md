# Methods

## Data
- **Dataset:** Allen Institute for Neural Dynamics, Visual Learning dataset (pre-release, SWDB 2026 Data Book).
- **Recording:** two-photon imaging (jGCaMP8s, pan-inhibitory Slc32a1-Cre line) of 8 planes in mouse VISp, about 10 Hz per plane.
- **Cohort:** 6 mice (782149, 788406, 790322, 800792, 800995, 804363), as listed in the Data Book tutorial.
- **Access:** anonymous, from `s3://aind-open-data` (CC-BY-4.0), in November 2026. `results/tables/sessions.csv` records the exact `behavior-nwb` and `processed` asset used per session, which pins the data version.

## Sessions
- 161 recorded sessions, of which 121 are task sessions with a trials table.
- 40 have no trials table (`STAGE_0`/`STAGE_1` passive sessions and `CENTER_MOUSEMOTION` calibration) and are excluded from behaviour.
- **Phases:**
  - `TRAINING_0-2`: gratings.
  - `TRAINING_3-5`, `OPHYS_1`: images A, the familiar set.
  - `OPHYS_4`: novel images B.
  - `OPHYS_6`: images B with reward omitted.

## Behaviour (src/02_behavior.py)
- Trials that are aborted, auto-rewarded or warm-up are excluded.
- Hit rate = hits / go trials. False-alarm rate = false alarms / catch trials.
- Both use the log-linear correction (count + 0.5) / (n + 1), and d′ = z(hit) − z(false alarm).

## Neural features (src/03_neural_features.py, pre-registered, and src/07_features_v2.py, post hoc)
- **Cells:** ROIs classified `is_soma` in each plane's ROI table. ΔF/F comes from the processed NWB.
- **Population:** each session's whole recorded population. Neurons are **not** tracked across sessions (see Limitations).
- **Flashes:**
  - Change flashes: `is_change`.
  - Non-change flashes: not a change and not omitted.
  - Image features are computed only for sessions that show images.
- **Response:** mean ΔF/F in [0, 0.5) s after onset minus [−0.5, 0) s, per cell.
- **change_index:** mean response to change flashes minus mean response to non-change flashes, over cells.
- **decode_acc (pre-registered):**
  - Model: logistic regression (standardised, C = 1) on the per-cell response vector, change vs non-change.
  - Classes balanced by subsampling. Stratified 5-fold CV with shuffling.
- **decode_acc_block (post hoc):** the same decoder, scored with 5 contiguous time blocks held out in turn (GroupKFold).
- **decode_acc_nolick (post hoc):**
  - Uses only flashes with no lick within 0.75 s of onset: missed changes vs lick-free non-changes.
  - The median hit latency is about 0.53 s, so the 0–0.5 s window mostly precedes licking even in the main analysis.
- **behav_decode_acc (post hoc):** the same decoder on behavioural variables only: lick within 0.75 s, and mean running speed in the response and baseline windows.
- **mean_corr:** mean pairwise Pearson correlation of cells' ΔF/F over the session.
- **Aggregation:** session features are the mean over planes.

## Hypotheses (prereg/prereg.md, frozen in commit 0f9a8f3 before neural data were loaded)
- **H1:**
  - Target: next-session Δd′.
  - Models: baseline [d′(t), d′(t−1), stage(t) dummies] vs augmented (+ change_index, decode_acc, mean_corr).
  - Evaluation: leave-one-mouse-out, MAE, with both models on the same transitions.
  - Inference: cluster bootstrap over mice, 2,000 replicates. Supported if the upper 95% bound of ΔMAE < 0.
- **H2:**
  - Statistic: per mouse, mean decode_acc in OPHYS_6 minus OPHYS_4.
  - Test: exact two-sided sign-flip test over the 6 mice. Supported if p < 0.05.

## Post-hoc sensitivity analyses (src/06_qc.py, src/08_sensitivity.py)
- **Plane QC:** z-drift status per plane from the AIND metadata API, following the Data Book. Missing QC = "NA", never treated as a pass. Three versions: all planes, pass only, exclude failed.
- **H1 refinements:**
  - Equal-weight mean of per-mouse ΔMAE, plus the count of mice improved.
  - Same-stage-only transitions.
- **H2 decoders:** each decoder variant above.

## Reproduce
```
micromamba create -f environment.yml
python src/01_sessions.py results/tables/sessions.csv
python src/02_behavior.py results/tables/sessions.csv
python src/03_neural_features.py 8      # pre-registered features
python src/04_analysis.py               # pre-registered H1/H2, Figures 2-3
python src/05_exploratory_drift.py      # exploratory
python src/06_qc.py                     # post hoc
python src/07_features_v2.py 8          # post hoc features
python src/08_sensitivity.py            # post hoc sensitivity
python src/09_figures.py                # annotated Figures 1-2
```
