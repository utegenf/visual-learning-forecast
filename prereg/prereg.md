# Pre-registration: does visual-cortex population activity forecast learning?

Frozen by the commit that adds this file, before any neural data are loaded. Behaviour (Figure 1) has been seen. Neural activity has not.

## Data
Allen Institute Visual Learning dataset (SWDB 2026 Data Book). Two-photon imaging of inhibitory neurons in mouse VISp across learning of a go/no-go image-change detection task, then novel images (OPHYS_4) and reward removal (OPHYS_6). 6 mice. Public bucket `s3://aind-open-data` (CC-BY-4.0). This is a pre-release dataset, so the exact asset names are recorded in `results/tables/sessions.csv`.

## Session-level neural features (per imaging plane, soma ROIs only, then averaged over planes)
- `resp_change`: mean ΔF/F in 0–0.5 s after image-change flashes minus the 0.5 s before, averaged over cells.
- `resp_nonchange`: the same for non-change, non-omitted image flashes.
- `change_index` = resp_change − resp_nonchange.
- `decode_acc`: 5-fold cross-validated logistic-regression accuracy for change vs non-change flashes, from the population response vector (0–0.5 s mean ΔF/F per cell). Classes are balanced by subsampling non-change flashes. Chance = 0.5.
- `mean_corr`: mean pairwise Pearson correlation of cells' ΔF/F over the session.
- `n_cells`: number of soma ROIs.
Sessions without image-change flashes (gratings-only training stages) get NaN for the image features.

## H1 (forecasting)
- **Target:** Δd′ from session t to session t+1, within a mouse, task sessions only.
- **Baseline:** linear model with d′(t), d′(t−1) and stage dummies.
- **Augmented:** baseline + session-t features (`change_index`, `decode_acc`, `mean_corr`).
- **Evaluation:** leave-one-mouse-out, using mean absolute error on the held-out mouse, pooled over all held-out predictions.
- **Statistic:** ΔMAE = MAE(augmented) − MAE(baseline). Uncertainty comes from a cluster bootstrap over mice (2,000 replicates).
- **Decision rule:** H1 is supported if the upper 95% bound of ΔMAE is below 0.
- **Expectation:** with 6 mice and about 100 transitions, the test is underpowered, so a null result is likely and will be reported as such.

## H2 (extinction)
- **Statistic:** per mouse, mean `decode_acc` in OPHYS_6 minus mean `decode_acc` in OPHYS_4. Same images and schedule, but reward is removed.
- **Test:** sign-flip permutation test over the 6 mice (all 64 sign patterns), two-sided.
- **Decision rule:** p < 0.05. With 6 mice the smallest possible two-sided p is 2/64 = 0.031, so only a unanimous direction can pass.

## Units of inference
Mice are the independent units. Cells and planes are nested measurements that feed session-level features. No test treats cells as independent.

## Not claimed
Calcium imaging measures activity, not synaptic change. All results are correlational. Subclass (PV/SST/VIP/LAMP5) analyses are out of scope.

## Results (analysis exactly as pre-registered)
- **Implementation detail the plan left open:** both H1 models are scored on the same 73 transitions, those whose session t has all neural features. 121 task sessions × 8 planes were extracted, with no failures.
- **H1, not supported.** Held-out MAE of next-session Δd′ was 0.442 for the behaviour-only model and 0.417 with neural features. ΔMAE −0.025, 95% CI −0.051 to +0.001. The upper bound is just above 0, so the rule is not met. Neural features lowered error in 5 of 6 held-out mice.
- **H2, supported.** Change-vs-non-change decoding accuracy fell from OPHYS_4 to OPHYS_6 in all 6 mice: mean −0.070, per mouse −0.03 to −0.09, exact two-sided sign-flip p = 0.031. Interpretation: a consistent phase-associated change. It is not evidence that reward removal caused the change (see the confound below).

## Exploratory (not pre-registered)
- **Drift check.** The OPHYS_4 → OPHYS_6 drop exceeds what each mouse's pre-OPHYS_4 trend predicts, in 6 of 6 mice (mean excess −0.073). See `results/tables/exploratory_h2_drift_check.csv`.
- **Confound.** OPHYS_6 always follows OPHYS_4, so images B are novel in OPHYS_4 and familiar by OPHYS_6. The drop can come from loss of novelty as well as from removing the reward, and behaviour (licking, engagement) also changes. This design cannot separate these.
