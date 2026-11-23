"""POST-HOC sensitivity features per session x plane (added after a methodological review, not pre-registered).

Adds to the pre-registered features (same definitions as 03_neural_features.py):
- decode_acc_block: change vs non-change decoding with temporal-block CV (5 contiguous time blocks held out).
- decode_acc_nolick: decoding restricted to flashes with no lick within 0.75 s of onset (missed changes vs
  lick-free non-changes), so neither the response window nor the time just after it contains licking.
- behav_decode_acc: a behaviour-only decoder (lick in window, running speed in window and baseline), random 5-fold CV.
- run_change, run_nonchange: mean running speed in the response window.
Usage: python src/07_features_v2.py [workers=8]
"""

import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import s3fs
import zarr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.simplefilter("ignore")
WIN, BASE, MIN_TRIALS = 0.5, 0.5, 10
FEATURE = "raw"  # "raw" matches the pre-registered decoder. "baseline_sub" was an earlier variant (kept for robustness)
LICK_WIN = 0.75  # 'no lick' = no lick within 0.75 s of flash onset (median hit latency here is about 0.53 s)
OUT = Path("results/tables/neural_features_v2.csv")
LOGDIR = Path("logs/neural_v2")
COLUMNS = ["mouse", "session", "session_number", "stage", "plane", "n_cells", "mean_corr", "change_index",
           "decode_acc", "decode_acc_block", "decode_acc_nolick", "n_nolick_change", "behav_decode_acc",
           "run_change", "run_nonchange"]


def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))


def window(ts, x, onsets, lo, hi):
    a, b = np.searchsorted(ts, onsets + lo), np.searchsorted(ts, onsets + hi)
    ok = b > a
    out = np.full((len(onsets),) + x.shape[1:], np.nan)
    out[ok] = np.stack([x[s:e].mean(0) for s, e in zip(a[ok], b[ok])]) if ok.any() else out[ok]
    return out


def balanced(rng, pos, neg):
    k = min(len(pos), len(neg))
    return rng.choice(pos, k, replace=False), rng.choice(neg, k, replace=False)


def random_cv(X, y):
    return float(cross_val_score(clf(), np.nan_to_num(X), y, cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean())


def block_cv(X, y, t):
    blocks = np.digitize(t, np.quantile(t, [0.2, 0.4, 0.6, 0.8]))
    if len(np.unique(blocks)) < 5:
        return np.nan
    scores = []
    for tr, te in GroupKFold(5).split(X, y, blocks):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        scores.append(clf().fit(np.nan_to_num(X[tr]), y[tr]).score(np.nan_to_num(X[te]), y[te]))
    return float(np.mean(scores)) if scores else np.nan


def one_session(s: dict) -> list:
    log = open(LOGDIR / f"{s['session']}.log", "w")
    fs = s3fs.S3FileSystem(anon=True)
    root = f"aind-open-data/{s['processed_asset']}/{s['session']}.nwb.zarr"
    if not fs.exists(root):
        print("SKIPPED: no processed NWB zarr", file=log)
        return []
    z = zarr.open_group(s3fs.S3Map(root, s3=fs), mode="r")
    stim = None
    if "intervals/stimulus_presentations" in z:
        sp = z["intervals/stimulus_presentations"]
        stim = pd.DataFrame({k: np.asarray(sp[k][:]) for k in ("start_time", "is_change", "omitted", "image_name",
                                                               "lick_latency")})
        if not stim.image_name.astype(str).str.startswith("im").any():
            stim = None
    rows, rng = [], np.random.default_rng(0)
    if stim is not None:
        stim["is_change"], stim["omitted"] = stim.is_change.astype(bool), stim.omitted.astype(bool)
        stim = stim[stim.is_change | ~stim.omitted].reset_index(drop=True)
        lat = pd.to_numeric(stim.lick_latency, errors="coerce").to_numpy()
        stim["lick_in_win"] = (lat >= 0) & (lat < LICK_WIN)
        rs_t = np.asarray(z["processing/running/speed/timestamps"][:])
        rs = np.asarray(z["processing/running/speed/data"][:], dtype=float)[:, None]
        t0 = stim.start_time.to_numpy()
        run_w, run_b = window(rs_t, rs, t0, 0, WIN)[:, 0], window(rs_t, rs, t0, -BASE, 0)[:, 0]
    for plane in sorted(k for k in z["processing"].group_keys() if k.startswith("VISp_")):
        try:
            p = z[f"processing/{plane}"]
            ts = np.asarray(p["dff_timeseries/dff_timeseries/timestamps"][:])
            dff = np.asarray(p["dff_timeseries/dff_timeseries/data"][:], dtype=np.float32)
            soma = np.asarray(p["image_segmentation/roi_table/is_soma"][:]).astype(bool)[
                np.asarray(p["dff_timeseries/dff_timeseries/rois"][:])]
            dff = dff[:, soma]
            row = dict(mouse=s["mouse"], session=s["session"], session_number=s["session_number"], stage=s["stage"],
                       plane=plane, n_cells=int(soma.sum()))
            if dff.shape[1] >= 2:
                c = np.corrcoef(np.nan_to_num(dff).T)
                row["mean_corr"] = float(np.nanmean(c[np.triu_indices_from(c, 1)]))
            if stim is not None and dff.shape[1] >= 2:
                raw = window(ts, dff, t0, 0, WIN)                 # pre-registered decoder input: 0-0.5 s mean dF/F
                resp = raw - window(ts, dff, t0, -BASE, 0)        # baseline-subtracted, used for change_index
                X_dec = raw if FEATURE == "raw" else resp
                ok = ~np.isnan(resp).any(1)
                ch, nc = np.where(ok & stim.is_change)[0], np.where(ok & ~stim.is_change)[0]
                if len(ch) >= MIN_TRIALS:
                    row["change_index"] = float(np.nanmean(resp[ch]) - np.nanmean(resp[nc]))
                    a, b = balanced(rng, ch, nc)
                    idx, y = np.r_[a, b], np.r_[np.ones(len(a)), np.zeros(len(b))]
                    row["decode_acc"] = random_cv(X_dec[idx], y)
                    row["decode_acc_block"] = block_cv(X_dec[idx], y, t0[idx])
                    B = np.c_[stim.lick_in_win.to_numpy(float), run_w, run_b]
                    row["behav_decode_acc"] = random_cv(B[idx], y)
                    row["run_change"], row["run_nonchange"] = float(np.nanmean(run_w[ch])), float(np.nanmean(run_w[nc]))
                    nl = ~stim.lick_in_win.to_numpy()
                    chn, ncn = ch[nl[ch]], nc[nl[nc]]
                    row["n_nolick_change"] = int(len(chn))
                    if len(chn) >= MIN_TRIALS:
                        a, b = balanced(rng, chn, ncn)
                        row["decode_acc_nolick"] = random_cv(X_dec[np.r_[a, b]], np.r_[np.ones(len(a)), np.zeros(len(b))])
            rows.append(row)
            print(f"{plane} ok", file=log, flush=True)
        except Exception as e:
            print(f"{plane} FAILED: {e}", file=log, flush=True)
    return rows


def main(workers: str = "8") -> None:
    LOGDIR.mkdir(parents=True, exist_ok=True)
    sess = pd.read_csv("results/tables/sessions.csv", dtype={"mouse": str})
    beh = pd.read_csv("results/tables/behavior_sessions.csv")
    sess = sess[sess.session.isin(beh.session) & sess.processed_asset.notna()]
    done = set(pd.read_csv(OUT).session) if OUT.exists() else set()
    todo = [r for r in sess.to_dict("records") if r["session"] not in done]
    print(f"task sessions={len(sess)} todo={len(todo)}", flush=True)
    with ProcessPoolExecutor(max_workers=int(workers)) as ex:
        futs = {ex.submit(one_session, r): r["session"] for r in todo}
        for f in as_completed(futs):
            try:
                rows = f.result()
            except Exception as e:
                print(f"{futs[f]} FAILED: {e}", flush=True)
                continue
            if rows:
                pd.DataFrame(rows).reindex(columns=COLUMNS).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
            print(f"{futs[f]} planes={len(rows)}", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:2])
