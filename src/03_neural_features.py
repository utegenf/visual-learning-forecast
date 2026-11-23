"""Session-level neural features for every task session (definitions fixed in prereg/prereg.md).

Streams each session's processed NWB (zarr) from s3://aind-open-data anonymously. Raw traces are not
stored. Output: one row per session x imaging plane in results/tables/neural_features.csv.
Usage: python src/03_neural_features.py [workers=8]
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
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.simplefilter("ignore")
WIN, BASE = 0.5, 0.5
OUT = Path("results/tables/neural_features.csv")
LOGDIR = Path("logs/neural")
COLUMNS = ["plane", "n_cells", "mean_corr", "resp_change", "resp_nonchange", "change_index", "decode_acc", "n_change",
           "n_nonchange_used", "mouse", "session", "session_number", "stage"]


def window_means(ts, dff, onsets):
    """Mean dF/F per cell in [t, t+WIN) and [t-BASE, t). Returns (n_onsets, n_cells) arrays."""
    a0, a1 = np.searchsorted(ts, onsets), np.searchsorted(ts, onsets + WIN)
    b0 = np.searchsorted(ts, onsets - BASE)
    ok = (a1 > a0) & (a0 > b0)
    resp = np.stack([dff[s:e].mean(0) for s, e in zip(a0[ok], a1[ok])]) if ok.any() else np.empty((0, dff.shape[1]))
    base = np.stack([dff[s:e].mean(0) for s, e in zip(b0[ok], a0[ok])]) if ok.any() else np.empty((0, dff.shape[1]))
    return resp, base


def plane_features(z, plane, stim, rng):
    p = z[f"processing/{plane}"]
    ts = np.asarray(p["dff_timeseries/dff_timeseries/timestamps"][:])
    dff = np.asarray(p["dff_timeseries/dff_timeseries/data"][:], dtype=np.float32)
    rois = np.asarray(p["dff_timeseries/dff_timeseries/rois"][:])
    soma = np.asarray(p["image_segmentation/roi_table/is_soma"][:]).astype(bool)[rois]
    dff = dff[:, soma]
    row = dict(plane=plane, n_cells=int(soma.sum()))
    if dff.shape[1] < 2:
        return row
    c = np.corrcoef(np.nan_to_num(dff).T)
    row["mean_corr"] = float(np.nanmean(c[np.triu_indices_from(c, 1)]))
    if stim is None:
        return row
    ch = stim.start_time[stim.is_change].to_numpy()
    nc = stim.start_time[~stim.is_change & ~stim.omitted].to_numpy()
    if len(ch) < 10:
        return row
    rc, bc = window_means(ts, dff, ch)
    rn, bn = window_means(ts, dff, nc)
    row["resp_change"] = float(np.nanmean(rc - bc))
    row["resp_nonchange"] = float(np.nanmean(rn - bn))
    row["change_index"] = row["resp_change"] - row["resp_nonchange"]
    k = min(len(rc), len(rn))
    X = np.vstack([rc[rng.choice(len(rc), k, replace=False)], rn[rng.choice(len(rn), k, replace=False)]])
    y = np.r_[np.ones(k), np.zeros(k)]
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    row["decode_acc"] = float(cross_val_score(clf, np.nan_to_num(X), y, cv=cv, scoring="accuracy").mean())
    row["n_change"], row["n_nonchange_used"] = int(len(rc)), int(k)
    return row


def one_session(s: dict) -> list:
    log = open(LOGDIR / f"{s['session']}.log", "w")
    fs = s3fs.S3FileSystem(anon=True)
    root = f"aind-open-data/{s['processed_asset']}/{s['session']}.nwb.zarr"
    if not fs.exists(root):
        print(f"SKIPPED: no processed NWB zarr at {root}", file=log)
        return []
    z = zarr.open_group(s3fs.S3Map(root, s3=fs), mode="r")
    stim = None
    if "intervals/stimulus_presentations" in z:
        sp = z["intervals/stimulus_presentations"]
        stim = pd.DataFrame({k: np.asarray(sp[k][:]) for k in ("start_time", "is_change", "omitted", "image_name")})
        stim["is_change"], stim["omitted"] = stim.is_change.astype(bool), stim.omitted.astype(bool)
        if not stim.image_name.astype(str).str.startswith("im").any():  # gratings-only stage: no image features
            stim = None
    rng = np.random.default_rng(0)
    rows = []
    for plane in sorted(k for k in z["processing"].group_keys() if k.startswith("VISp_")):
        try:
            r = plane_features(z, plane, stim, rng)
        except Exception as e:
            print(f"{plane} FAILED: {e}", file=log)
            continue
        r.update(mouse=s["mouse"], session=s["session"], session_number=s["session_number"], stage=s["stage"])
        rows.append(r)
        print(f"{plane} ok n_cells={r['n_cells']}", file=log, flush=True)
    return rows


def main(workers: str = "8") -> None:
    LOGDIR.mkdir(parents=True, exist_ok=True)
    sess = pd.read_csv("results/tables/sessions.csv", dtype={"mouse": str})
    beh = pd.read_csv("results/tables/behavior_sessions.csv", dtype={"mouse": str})
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
            if rows:  # fixed column order, so sessions without image features keep the CSV aligned
                pd.DataFrame(rows).reindex(columns=COLUMNS).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
            print(f"{futs[f]} planes={len(rows)}", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:2])
