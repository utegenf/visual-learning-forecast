"""Per-session behaviour (hit rate, false-alarm rate, d') for the Visual Learning cohort, plus Figure 1.

Reads only the trials table of each session's behavior NWB (zarr) from s3://aind-open-data, anonymously,
and caches it as CSV under data/raw/behavior/. d' uses the log-linear correction (Hautus, 1995):
rate = (count + 0.5) / (n + 1). Trials that are aborted, auto-rewarded or warm-up are excluded.
Usage: python src/02_behavior.py results/tables/sessions.csv
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import s3fs
import zarr
from scipy.stats import norm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLS = ["start_time", "go", "catch", "hit", "miss", "false_alarm", "correct_reject", "aborted", "auto_rewarded",
        "warm_up", "change_time", "response_latency"]
CACHE = Path("data/raw/behavior")


def trials(fs, asset: str) -> pd.DataFrame:
    out = CACHE / f"{asset}_trials.csv"
    if out.exists():
        return pd.read_csv(out)
    root = zarr.open_group(s3fs.S3Map(f"aind-open-data/{asset}/behavior.nwb.zarr/intervals/trials", s3=fs), mode="r")
    df = pd.DataFrame({c: np.asarray(root[c][:]) for c in COLS if c in root})
    df.to_csv(out, index=False)
    return df


def dprime(h, n_go, fa, n_catch):
    hr, far = (h + 0.5) / (n_go + 1), (fa + 0.5) / (n_catch + 1)
    return norm.ppf(hr) - norm.ppf(far), hr, far


def main(sessions_csv: str) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    fs = s3fs.S3FileSystem(anon=True)
    sess = pd.read_csv(sessions_csv, dtype={"mouse": str})
    sess = sess.dropna(subset=["behavior_asset"])
    # Prefetch trials tables in parallel (cached to CSV), then compute sequentially.
    from concurrent.futures import ThreadPoolExecutor

    def prefetch(asset):
        try:
            trials(fs, asset)
        except Exception:
            pass
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(prefetch, sess.behavior_asset))
    rows = []
    for _, s in sess.iterrows():
        try:
            t = trials(fs, s.behavior_asset)
        except zarr.errors.GroupNotFoundError:  # no trials group: passive or calibration session (STAGE_0/1, CENTER_MOUSEMOTION)
            print(f"SKIPPED {s.session} ({s.session_type}): no trials table, not a task session", flush=True)
            continue
        except Exception as e:  # report and continue, never drop silently
            print(f"FAILED {s.session}: {e}", flush=True)
            continue
        keep = ~(t.get("aborted", False).astype(bool) | t.get("auto_rewarded", False).astype(bool) |
                 t.get("warm_up", False).astype(bool))
        t = t[keep]
        n_go, n_catch = int(t.go.astype(bool).sum()), int(t["catch"].astype(bool).sum())
        h, fa = int(t.hit.astype(bool).sum()), int(t.false_alarm.astype(bool).sum())
        d, hr, far = dprime(h, n_go, fa, n_catch) if n_go and n_catch else (np.nan, np.nan, np.nan)
        rows.append(dict(mouse=s.mouse, session=s.session, session_number=s.session_number, stage=s.stage,
                         session_type=s.session_type, n_go=n_go, n_catch=n_catch, hits=h, false_alarms=fa,
                         hit_rate=hr, fa_rate=far, dprime=d))
    beh = pd.DataFrame(rows)
    beh.to_csv("results/tables/behavior_sessions.csv", index=False)
    print(beh.groupby("mouse").agg(sessions=("session", "size"), with_dprime=("dprime", "count"),
                                   max_dprime=("dprime", "max")).round(2))

    mice = sorted(beh.mouse.unique())
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.5), sharey=True)
    for ax, m in zip(axes.flat, mice):
        b = beh[beh.mouse == m]
        ax.plot(b.session_number, b.dprime, "-o", color="0.3", ms=3, lw=1)
        for stage, colr in (("OPHYS_4", "tab:blue"), ("OPHYS_6", "tab:red")):
            x = b[b.stage == stage]
            ax.scatter(x.session_number, x.dprime, color=colr, zorder=3, s=25, label=stage)
        ax.axhline(0, color="0.8", lw=0.8)
        ax.set_title(f"mouse {m}", fontsize=9)
        ax.set_xlabel("session")
    axes[0, 0].set_ylabel("d'")
    axes[1, 0].set_ylabel("d'")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Learning and extinction: task performance across sessions (blue: novel images, red: reward removed)", fontsize=10)
    fig.tight_layout()
    fig.savefig("results/figures/fig1_learning_curves.png", dpi=200)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/tables/sessions.csv")
