"""POST-HOC sensitivity analyses (after the methodological review). The pre-registered results in 04_analysis.py are unchanged.

QC versions: all planes / pass only (Fail and NA excluded) / exclude failed (Pass + NA kept).
For each version:
- H1 pooled: LOMO dMAE with cluster bootstrap over mice, as pre-registered.
- H1 per-mouse (equal weight): mean of the 6 per-mouse dMAE.
- H1 same-stage: only transitions where session t and t+1 are the same stage.
- H2 with each decoder: random-fold (pre-registered definition), temporal-block CV, no-lick flashes, behaviour-only.
Usage: python src/08_sensitivity.py
"""

import importlib.util
import itertools

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("a", "src/04_analysis.py")
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)
NEURAL = ["change_index", "decode_acc", "mean_corr"]
DECODERS = ["decode_acc", "decode_acc_block", "decode_acc_nolick", "behav_decode_acc"]


def session_table(nf):
    beh = pd.read_csv("results/tables/behavior_sessions.csv", dtype={"mouse": str})
    s = nf.groupby(["mouse", "session"], as_index=False)[NEURAL + DECODERS[1:]].mean()
    df = beh.merge(s, on=["mouse", "session"], how="left").sort_values(["mouse", "session_number"])
    df["d_prev"] = df.groupby("mouse").dprime.shift(1)
    df["d_next"] = df.groupby("mouse").dprime.shift(-1)
    df["stage_next"] = df.groupby("mouse").stage.shift(-1)
    df["y"] = df.d_next - df.dprime
    return df


def h1(df, rng, same_stage=False):
    d = df.dropna(subset=["y", "dprime", "d_prev"] + NEURAL).copy()
    if same_stage:
        d = d[d.stage == d.stage_next]
    if d.mouse.nunique() < 3:
        return dict(n=len(d))
    d["eb"] = (A.lomo(d, ["dprime", "d_prev"]) - d.y).abs()
    d["ea"] = (A.lomo(d, ["dprime", "d_prev"] + NEURAL) - d.y).abs()
    mice = d.mouse.unique()
    by = {m: d[d.mouse == m] for m in mice}
    boots = [pd.concat([by[m] for m in rng.choice(mice, len(mice))]).pipe(lambda s: s.ea.mean() - s.eb.mean())
             for _ in range(2000)]
    per = d.groupby("mouse").apply(lambda g: g.ea.mean() - g.eb.mean())
    return dict(n=len(d), mae_base=d.eb.mean(), mae_aug=d.ea.mean(), dmae=d.ea.mean() - d.eb.mean(),
                ci_lo=np.percentile(boots, 2.5), ci_hi=np.percentile(boots, 97.5),
                dmae_per_mouse_mean=per.mean(), mice_improved=int((per < 0).sum()), n_mice=len(per))


def h2(df, col):
    s = df[df.stage.isin(["OPHYS_4", "OPHYS_6"])].groupby(["mouse", "stage"])[col].mean().unstack().dropna()
    if len(s) < 2:
        return dict(n_mice=len(s))
    diff = (s["OPHYS_6"] - s["OPHYS_4"]).to_numpy()
    null = [np.mean(diff * np.array(sg)) for sg in itertools.product([1, -1], repeat=len(diff))]
    return dict(n_mice=len(diff), mean_diff=diff.mean(), mice_lower=int((diff < 0).sum()),
                p_two_sided=float(np.mean(np.abs(null) >= abs(diff.mean()) - 1e-12)),
                mean_O4=s["OPHYS_4"].mean(), mean_O6=s["OPHYS_6"].mean())


def main():
    rng = np.random.default_rng(123)
    nf = pd.read_csv("results/tables/neural_features_v2.csv", dtype={"mouse": str})
    qc = pd.read_csv("results/tables/plane_qc.csv", dtype={"mouse": str})[["session", "plane", "qc"]]
    nf = nf.merge(qc, on=["session", "plane"], how="left")
    nf["qc"] = nf.qc.fillna("NA")
    versions = {"all_planes": nf, "pass_only": nf[nf.qc == "Pass"], "exclude_failed": nf[nf.qc != "Fail"]}
    rows = []
    for v, d in versions.items():
        df = session_table(d)
        for name, res in (("H1_pooled_and_per_mouse", h1(df, rng)), ("H1_same_stage_only", h1(df, rng, True))):
            rows.append(dict(qc_version=v, analysis=name, **res))
        for col in DECODERS:
            rows.append(dict(qc_version=v, analysis=f"H2_{col}", **h2(df, col)))
    out = pd.DataFrame(rows)
    out.to_csv("results/tables/sensitivity_results.csv", index=False)
    pd.set_option("display.width", 250)
    print(out.round(3).to_string(index=False))
    na = nf.groupby("session").qc.apply(lambda s: (s == "NA").all())
    print("sessions with no QC at all:", int(na.sum()), "| planes by QC:", nf.qc.value_counts().to_dict())


if __name__ == "__main__":
    main()
