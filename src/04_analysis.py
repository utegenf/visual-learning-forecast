"""Pre-registered analyses H1 (forecasting) and H2 (extinction), plus Figures 2 and 3.

H1: leave-one-mouse-out (LOMO) forecast of next-session change in d'. The baseline uses d'(t), d'(t-1) and stage
dummies. The augmented model adds session-t neural features. Both models are evaluated on the same rows: transitions
whose session t has all neural features. Uncertainty: cluster bootstrap over mice of the held-out absolute-error
differences, 2,000 replicates.
H2: per mouse, mean decode_acc in OPHYS_6 minus OPHYS_4, two-sided exact sign-flip test over the 6 mice.
Usage: python src/04_analysis.py
"""

import itertools

import matplotlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

NEURAL = ["change_index", "decode_acc", "mean_corr"]


def load():
    beh = pd.read_csv("results/tables/behavior_sessions.csv", dtype={"mouse": str})
    nf = pd.read_csv("results/tables/neural_features.csv", dtype={"mouse": str})
    sess_nf = nf.groupby(["mouse", "session"], as_index=False)[NEURAL + ["n_cells"]].mean()
    df = beh.merge(sess_nf, on=["mouse", "session"], how="left").sort_values(["mouse", "session_number"])
    df["d_prev"] = df.groupby("mouse").dprime.shift(1)
    df["d_next"] = df.groupby("mouse").dprime.shift(-1)
    df["y"] = df.d_next - df.dprime
    return df


def lomo(df, cols):
    preds = pd.Series(index=df.index, dtype=float)
    X = pd.get_dummies(df[cols + ["stage"]], columns=["stage"], drop_first=True, dtype=float)
    for m in df.mouse.unique():
        tr, te = df.mouse != m, df.mouse == m
        preds[te] = LinearRegression().fit(X[tr], df.y[tr]).predict(X[te])
    return preds


def h1(df, rng):
    d = df.dropna(subset=["y", "dprime", "d_prev"] + NEURAL).copy()
    base, aug = ["dprime", "d_prev"], ["dprime", "d_prev"] + NEURAL
    d["err_base"] = (lomo(d, base) - d.y).abs()
    d["err_aug"] = (lomo(d, aug) - d.y).abs()
    mice = d.mouse.unique()
    by = {m: d[d.mouse == m] for m in mice}
    boots = []
    for _ in range(2000):
        s = pd.concat([by[m] for m in rng.choice(mice, len(mice))])
        boots.append(s.err_aug.mean() - s.err_base.mean())
    dmae = d.err_aug.mean() - d.err_base.mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    res = dict(n_transitions=len(d), n_mice=len(mice), mae_base=d.err_base.mean(), mae_aug=d.err_aug.mean(),
               delta_mae=dmae, ci_lo=lo, ci_hi=hi, H1_supported=bool(hi < 0))
    return res, d


def h2(df):
    s = df[df.stage.isin(["OPHYS_4", "OPHYS_6"])].groupby(["mouse", "stage"]).decode_acc.mean().unstack()
    s = s.dropna()
    diff = (s["OPHYS_6"] - s["OPHYS_4"]).to_numpy()
    obs = diff.mean()
    null = [np.mean(diff * np.array(sg)) for sg in itertools.product([1, -1], repeat=len(diff))]
    p = np.mean(np.abs(null) >= abs(obs) - 1e-12)
    return dict(n_mice=len(diff), mean_diff=obs, per_mouse=np.round(diff, 3).tolist(), p_two_sided=p,
                H2_supported=bool(p < 0.05)), s


def main():
    rng = np.random.default_rng(123)
    df = load()
    r1, d1 = h1(df, rng)
    r2, s2 = h2(df)
    pd.DataFrame([r1]).to_csv("results/tables/h1_forecast.csv", index=False)
    pd.DataFrame([{k: (str(v) if isinstance(v, list) else v) for k, v in r2.items()}]).to_csv(
        "results/tables/h2_extinction.csv", index=False)
    print("H1", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r1.items()})
    print("H2", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r2.items()})

    # Figure 2: neural change signal across sessions, per mouse
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for m, g in df.dropna(subset=["decode_acc"]).groupby("mouse"):
        axes[0].plot(g.session_number, g.decode_acc, "-o", ms=3, lw=1, label=m)
        axes[1].plot(g.session_number, g.change_index, "-o", ms=3, lw=1)
    axes[0].axhline(0.5, color="0.7", lw=0.8, ls="--")
    axes[0].set(xlabel="session", ylabel="change vs non-change decoding accuracy", title="Population decoding of image change")
    axes[1].set(xlabel="session", ylabel="ΔF/F change minus non-change", title="Change response index")
    axes[0].legend(title="mouse", fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig("results/figures/fig2_neural_across_sessions.png", dpi=200)

    # Figure 3: H1 errors and H2 paired comparison
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    e = d1.groupby("mouse")[["err_base", "err_aug"]].mean()
    x = np.arange(len(e))
    axes[0].bar(x - 0.2, e.err_base, 0.4, label="behaviour only")
    axes[0].bar(x + 0.2, e.err_aug, 0.4, label="+ neural features")
    axes[0].set_xticks(x, e.index, rotation=30, fontsize=8)
    axes[0].set(ylabel="held-out MAE of next-session Δd'",
                title=f"H1: ΔMAE {r1['delta_mae']:.3f} [{r1['ci_lo']:.3f}, {r1['ci_hi']:.3f}]")
    axes[0].legend(frameon=False, fontsize=8)
    for m, row in s2.iterrows():
        axes[1].plot([0, 1], [row["OPHYS_4"], row["OPHYS_6"]], "-o", label=m)
    axes[1].set_xticks([0, 1], ["novel images\n(rewarded)", "same images\n(reward removed)"])
    axes[1].set(ylabel="decoding accuracy", title=f"H2: mean diff {r2['mean_diff']:.3f}, p = {r2['p_two_sided']:.3f}")
    fig.tight_layout()
    fig.savefig("results/figures/fig3_h1_h2.png", dpi=200)


if __name__ == "__main__":
    main()
