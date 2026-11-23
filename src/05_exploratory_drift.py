"""EXPLORATORY (not pre-registered): is the OPHYS_4 -> OPHYS_6 decoding drop larger than each mouse's prior trend?

The expected drift is the slope of decode_acc over image sessions before OPHYS_4, multiplied by the session gap.
Usage: python src/05_exploratory_drift.py
"""
import numpy as np
import pandas as pd

nf = pd.read_csv("results/tables/neural_features.csv", dtype={"mouse": str})
s = (nf.groupby(["mouse", "session", "session_number", "stage"], as_index=False).decode_acc.mean()
     .dropna().sort_values(["mouse", "session_number"]))
rows = []
for m, g in s.groupby("mouse"):
    o4, o6 = g[g.stage == "OPHYS_4"], g[g.stage == "OPHYS_6"]
    pre = g[g.session_number < o4.session_number.min()]
    slope = np.polyfit(pre.session_number, pre.decode_acc, 1)[0] if len(pre) >= 3 else np.nan
    gap = o6.session_number.mean() - o4.session_number.mean()
    rows.append(dict(mouse=m, drop=o6.decode_acc.mean() - o4.decode_acc.mean(), drift_expected=slope * gap,
                     gap_sessions=gap, o4_vs_last3_pre=o4.decode_acc.mean() - pre.decode_acc.tail(3).mean()))
r = pd.DataFrame(rows)
r["drop_minus_drift"] = r["drop"] - r.drift_expected
print(r.round(3).to_string(index=False))
r.to_csv("results/tables/exploratory_h2_drift_check.csv", index=False)
