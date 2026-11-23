"""Per-plane z-drift QC for each processed asset used, from the public AIND metadata API (POST-HOC addition).

Follows the SWDB 2026 Data Book tutorial: quality_control.metrics entries whose name matches z-drift, latest
status in status_history. The plane name leads or trails the metric name. Planes without a metric stay "NA".
They are never treated as a pass.
Usage: python src/06_qc.py
"""

import json
import re
import urllib.parse
import urllib.request

import pandas as pd

API = "https://api.allenneuraldynamics.org/v2/metadata_index/data_assets"
BATCH = 2


def fetch(names):
    q = urllib.parse.urlencode({"filter": json.dumps({"name": {"$in": names}}),
                                "projection": json.dumps({"name": 1, "quality_control.metrics.name": 1, "quality_control.metrics.status_history": 1}), "limit": 100})
    with urllib.request.urlopen(f"{API}?{q}", timeout=120) as r:
        return json.load(r)


def main():
    sess = pd.read_csv("results/tables/sessions.csv", dtype={"mouse": str})
    beh = pd.read_csv("results/tables/behavior_sessions.csv")
    sess = sess[sess.session.isin(beh.session) & sess.processed_asset.notna()]
    assets = sess.processed_asset.tolist()
    rows = []
    for i in range(0, len(assets), BATCH):
        for doc in fetch(assets[i:i + BATCH]):
            for m in ((doc.get("quality_control") or {}).get("metrics") or []):
                name = str(m.get("name"))
                if not re.search(r"z-?drift", name, re.I):
                    continue
                plane = re.search(r"(VISp_\d+)", name)
                hist = m.get("status_history") or []
                rows.append(dict(processed_asset=doc["name"], plane=plane.group(1) if plane else None,
                                 metric=name, status=hist[-1].get("status") if hist else None))
    qc = pd.DataFrame(rows).drop_duplicates(["processed_asset", "plane"], keep="last")
    planes = sess[["mouse", "session", "stage", "processed_asset"]].merge(
        pd.DataFrame({"plane": [f"VISp_{k}" for k in range(8)]}), how="cross")
    out = planes.merge(qc[["processed_asset", "plane", "status"]], on=["processed_asset", "plane"], how="left")
    out["qc"] = out.status.fillna("NA")
    out.to_csv("results/tables/plane_qc.csv", index=False)
    print(out.qc.value_counts().to_string())
    print("sessions with >=1 failed plane:", out[out.qc.str.lower() == "fail"].session.nunique())
    print("sessions with no QC:", out.groupby("session").qc.apply(lambda s: (s == "NA").all()).sum())


if __name__ == "__main__":
    main()
