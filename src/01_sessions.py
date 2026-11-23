"""Build the session table for the Allen Visual Learning cohort (6 mice).

Sources (public, anonymous):
- AIND metadata API (docdb): session type per raw multiplane-ophys session.
- s3://aind-open-data: latest behavior-nwb and processed assets per session.
The mouse list is the one used in the SWDB 2026 Data Book tutorial.
Usage: python src/01_sessions.py results/tables/sessions.csv
"""

import json
import re
import sys
import urllib.parse
import urllib.request

import pandas as pd
import s3fs

MICE = ["782149", "790322", "788406", "800792", "800995", "804363"]
API = "https://api.allenneuraldynamics.org/v2/metadata_index/data_assets"
RAW = re.compile(r"^multiplane-ophys_(\d+)_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})$")


def api(filter_, projection, limit=1000):
    q = urllib.parse.urlencode({"filter": json.dumps(filter_), "projection": json.dumps(projection), "limit": limit})
    with urllib.request.urlopen(f"{API}?{q}", timeout=60) as r:
        return json.load(r)


def main(out_csv: str) -> None:
    recs = api({"subject.subject_id": {"$in": MICE}},
               {"name": 1, "acquisition.acquisition_type": 1, "session.session_type": 1})
    types = {}
    for r in recs:
        m = RAW.match(r["name"])
        if m:
            t = (r.get("acquisition") or {}).get("acquisition_type") or (r.get("session") or {}).get("session_type")
            types[r["name"]] = t

    fs = s3fs.S3FileSystem(anon=True)
    names = [p.split("/", 1)[1] for p in fs.ls("aind-open-data") if any(f"multiplane-ophys_{m}_" in p for m in MICE)]
    rows = []
    for raw in sorted(n for n in names if RAW.match(n)):
        mouse, stamp = RAW.match(raw).groups()
        beh = sorted(n for n in names if n.startswith(raw + "_behavior-nwb_"))
        proc = sorted(n for n in names if re.fullmatch(re.escape(raw) + r"_processed_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", n))
        rows.append(dict(mouse=mouse, session=raw, date=stamp[:10], session_type=types.get(raw),
                         behavior_asset=beh[-1] if beh else None, processed_asset=proc[-1] if proc else None))
    df = pd.DataFrame(rows).sort_values(["mouse", "date"])
    # Fill missing session types from the behavior asset's session.json (field "session_type").
    for i in df.index[df.session_type.isna() & df.behavior_asset.notna()]:
        try:
            with fs.open(f"aind-open-data/{df.at[i, 'behavior_asset']}/session.json") as fh:
                df.at[i, "session_type"] = json.load(fh).get("session_type")
        except FileNotFoundError:
            pass
    df["session_number"] = df.groupby("mouse").cumcount() + 1
    df["stage"] = df.session_type.str.extract(r"^(TRAINING_\d|OPHYS_\d|STAGE_\d)", expand=False)
    df.to_csv(out_csv, index=False)
    print(df.groupby("mouse").agg(n=("session", "size"), behavior=("behavior_asset", "count"),
                                  processed=("processed_asset", "count"), typed=("session_type", "count")))
    print(df.stage.value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/tables/sessions.csv")
