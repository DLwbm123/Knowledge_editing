#!/usr/bin/env python3
"""Create the source-pinned deterministic MedMKEB split without copying images."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from methods.liveedit_med.data import deterministic_split


def main():
    p=argparse.ArgumentParser(); p.add_argument("--source-json",type=Path,required=True); p.add_argument("--image-root",type=Path,required=True); p.add_argument("--out-dir",type=Path,required=True); a=p.parse_args()
    a.out_dir.mkdir(parents=True,exist_ok=False); raw=json.loads(a.source_json.read_text()); split=deterministic_split(raw,a.image_root)
    compact={name:[{"record_id":r["record_id"],"selection_hash":r["selection_hash"]} for r in rows] for name,rows in split.items()}
    compact["counts"]={name:len(rows) for name,rows in split.items()}; compact["record953_excluded"]=all(r["record_id"]!="953" for rows in split.values() for r in rows)
    (a.out_dir/"edit_level_split.json").write_text(json.dumps(compact,indent=2,sort_keys=True)+"\n")
    with (a.out_dir/"medical_pool_audit.csv").open("w",newline="") as h:
        w=csv.DictWriter(h,fieldnames=["split","record_id","selection_hash","has_native","has_textual","has_visual","has_paired","has_image_locality"]); w.writeheader()
        for name,rows in split.items():
            for r in rows:w.writerow({"split":name,"record_id":r["record_id"],"selection_hash":r["selection_hash"],"has_native":True,"has_textual":True,"has_visual":True,"has_paired":True,"has_image_locality":True})
    (a.out_dir/"source_records.json").write_text(json.dumps({"source_json":str(a.source_json),"image_root":str(a.image_root),"records":split},indent=2,sort_keys=True)+"\n")
    print(json.dumps(compact["counts"],sort_keys=True))


if __name__=="__main__": main()
