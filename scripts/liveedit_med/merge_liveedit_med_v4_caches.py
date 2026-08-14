#!/usr/bin/env python3
"""Merge deterministic non-overlapping layer-21 cache shards in source order."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--source-records",type=Path,required=True);parser.add_argument("--shard",type=Path,action="append",required=True);parser.add_argument("--out",type=Path,required=True);args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);expected=[row["record_id"] for row in json.loads(args.source_records.read_text())["records"]["train"]]
    rows={}
    for shard in args.shard:
        manifest=json.loads((shard/"manifest.json").read_text())
        for row in manifest["records"]:
            if row["record_id"] in rows: raise RuntimeError("LIVEEDIT_MED_DUPLICATE_CACHE_RECORD")
            source=shard/row["file"];target=args.out/row["file"]
            os.link(source,target);rows[row["record_id"]]=row
    if set(rows)!=set(expected): raise RuntimeError(f"LIVEEDIT_MED_CACHE_MERGE_COVERAGE:{len(rows)}/{len(expected)}")
    ordered=[rows[rid] for rid in expected]
    manifest={"protocol":"LIVEEDIT_MED_V4_LAYER21_CACHE_MERGED","source_records":str(args.source_records),"records":ordered,"count":len(ordered),"base_model_frozen":True,"layer_path":"model.layers.21","shards":[str(x) for x in args.shard],"source_order_exact":True}
    (args.out/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n");print(json.dumps({"count":len(ordered),"source_order_exact":True}))


if __name__=="__main__":main()
