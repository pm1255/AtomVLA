"""Merge shard receipts into a read-only progress report.

It never copies records or marks unresolved episodes train eligible.  The
individual shard roots remain the source of truth for resume and release.
"""
import argparse, json, time
from pathlib import Path


def aggregate(root):
    root = Path(root); shards = []
    for p in sorted(root.glob("shard_*/run_result.json")):
        d = json.loads(p.read_text()); shards.append(d)
    totals = {}
    for d in shards:
        for source, c in d.get("counts", {}).items():
            out = totals.setdefault(source, {})
            for k, v in c.items(): out[k] = out.get(k, 0) + v
    planned = sum(d.get("planned_episodes", 0) for d in shards)
    processed = sum(sum(c.get("processed", 0) for c in d.get("counts", {}).values()) for d in shards)
    result = {"schema": "atomvla_large_scale_progress_v1", "shards": len(shards),
              "planned_episodes": planned, "processed_episodes": processed,
              "counts": totals, "full_dataset_ready": False,
              "training_ready": False, "updated_at": time.time()}
    (root / "aggregate_progress.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--root", required=True)
    print(json.dumps(aggregate(p.parse_args().root), ensure_ascii=False))
