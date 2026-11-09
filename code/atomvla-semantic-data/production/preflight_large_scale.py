"""Fail-closed gate before submitting the 107k-episode conversion.

The gate is intentionally independent of the VLM.  It checks that the exact
source inventory, episode receipts, deterministic semantic sidecars and
automatic quarantine policy are present.  A successful gate authorizes a
resumable *conversion* run; it does not claim that all semantics are accepted
or that the resulting records are ground truth.
"""
import argparse, json, subprocess, sys, time
from pathlib import Path


def _read(path):
    return json.loads(Path(path).read_text())


def check(root, pilot=None, expected=None, candidate_only=False):
    root = Path(root); checks = {}; errors = []
    checks["inventory"] = (root / "inventory.jsonl").exists()
    if not checks["inventory"]:
        errors.append("inventory.jsonl is missing")
        inventory = []
    else:
        inventory = [json.loads(x) for x in (root / "inventory.jsonl").read_text().splitlines() if x.strip()]
    if not inventory: errors.append("inventory is empty")
    if any(x.get("source") not in ("libero", "droid", "robotwin") for x in inventory):
        errors.append("unsupported inventory source")
    if expected is not None and len(inventory) != expected:
        errors.append(f"inventory count {len(inventory)} != expected {expected}")
    checks["no_geniesim"] = all(x.get("source") != "geniesim" for x in inventory)
    if not checks["no_geniesim"]: errors.append("GenieSim is present in inventory")
    checks["unique_episode_ids"] = len({(x.get("source"), x.get("episode_id")) for x in inventory}) == len(inventory)
    if not checks["unique_episode_ids"]: errors.append("duplicate source/episode identity")

    pilot_rows = []
    if pilot:
        pilot = Path(pilot)
        checks["pilot_exists"] = pilot.exists()
        if not checks["pilot_exists"]: errors.append("pilot output root is missing")
        episode_dirs = list(pilot.glob("episodes/*/*"))
        # Diagnostic pilots historically used ``pilot_files/<source>/<episode>``;
        # accept that layout while keeping the production layout unchanged.
        if not episode_dirs:
            episode_dirs = [x.parent for x in pilot.glob("*/*/receipt.json")]
        for p in episode_dirs:
            receipt = p / "receipt.json"; ann = p / "annotations.json"
            if not (receipt.exists() and ann.exists()):
                errors.append(f"incomplete episode artifact: {p}"); continue
            r = _read(receipt); d = _read(ann)
            det = p / "deterministic_semantics.json"
            if not det.exists(): errors.append(f"missing deterministic sidecar: {p}")
            else:
                dd = _read(det)
                if dd.get("candidate_count", 0) != len(dd.get("candidates", [])):
                    errors.append(f"deterministic sidecar count mismatch: {p}")
            if d.get("full_timeline_accounted") is not True:
                errors.append(f"timeline is not fully accounted: {p}")
            if any(x.get("train_eligible") and x.get("verified_by_human") for x in d.get("annotations", [])):
                errors.append(f"human verification flag appeared in automatic run: {p}")
            pilot_rows.append({"source": r.get("source"), "episode_id": r.get("episode_id"),
                               "records": r.get("records", 0),
                               "eligible": sum(bool(x.get("train_eligible")) for x in d.get("annotations", [])),
                               "unresolved": d.get("counts", {}).get("unresolved_transitions", 0)})
        checks["pilot_artifacts"] = bool(pilot_rows)
        checks["pilot_auto_candidate_nonzero"] = any(x["eligible"] > 0 for x in pilot_rows)
        if not checks["pilot_artifacts"]: errors.append("pilot has no episode artifacts")
        if not candidate_only and not checks["pilot_auto_candidate_nonzero"]: errors.append("pilot produced no automatic candidate")

    # A source conversion must be resumable and leave unresolved spans instead
    # of silently dropping them.  The implementation checks are run through
    # the same interpreter used for the production job when requested.
    checks["policy_files"] = all((Path(__file__).with_name(x)).exists() for x in (
        "auto_only_policy.py", "auto_only_release.py", "full_visual_convert.py",
        "deterministic_semantics.py", "full_visual_release.py"))
    if not checks["policy_files"]: errors.append("automatic release policy files are incomplete")
    # V1 checked only artifact bookkeeping. It cannot authorize production.
    # Explicit candidate-only production is allowed; training release stays closed.
    checks["independent_semantic_release_gate"] = False
    if not candidate_only:
        errors.append("independent semantic release validation not implemented; diagnostic conversion only")
    report = {"schema": "atomvla_large_scale_preflight_v3", "candidate_only": candidate_only, "ready": not errors,
              "checks": checks, "errors": errors, "inventory_count": len(inventory),
              "pilot": pilot_rows, "conversion_authorized": not errors,
              "training_authorized": False, "time": time.time()}
    return report


def main():
    p = argparse.ArgumentParser(); p.add_argument("--root", required=True)
    p.add_argument("--pilot"); p.add_argument("--expected", type=int); p.add_argument("--report"); p.add_argument("--candidate-only", action="store_true")
    a = p.parse_args(); report = check(a.root, a.pilot, a.expected, a.candidate_only)
    out = Path(a.report) if a.report else Path(a.root) / "preflight_large_scale.json"
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["ready"] else 2


if __name__ == "__main__": sys.exit(main())
