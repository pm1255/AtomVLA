import json, tempfile, unittest
from pathlib import Path
from preflight_large_scale import check


class Preflight(unittest.TestCase):
    def test_missing_inventory_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            r = check(d, expected=1)
            self.assertFalse(r["ready"])
            self.assertIn("inventory.jsonl is missing", r["errors"])

    def test_bookkeeping_fixture_cannot_authorize_conversion(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root / "inventory.jsonl").write_text('{"source":"libero","episode_id":"e"}\n')
            pilot = root / "pilot" / "episodes" / "libero" / "e"; pilot.mkdir(parents=True)
            (pilot / "receipt.json").write_text(json.dumps({"source":"libero", "episode_id":"e", "records":1}))
            (pilot / "deterministic_semantics.json").write_text(json.dumps({"candidate_count":1,"candidates":[{}]}))
            (pilot / "annotations.json").write_text(json.dumps({"full_timeline_accounted":True,"counts":{"unresolved_transitions":0},"annotations":[{"train_eligible":True,"verified_by_human":False}]}))
            r = check(root, pilot / ".." / ".." / ".." / ".." / "pilot", expected=1)
            self.assertFalse(r["ready"], r)
            self.assertFalse(r["checks"]["independent_semantic_release_gate"])
            self.assertFalse(r["training_authorized"])


if __name__ == "__main__": unittest.main()
