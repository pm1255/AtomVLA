import unittest

from deterministic_semantics import candidates, canonicalize_prediction


class DeterministicSemantics(unittest.TestCase):
    def evidence(self):
        return {
            "physical_arms": ["left"],
            "events": [
                {"arm": "left", "kind": "articulation_change", "start": 10, "end": 20,
                 "evidence": {"joint": "flat_stove_1_button", "body": "flat_stove_1_button"}},
                {"arm": "left", "kind": "gripper_closing", "start": 30, "end": 34,
                 "evidence": {"delta": -.04}},
                {"arm": "left", "kind": "object_motion", "start": 34, "end": 50,
                 "evidence": {"object": "moka_pot_1_joint0", "displacement_xyz": [0, 0, .04]}},
                {"arm": "left", "kind": "object_motion", "start": 54, "end": 58,
                 "evidence": {"object": "moka_pot_1_joint0", "displacement_xyz": [0, 0, .001]}},
                {"arm": "left", "kind": "gripper_opening", "start": 55, "end": 60,
                 "evidence": {"delta": .05}},
            ],
        }

    def test_libero_contract_emits_toggle_pick_place(self):
        scene = {
            "objects": {"moka_pot_1_joint0": {}},
            "articulations": {"flat_stove_1_button": {}},
        }
        atoms, issues = candidates(None, None, self.evidence(), scene,
                                   "turn on the stove and put the moka pot on it", "libero")
        self.assertEqual([a["verb"] for a in atoms], ["toggle", "pick", "place"])
        self.assertTrue(all(not a["train_eligible"] for a in atoms))
        self.assertFalse(any("unknown" in str(a) for a in atoms))

    def test_ambiguous_without_object_state_abstains(self):
        ev = self.evidence()
        ev["events"] = [x for x in ev["events"] if x["kind"] not in ("object_motion", "articulation_change")]
        atoms, _ = candidates(None, None, ev, {}, "pick up the bottle and place it on the table", "droid")
        self.assertEqual(atoms, [])

    def test_visual_rows_are_replaced_by_measured_rows(self):
        raw = {"verb": "pick", "object": "bottle", "target": "table",
               "events": {"closed": "F1", "lifted": "F2"}, "evidence": "The bottle is picked."}
        out = canonicalize_prediction(raw, {"arm": "left", "start": 0, "end": 60},
                                      {"F1": 30, "F2": 60}, self.evidence(),
                                      "pick up the moka pot and place it on the stove",
                                      {"objects": {"moka_pot_1_joint0": {}}, "articulations": {}}, "libero")
        self.assertEqual(out["object"], "moka_pot_1_joint0")
        self.assertEqual(out["events"], {"closed": "F1", "lifted": "F2"})


if __name__ == "__main__":
    unittest.main()
