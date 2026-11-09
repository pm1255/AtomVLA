import tempfile
import unittest
from pathlib import Path
from candidate_policy import quarantine
from preflight_large_scale import check


class CandidatePolicy(unittest.TestCase):
    def test_bad_label_becomes_masked_gap(self):
        atom = {'kind': 'primitive', 'verb': 'place', 'start': 0, 'end': 8,
                'arguments': {'object': 'cup', 'target': 'cup', 'arm': 'left', 'completion': 'released'},
                'events': [{'role': 'supported', 'row': 5}, {'role': 'released', 'row': 8}],
                'observation': 'An erroneous self-target candidate.', 'train_eligible': True}
        out = quarantine({'segments': [atom]}, 10, ['left'])
        self.assertEqual(out['annotations'], [])
        self.assertEqual(out['counts']['unresolved_transitions'], 9)
        self.assertFalse(any(out['missing_supervision_intervals'][0]['loss_mask'].values()))
        self.assertEqual(out['rejected'][0]['reason'], 'self_target')

    def test_candidate_mode_does_not_open_training_gate(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, 'inventory.jsonl').write_text('{"source":"libero","episode_id":"e"}\n')
            out = check(d, candidate_only=True)
            self.assertTrue(out['conversion_authorized'])
            self.assertFalse(out['training_authorized'])
            self.assertFalse(check(d)['conversion_authorized'])

    def test_candidate_mode_still_rejects_duplicate_inventory(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, 'inventory.jsonl').write_text('{"source":"libero","episode_id":"e"}\n' * 2)
            self.assertFalse(check(d, candidate_only=True)['conversion_authorized'])


if __name__ == '__main__':
    unittest.main()
