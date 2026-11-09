"""Read completed results into a separate, unambiguous final-summary run."""
import argparse
import json
from pathlib import Path
from monitor import Monitor

p = argparse.ArgumentParser()
p.add_argument('--root', required=True)
p.add_argument('--backend', choices=['libero', 'robotwin'], required=True)
a = p.parse_args()
root = Path(a.root)
records = json.loads((root / (a.backend + '_results.json')).read_text())
valid = [r for r in records if r.get('valid_reference')]
executed = sum(len(r['conditions']) for r in records)
metrics = {'evaluation/finished': 1, 'evaluation/running': 0,
           'scenes/processed': len(records), 'scenes/processed_fraction': 1,
           'scenes/valid_references': len(valid),
           'scenes/errors': sum(r['status'] == 'error' for r in records),
           'rollouts/executed': executed, 'rollouts/planned': 4 * len(records),
           'rollouts/unexecuted_due_to_error': 4 * len(records) - executed}
for mode in valid[0]['conditions']:
    metrics[mode + '/successes'] = sum(r['conditions'][mode]['final_native_success'] for r in valid)
    metrics[mode + '/valid_reference_denominator'] = len(valid)
m = Monitor(root, a.backend + '_final')
m.write(1, **metrics)
m.close()
print(json.dumps(metrics, indent=2))
