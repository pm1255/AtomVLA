"""Missing labels are explicit loss masks, never negative labels or ground truth."""
import copy
from visual_annotation import validate_prediction


def quarantine(doc, n, arms):
    original = copy.deepcopy(doc)
    segments = []
    removed = []
    for raw in original['segments']:
        a = copy.deepcopy(raw)
        if a['kind'] == 'primitive':
            args = a['arguments']
            if a['verb'] == 'place' and args['object'] == args['target']:
                removed.append({'reason': 'self_target', 'prediction': a})
                continue
            a['train_eligible'] = False
            a['auto_decision'] = {'status': 'auto_unresolved', 'reason': 'candidate_requires_validation'}
            a['auto_policy'] = 'candidate_only_missing_supervision_v1'
        segments.append(a)
    rows = sorted({x for a in segments for x in [a['start'], a['end']] + [e['row'] for e in a.get('events', [])]})
    out = validate_prediction({'segments': segments}, n, arms, rows, original.get('evidence', {}))
    for key in ('review_queue', 'local_attempts', 'prediction_error'):
        if key in original:
            out[key] = original[key]
    out['rejected'] = original.get('rejected', []) + removed + out['rejected']
    out['supervision_policy'] = {
        'missing': 'mask_loss_keep_source_interval', 'training_authorized': False,
        'missingness_is_random': False, 'candidate_only': True,
    }
    out['missing_supervision_intervals'] = [
        {'arm': a['arm'], 'start': a['start'], 'end': a['end'],
         'loss_mask': {'program': False, 'terminal_pose': False, 'future_rgb': False, 'action': False}}
        for a in out['segments'] if a['kind'] == 'unresolved'
    ]
    return out
