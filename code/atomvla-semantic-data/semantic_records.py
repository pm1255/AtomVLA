"""Versioned reference converter from reviewed semantic annotations, not a raw-video annotator.

State row c is the completion state; action row j moves state j to j+1.
All input geometry must already be in the calibrated robot base frame, metres,
quaternion xyzw. Source adapters must establish these conventions explicitly.
"""
from dataclasses import dataclass
import json
import math
import re
import numpy as np

VERSION = 'semantic_events_v1'

class ReviewRequired(ValueError):
    """Insufficient semantic evidence. Route to review, never fabricate a target."""

@dataclass(frozen=True)
class Schema:
    roles: tuple[str, ...]
    completion: str

# Starter schemas, not a claim of implementing the paper's full vocabulary.
SCHEMAS = {
    'pick': Schema(('pregrasp', 'closed', 'lifted'), 'held'),
    'move': Schema(('arrived',), 'arrived'),
    'place': Schema(('preplace', 'supported', 'released'), 'released_supported'),
}
_KEYS = ('object', 'target', 'arm', 'completion')
_SYMBOL = re.compile(r'^[A-Za-z_][A-Za-z0-9_./:-]*$')

def require(condition, message):
    if not condition:
        raise ReviewRequired(message)

def evidence(item, label):
    require(isinstance(item, dict) and item.get('verified') is True
            and isinstance(item.get('source'), str) and item['source'].strip(),
            label + ': missing verified evidence and provenance')

def discrete_call(verb, arguments):
    require(verb in SCHEMAS, 'unsupported primitive: define and review its schema first')
    require(set(arguments) == set(_KEYS), 'unexpected/missing text argument slots')
    for key, value in arguments.items():
        require(isinstance(value, str) and bool(_SYMBOL.fullmatch(value)),
                key + ': expected symbolic string, not numeric/vector parameters')
    require(arguments['completion'] == SCHEMAS[verb].completion,
            'completion contract inconsistent with primitive')
    return verb + '(' + ', '.join(k + '=' + json.dumps(arguments[k]) for k in _KEYS) + ')'

def make_record(episode, annotation, current_row, horizon=8):
    """Return one record; annotations hold future labels, inputs never do.

    episode: id/split/instruction, timestamp[N], poses[N,A,7], state[N,D],
      actions[N-1,D], action_valid[N-1], arm_names[A], cameras{name: list[N]},
      calibration evidence. Each camera slot: {ref, timestamp} or None.
    annotation: start/end (state rows, inclusive), verb/arguments,
      semantics/binding/completion evidence, and ordered events with
      role/row/verified/source. Review status is accepted, not independently proven.
    """
    require(type(horizon) is int and horizon > 0, 'invalid action horizon')
    require(type(current_row) is int, 'current row must be integer')
    require(episode.get('split') in ('train', 'validation', 'test'), 'episode split required')
    evidence(episode.get('calibration'), 'base-frame calibration')
    require(episode.get('pose_convention') == 'base_m_quat_xyzw', 'unknown pose convention')
    require(episode.get('action_convention') == 'state_j_to_j_plus_1', 'unknown action alignment')
    stamp = np.asarray(episode['timestamp'], dtype=float)
    pose = np.asarray(episode['poses'], dtype=float)
    state = np.asarray(episode['state'], dtype=float)
    acts = np.asarray(episode['actions'], dtype=float)
    av = np.asarray(episode['action_valid'])
    names = episode['arm_names']
    n = len(stamp)
    require(n >= 2 and stamp.shape == (n,) and np.isfinite(stamp).all()
            and (np.diff(stamp) > 0).all(), 'nonmonotonic/nonfinite state timestamps')
    require(pose.shape == (n, len(names), 7) and len(names) in (1, 2)
            and len(set(names)) == len(names), 'invalid arm/pose dimensions')
    require(np.isfinite(pose).all() and np.max(np.abs(np.linalg.norm(pose[...,3:],axis=-1)-1)) < .015,
            'invalid pose or nonunit quaternion')
    require(state.ndim == 2 and len(state) == n and np.isfinite(state).all(), 'invalid state')
    require(acts.ndim == 2 and len(acts) == n-1 and np.isfinite(acts).all(), 'invalid actions')
    require(av.shape == (n-1,) and av.dtype == np.bool_, 'explicit boolean action validity required')
    start, end = annotation['start'], annotation['end']
    require(type(start) is int and type(end) is int and 0 <= start <= current_row < end < n,
            'current/completion rows outside primitive or episode')
    for name in ('semantics', 'binding', 'completion'):
        evidence(annotation.get(name), name)
    verb, arguments = annotation['verb'], annotation['arguments']
    program = discrete_call(verb, arguments)
    active = list(names) if arguments['arm'] == 'both' else [arguments['arm']]
    require(all(a in names for a in active) and (arguments['arm'] != 'both' or len(names)==2),
            'call arm not available')
    events = annotation['events']
    require(tuple(e['role'] for e in events) == SCHEMAS[verb].roles,
            'missing/reordered event roles; do not substitute uniform frames')
    prev = start
    for event in events:
        evidence(event, 'event ' + event['role'])
        row = event['row']
        require(type(row) is int and prev < row <= end, 'invalid/duplicate event timestamp')
        prev = row
    require(events[-1]['row'] == end, 'final event must match verified completion')
    cameras = episode['cameras']
    require(bool(cameras), 'no cameras')
    tolerance = episode.get('camera_tolerance_seconds')
    require(isinstance(tolerance, (int,float)) and math.isfinite(tolerance) and tolerance >= 0,
            'explicit finite camera tolerance required')
    for slots in cameras.values():
        require(len(slots) == n, 'camera length differs from state timeline')
    def images_at(row, current=False):
        result = {}
        for camera, slots in cameras.items():
            item = slots[row]
            if item is None:
                result[camera] = {'ref': None, 'valid': False}
                continue
            ts = item.get('timestamp')
            require(isinstance(ts, (int,float)) and math.isfinite(ts)
                    and abs(ts-stamp[row]) <= tolerance, 'unsynchronized camera frame')
            require(isinstance(item.get('ref'), str) and bool(item['ref']), 'missing image reference')
            require((not current) or ts <= stamp[current_row], 'future frame leaked into current input')
            result[camera] = {'ref': item['ref'], 'timestamp': ts, 'valid': True}
        require(any(x['valid'] for x in result.values()), 'all cameras missing for required event')
        return result
    remaining = [e for e in events if e['row'] > current_row]
    keyframes = [{'role':e['role'], 'row':e['row'], 'timestamp':float(stamp[e['row']]),
                  'images':images_at(e['row']), 'evidence':e['source']} for e in remaining]
    chunk = np.zeros((horizon, acts.shape[1]), dtype=float)
    mask = np.zeros(horizon, dtype=bool)
    rows = []
    for k in range(horizon):
        j = current_row+k
        ok = j < end and j < len(acts) and av[j]
        if ok:
            chunk[k] = acts[j]
            mask[k] = True
        rows.append(j if ok else None)
    arm_mask = np.array([a in active for a in names], dtype=bool)
    terminal = pose[end].copy()
    terminal[:,3:] /= np.linalg.norm(terminal[:,3:],axis=-1,keepdims=True)
    terminal[~arm_mask] = 0
    # Frame roles are targets here. Runtime phase inference must use current/history only.
    # Do not promote remaining GT roles/timestamps into model conditioning silently.
    return {
        'version':VERSION, 'episode_id':episode['id'], 'split':episode['split'],
        'inputs':{'images':images_at(current_row, current=True), 'state':state[current_row].tolist(),
                  'instruction':episode['instruction']},
        'targets':{'program_text':program, 'terminal_pose_xyzw':terminal.tolist(),
                   'arm_mask':arm_mask.tolist(), 'keyframes':keyframes,
                   'actions':chunk.tolist(), 'action_mask':mask.tolist()},
        'metadata':{'current_row':current_row, 'start':start, 'completion_row':end,
                    'keyframe_count':len(keyframes), 'action_rows':rows,
                    'annotation':annotation, 'calibration':episode['calibration'],
                    'pose_convention':episode['pose_convention'],
                    'role_schedule_status':'label_only_requires_causal_runtime_phase_estimator'}
    }

def pack_keyframes(records):
    """Pad references/role labels for a loader. Padding is never image supervision."""
    require(bool(records), 'empty batch')
    require(all(r.get('version') == VERSION for r in records), 'legacy record version rejected')
    width = max(len(r['targets']['keyframes']) for r in records)
    slots, masks = [], []
    for r in records:
        frames = r['targets']['keyframes']
        slots.append(frames + [None]*(width-len(frames)))
        masks.append([True]*len(frames)+[False]*(width-len(frames)))
    return {'keyframes':slots, 'temporal_mask':masks}
