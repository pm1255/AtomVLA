"""Audit saved physical rollouts and export labels without rerunning simulation."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def read(p):
    return json.loads(p.read_text())


def save(p, x):
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False))


def export_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def audit(root):
    dest = root / 'artifacts'
    dest.mkdir(exist_ok=True)
    full = {}
    for backend, arms, raw in [('libero', ['panda'], 'raw_action_replay'),
                                ('robotwin', ['left', 'right'], 'raw_control_replay')]:
        records = read(root / (backend + '_results.json'))
        segment_rows, local_rows, checks, endpoint_rows = [], [], [], []
        valid = [r for r in records if r.get('valid_reference')]
        for r in valid:
            ep = root / backend / (r['task'] + '__seed' + str(r['seed']))
            labels = read(ep / 'direction_annotations.json')
            def trace(mode):
                return np.load(ep / (mode + '_trace.npz') if backend == 'libero'
                               else ep / mode / 'trace.npz')
            def segments(mode):
                if backend == 'robotwin':
                    return read(ep / mode / 'segments.json')
                end, result = 0, []
                for s in r['conditions'][mode]['segment_results']:
                    result.append({'start': end, 'end': end + s['steps']})
                    end += s['steps']
                return result
            source = trace('source')
            p = source['positions']
            if p.ndim == 2:
                p = p[:, None, :]
            n = len(p) - 1
            assert np.isfinite(p).all()
            q = source['quaternions_xyzw' if backend == 'libero' else 'quaternions_wxyz']
            assert np.allclose(np.linalg.norm(q, axis=-1), 1, atol=1e-3)
            if backend == 'libero':
                assert n == len(source['actions']) == len(source['success'])
            else:
                assert n == r['conditions']['source']['steps_250hz']
            rawp = trace(raw)['positions']
            assert np.array_equal(source['positions'], rawp)
            assert r['conditions'][raw]['max_measured_position_difference_m'] == 0
            program = read(ep / 'program_xyz.json')
            lookup = {}
            for arm_i, arm in enumerate(arms):
                rows = [s for s in labels['segments'] if s['arm'] == arm]
                cursor = 0
                for s in rows:
                    sid, a, b = s['segment_id'], s['start'], s['end']
                    assert a == cursor and a < b <= n
                    cursor = b
                    xyz = p[a:b+1, arm_i]
                    delta = xyz[-1] - xyz[0]
                    length = np.linalg.norm(delta)
                    assert np.allclose(xyz[0], s['start_position'], atol=1e-12, rtol=0)
                    assert np.allclose(xyz[-1], s['endpoint_position'], atol=1e-12, rtol=0)
                    assert np.allclose(delta, s['net_displacement'], atol=1e-12, rtol=0)
                    ok = length > labels['epsilon_m']
                    assert ok == s['net_direction_valid']
                    assert (s['net_direction'] is not None) == ok
                    if ok:
                        assert np.allclose(delta / length, s['net_direction'], atol=1e-12)
                    primitive = s.get('primitive', s.get('primitive_by_arm', {}).get(arm))
                    calls = [program[sid]] if backend == 'libero' else program[sid]
                    for c in calls:
                        if backend == 'robotwin' and c['arm'] != arm:
                            continue
                        assert set(c) == ({'primitive', 'target_xyz_m'} if backend == 'libero'
                                          else {'arm', 'primitive', 'target_xyz_m'})
                        assert c['primitive'] == primitive
                        assert np.array_equal(c['target_xyz_m'], xyz[-1])
                    lookup[(sid, arm)] = s
                    segment_rows.append({
                        'episode': ep.name, 'segment_id': sid, 'primitive': primitive, 'arm': arm,
                        'start_action': a, 'end_action_exclusive': b,
                        'coordinate_frame': labels['coordinate_frame'], 'dt_s': labels['dt_s'],
                        'duration_s': s['duration_s'],
                        **{f'endpoint_{k}_m': xyz[-1, j] for j, k in enumerate('xyz')},
                        **{f'direction_{k}': s['net_direction'][j] if ok else None
                           for j, k in enumerate('xyz')},
                        'direction_valid': bool(ok), 'motion_type': s['motion_type'],
                        'net_distance_m': s['net_distance_m'], 'path_length_m': s['path_length_m'],
                        'straightness': s['straightness'], 'rotation_excursion_deg': s['rotation_excursion_deg'],
                        'boundary_source': s['boundary_source'], 'semantic_verified': s['semantic_verified']})
                assert cursor == n
            seen = set()
            for d in labels['local_directions']:
                s = lookup[(d['segment_id'], d['arm'])]
                t, f = d['frame'], d['future_frame']
                assert s['start'] <= t < f <= s['end']
                assert f == min(t + labels['horizon_frames'], s['end'])
                delta = p[f, arms.index(d['arm'])] - p[t, arms.index(d['arm'])]
                length = np.linalg.norm(delta)
                assert np.allclose(delta, d['displacement_m'], atol=1e-12)
                ok = length > labels['epsilon_m']
                assert ok == d['valid'] and (d['direction'] is not None) == ok
                if ok:
                    assert np.allclose(delta / length, d['direction'], atol=1e-12)
                assert (t, d['arm']) not in seen
                seen.add((t, d['arm']))
                local_rows.append({'episode': ep.name, 'segment_id': d['segment_id'],
                    'arm': d['arm'], 'frame': t, 'future_frame': f, 'horizon_s': d['horizon_s'],
                    'coordinate_frame': labels['coordinate_frame'],
                    **{f'displacement_{k}_m': delta[j] for j, k in enumerate('xyz')},
                    **{f'direction_{k}': d['direction'][j] if ok else None for j, k in enumerate('xyz')},
                    'valid': bool(ok)})
            assert len(seen) == n * len(arms)
            srcseg = segments('source')
            for mode in ['source', raw, 'endpoint_xyz', 'endpoint_pose']:
                saved = read(ep / (mode + '_result.json') if backend == 'libero'
                             else ep / mode / 'result.json')
                assert saved == r['conditions'][mode]
                tr = trace(mode)
                if backend == 'libero':
                    assert bool(tr['success'][-1]) == saved['final_native_success']
                if mode in ['endpoint_xyz', 'endpoint_pose']:
                    ps = tr['positions']
                    if ps.ndim == 2:
                        ps = ps[:, None, :]
                    segs = segments(mode)
                    assert len(segs) == len(srcseg) and segs[-1]['end'] == len(ps) - 1
                    for sid, (s, actual) in enumerate(zip(srcseg, segs)):
                        for j, arm in enumerate(arms):
                            lab = lookup[(sid, arm)]
                            kind = lab.get('primitive', lab.get('primitive_by_arm', {}).get(arm))
                            if kind == 'hold':
                                continue
                            endpoint_rows.append({'episode': ep.name, 'mode': mode, 'segment_id': sid,
                                'arm': arm, 'primitive': kind,
                                'position_error_m': float(np.linalg.norm(ps[actual['end'], j] - p[s['end'], j]))})
            checks.append({'episode': ep.name, 'measured_states': n + 1, 'actions_or_physics_steps': n,
                           'temporal_segments': len(srcseg), 'arm_records': len(labels['segments']),
                           'local_direction_rows': len(seen), 'raw_replay_max_difference_m': 0.0,
                           'serialized_xyz_exactly_measured_endpoint': True,
                           'schema_has_no_orientation_timing_or_path': True,
                           'no_direction_window_crosses_boundary': True})
        aggregate = {'attempted_reference_scenes': len(records), 'valid_reference_scenes': len(valid),
            'environment_or_execution_errors': [dict(seed=r['seed'], error=r.get('error'))
                                               for r in records if r['status'] == 'error'],
            'temporal_segments': sum(c['temporal_segments'] for c in checks),
            'arm_segment_records': len(segment_rows), 'local_direction_records': len(local_rows),
            'conditions': {m: {'successes': sum(r['conditions'][m]['final_native_success'] for r in valid),
                               'denominator_valid_references': len(valid)}
                           for m in ['source', raw, 'endpoint_xyz', 'endpoint_pose']}}
        for mode in ['endpoint_xyz', 'endpoint_pose']:
            vals = [x['position_error_m'] for x in endpoint_rows if x['mode'] == mode]
            aggregate['conditions'][mode]['mean_active_segment_endpoint_error_m'] = float(np.mean(vals))
        export_csv(dest / (backend + '_atomic_directions.csv'), segment_rows)
        export_csv(dest / (backend + '_frame_directions.csv'), local_rows)
        export_csv(dest / (backend + '_endpoint_errors.csv'), endpoint_rows)
        save(dest / (backend + '_aggregate.json'), aggregate)
        full[backend] = {'checks_passed': True, 'episodes': checks, 'aggregate': aggregate}
    full['scope_limits'] = [
        'Only the saved pilot rollouts are covered, not public demonstration corpora.',
        'Native RoboTwin success is recorded from check_success; this audit does not rerun physics.',
        'Initial state comparison was enforced during execution; only one initial scene artifact was saved.',
        'Boundaries are instrumented calls, not independently human-verified semantic alignments.']
    save(dest / 'validation.json', full)
    manifest = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(root.rglob('*')) if p.is_file() and 'tensorboard' not in p.parts
                and 'artifacts' not in p.parts and 'scratch' not in p.parts}
    save(dest / 'raw_artifact_sha256.json', manifest)
    print(json.dumps({b: full[b]['aggregate'] for b in ['libero', 'robotwin']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    audit(Path(p.parse_args().root))
