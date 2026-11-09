"""Derive native gripper-center positions from recorded native EEF poses.

Robot._trans_endpose(is_endpose=True) differs from is_endpose=False by
R @ [0.12, 0, 0]. This is a local rigid offset, not a world-X translation.
The execution program remains in the original EEF convention.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from audit_results import export_csv, save
from geometry import annotate


def tcp_from_eef(p, q):
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    axis = np.stack([1 - 2*(y*y + z*z), 2*(x*y + w*z), 2*(x*z - w*y)], axis=-1)
    assert np.allclose(np.linalg.norm(axis, axis=-1), 1, atol=1e-9)
    return p + .12 * axis


def main(root):
    # Verify axis convention on two analytic rotations, including quaternion sign equivalence.
    origin = np.zeros((3, 3))
    q = np.array([[1, 0, 0, 0], [2**-.5, 0, 0, 2**-.5], [-2**-.5, 0, 0, -2**-.5]])
    assert np.allclose(tcp_from_eef(origin, q), [[.12, 0, 0], [0, .12, 0], [0, .12, 0]])
    source = root / 'provenance/robotwin/source/envs/robot/robot.py'
    info = {'position_source': 'derived from recorded EEF pose and native rigid tool offset',
            'transform': 'p_tcp_world = p_eef_world + R(q_wxyz) @ [0.12, 0, 0]',
            'native_source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'direct_tcp_sampling': False, 'execution_program_changed': False, 'episodes': []}
    rows, local = [], []
    for ep in sorted((root / 'robotwin').glob('*')):
        f = ep / 'source/trace.npz'
        if not f.exists():
            continue
        tr = np.load(f)
        p, q = tr['positions'], tr['quaternions_wxyz']
        tcp = tcp_from_eef(p, q)
        assert np.allclose(np.linalg.norm(tcp - p, axis=-1), .12, atol=1e-9)
        raw = np.load(ep / 'raw_control_replay/trace.npz')
        raw_tcp = tcp_from_eef(raw['positions'], raw['quaternions_wxyz'])
        assert np.array_equal(raw_tcp, tcp)
        segments = json.loads((ep / 'source/segments.json').read_text())
        labels = annotate(tcp, segments, frame='robotwin_native_tcp_world', arm_names=['left', 'right'],
                          dt=1/250, horizon=25, quaternions_xyzw=q[:, :, [1, 2, 3, 0]])
        labels.update(position_source=info['position_source'], transform=info['transform'])
        save(ep / 'tcp_direction_annotations.json', labels)
        for s in labels['segments']:
            d = s['net_direction']
            rows.append({'episode': ep.name, 'segment_id': s['segment_id'], 'arm': s['arm'],
                'primitive': s['primitive_by_arm'][s['arm']], 'start_action': s['start'],
                'end_action_exclusive': s['end'], 'coordinate_frame': labels['coordinate_frame'],
                'position_source': labels['position_source'],
                **{f'endpoint_{k}_m': s['endpoint_position'][j] for j, k in enumerate('xyz')},
                **{f'direction_{k}': d[j] if d is not None else None for j, k in enumerate('xyz')},
                'direction_valid': s['net_direction_valid'], 'motion_type': s['motion_type'],
                'net_distance_m': s['net_distance_m'], 'path_length_m': s['path_length_m']})
        for s in labels['local_directions']:
            d = s['direction']
            local.append({'episode': ep.name, 'segment_id': s['segment_id'], 'arm': s['arm'],
                'frame': s['frame'], 'future_frame': s['future_frame'], 'horizon_s': s['horizon_s'],
                'coordinate_frame': labels['coordinate_frame'],
                **{f'direction_{k}': d[j] if d is not None else None for j, k in enumerate('xyz')},
                'valid': s['valid']})
        info['episodes'].append({'episode': ep.name, 'arm_segment_records': len(labels['segments']),
                                 'short_window_records': len(labels['local_directions']),
                                 'raw_replay_max_tcp_position_difference_m': 0.0})
    dest = root / 'artifacts'
    export_csv(dest / 'robotwin_tcp_atomic_directions.csv', rows)
    export_csv(dest / 'robotwin_tcp_frame_directions.csv', local)
    save(dest / 'tcp_derivation.json', info)
    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    main(Path(p.parse_args().root))
