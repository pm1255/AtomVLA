"""Measured EEF endpoint and translation labels; no simulator dependencies.

Segments index actions [start, end), and consume measured states [start, end].
There must be N+1 states for N actions. Translation is not the tool pointing axis.
"""
import numpy as np


def validate_segments(segments, n_actions):
    cursor = 0
    for s in segments:
        a, b = s['start'], s['end']
        if not isinstance(a, int) or not isinstance(b, int) or a != cursor or b <= a or b > n_actions:
            raise ValueError('Segments must partition [0, N) without gaps/overlap')
        cursor = b
    if cursor != n_actions:
        raise ValueError('Incomplete episode coverage')


def direction(delta, epsilon=0.002):
    norm = float(np.linalg.norm(delta))
    return (np.asarray(delta) / norm).tolist() if norm > epsilon else None


def annotate(positions, segments, *, frame, arm_names, horizon=4, epsilon=0.002,
             dt=0.05, quaternions_xyzw=None):
    p = np.asarray(positions, dtype=float)
    if p.ndim == 2:
        p = p[:, None, :]
    if p.ndim != 3 or p.shape[2] != 3 or not np.isfinite(p).all():
        raise ValueError('Expected finite measured positions [N+1, arms, 3] in metres')
    if not frame or len(arm_names) != p.shape[1] or horizon < 1 or dt <= 0:
        raise ValueError('Declare coordinate frame, arm names, positive horizon and dt')
    validate_segments(segments, len(p) - 1)
    q = None if quaternions_xyzw is None else np.asarray(quaternions_xyzw, dtype=float)
    if q is not None and (q.shape != (*p.shape[:2], 4) or not np.isfinite(q).all()
                          or not np.allclose(np.linalg.norm(q, axis=-1), 1, atol=1e-3)):
        raise ValueError('Expected normalized finite xyzw quaternions [N+1, arms, 4]')
    rows, local = [], []
    for sid, s in enumerate(segments):
        a, b = s['start'], s['end']
        for j, arm in enumerate(arm_names):
            xyz = p[a:b+1, j]
            net = xyz[-1] - xyz[0]
            distance = float(np.linalg.norm(net))
            length = float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())
            excursion = float(np.linalg.norm(xyz - xyz[0], axis=1).max())
            alpha = np.linspace(0, 1, len(xyz))[:, None]
            linear = xyz[0] + alpha * net
            rot_deg = None
            if q is not None:
                dots = np.abs(np.sum(q[a:b+1, j] * q[a, j], axis=1)).clip(0, 1)
                rot_deg = float(np.degrees(2 * np.arccos(dots)).max())
            motion = ('stationary_translation' if excursion <= epsilon else
                      'returning_motion' if distance <= epsilon else
                      'translation')
            if motion == 'stationary_translation' and rot_deg is not None and rot_deg > 3:
                motion = 'rotation_without_translation'
            rows.append({**s, 'segment_id': sid, 'arm': arm,
                         'coordinate_frame': frame, 'position_unit': 'm',
                         'start_position': xyz[0].tolist(), 'endpoint_position': xyz[-1].tolist(),
                         'net_displacement': net.tolist(), 'net_direction': direction(net, epsilon),
                         'net_direction_valid': distance > epsilon,
                         'motion_type': motion, 'path_length_m': length,
                         'net_distance_m': distance, 'max_excursion_m': excursion,
                         'straightness': distance / length if length > epsilon else None,
                         'timed_linear_max_error_m': float(np.linalg.norm(xyz-linear, axis=1).max()),
                         'rotation_excursion_deg': rot_deg,
                         'duration_s': (b-a)*dt})
            for t in range(a, b):
                future = min(t + horizon, b)
                delta = p[future, j] - p[t, j]
                d = direction(delta, epsilon)
                local.append({'segment_id': sid, 'arm': arm, 'frame': t,
                              'future_frame': future, 'horizon_s': (future-t)*dt,
                              'displacement_m': delta.tolist(), 'direction': d,
                              'valid': d is not None})
    return {'schema_version': 1, 'coordinate_frame': frame,
            'direction_meaning': 'measured translational motion, NOT wrist/tool orientation',
            'boundary_convention': 'actions [start,end), measured states [start,end]',
            'epsilon_m': epsilon, 'horizon_frames': horizon, 'dt_s': dt,
            'segments': rows, 'local_directions': local}


def heuristic_segments(positions, gripper_actions):
    """Diagnostic only: command changes and sustained vertical-motion changes.

    These are NOT human-verified semantic atom annotations. No duration cap is
    used: arbitrarily shrinking segments would make endpoint reconstruction easy.
    """
    p = np.asarray(positions)
    g = np.asarray(gripper_actions)
    if p.shape != (len(g)+1, 3):
        raise ValueError('Expected N+1 measured positions')
    closed = g > 0
    dz = np.convolve(np.diff(p[:, 2]), np.ones(5)/5, mode='same')
    vertical = np.where(dz > 0.001, 1, np.where(dz < -0.001, -1, 0))
    cuts = {0, len(g), *(np.flatnonzero(closed[1:] != closed[:-1])+1).tolist()}
    for i in range(3, len(g)-3):
        if vertical[i] != vertical[i-1] and np.all(vertical[i:i+3] == vertical[i]):
            cuts.add(i)
    result = []
    previous_closed = False
    for a, b in zip(sorted(cuts)[:-1], sorted(cuts)[1:]):
        c = bool(closed[a])
        delta = p[b] - p[a]
        if c != previous_closed:
            kind = 'grasp' if c else 'release'
        elif c:
            kind = 'lift' if delta[2] > 0.015 else 'place' if delta[2] < -0.015 else 'transport'
        else:
            kind = 'approach'
        result.append({'start': int(a), 'end': int(b), 'primitive': kind,
                       'boundary_source': 'heuristic_gripper_and_vertical_change',
                       'semantic_verified': False})
        previous_closed = c
    return result


def endpoint_program(positions, segments):
    """Strict execution payload: verb and XYZ only. Labels stay in a sidecar."""
    validate_segments(segments, len(positions)-1)
    return [{'primitive': s['primitive'], 'target_xyz_m': np.asarray(positions)[s['end']].tolist()}
            for s in segments]
