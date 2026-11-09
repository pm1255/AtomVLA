"""Expand immutable real/simulation data, retaining pilot episode splits."""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_key, '1')
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation
from torch.utils.tensorboard import SummaryWriter
from prepare_data import digest, exact_video_frames, geometry_targets, motion_program, numeric_table, write_json

pa.set_cpu_count(1)


def metadata(unit):
    rows = []
    for path in sorted((unit / 'meta/episodes').glob('*/*.parquet')):
        table = pq.ParquetFile(path)
        columns = [k for k in table.schema_arrow.names if not k.startswith('stats/')]
        for batch in table.iter_batches(batch_size=512, columns=columns):
            rows.extend(batch.to_pylist())
    ids = [int(r['episode_index']) for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f'Duplicate source episode IDs: {unit}')
    return rows


def vector_labels(positions):
    displacement = np.diff(positions, axis=0, prepend=positions[:1])
    distance = np.linalg.norm(displacement, axis=-1)
    valid = distance > 0.001
    direction = np.where(valid[..., None], displacement / np.maximum(distance[..., None], 1e-8), 0)
    return displacement, direction, valid


def convert(entry):
    started = time.time()
    unit, output = Path(entry['source_root']), Path(entry['output'])
    info, row, domain = entry['info'], entry['row'], entry['domain']
    eid = int(row['episode_index']); uid = f'{domain}__{eid:06d}'
    fps = float(info['fps']); delta = round(fps * 0.5)
    path = unit / info['data_path'].format(chunk_index=row['data/chunk_index'], file_index=row['data/file_index'])
    table = numeric_table(str(path))
    data = table.filter(pc.equal(table['episode_index'], eid)).to_pydict()
    n = len(data['frame_index'])
    if n != int(row['length']) or not np.array_equal(data['frame_index'], np.arange(n)):
        raise ValueError('Episode length/frame index mismatch')
    if not np.allclose(data['timestamp'], np.arange(n) / fps, atol=0.005):
        raise ValueError('Numeric timestamp mismatch')
    poses = np.asarray(data['observation.state.eef_pose'], np.float32).reshape(n, -1, 7)
    arms = poses.shape[1]
    if arms not in (1, 2) or not np.isfinite(poses).all():
        raise ValueError('Invalid EEF poses')
    norms = np.linalg.norm(poses[..., 3:], axis=-1)
    if np.max(np.abs(norms - 1)) > 0.01:
        raise ValueError('Invalid quaternion norms')
    poses[..., 3:] /= norms[..., None]
    sj = np.asarray(data['observation.state.joint_position'], np.float32)
    sg = np.asarray(data['observation.state.gripper'], np.float32)
    aj = np.asarray(data['action.joint_position'], np.float32)
    ag = np.asarray(data['action.gripper'], np.float32)
    state, action = np.concatenate((sj, sg), -1), np.concatenate((aj, ag), -1)
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError('Nonfinite state/action')
    copied_fraction = float(np.all(sj == aj, axis=-1).mean())
    action_enabled = domain == 'droid'
    if action_enabled and copied_fraction == 1.0:
        raise ValueError('DROID action target is entirely identical to state')
    indices = np.unique(np.rint(np.linspace(0, n - delta - 1, 8)).astype(int))
    if len(indices) != 8:
        raise ValueError('Episode too short for eight distinct windows')
    cameras = ['observation.images.cam_ext_0' if domain == 'droid' else 'observation.images.cam_head_0',
               'observation.images.cam_wrist_left']
    images, image_meta = {}, {}
    for camera in cameras:
        prefix = 'videos/' + camera
        video = unit / info['video_path'].format(video_key=camera, chunk_index=row[prefix + '/chunk_index'], file_index=row[prefix + '/file_index'])
        frames = sorted(set(indices.tolist() + (indices + delta).tolist()))
        times = [float(row[prefix + '/from_timestamp']) + t / fps for t in frames]
        decoded = exact_video_frames(video, times, fps)
        for frame, seconds in zip(frames, times):
            dest = output / 'images' / f'{uid}__{camera.rsplit(".", 1)[-1]}__{frame:06d}.png'
            decoded[seconds][0].save(dest)
            images[camera, frame] = str(dest)
            image_meta[camera, frame] = {'source_video': str(video), 'requested_timestamp': seconds,
                                       'decoded_timestamp': decoded[seconds][1], 'image_sha256': digest(dest)}
    displacement, direction, direction_valid = vector_labels(poses[..., :3])
    angular = np.concatenate((np.zeros((1, arms, 3)), np.stack([
        (Rotation.from_quat(poses[:-1, a, 3:]).inv() * Rotation.from_quat(poses[1:, a, 3:])).as_rotvec()
        for a in range(arms)], axis=1)), axis=0)
    epfile = output / 'episodes' / f'{uid}.npz'
    np.savez_compressed(epfile, poses_xyzw=poses, state=state, action=action,
                        timestamp=np.asarray(data['timestamp']), original_frame_index=np.asarray(data['frame_index']),
                        displacement_m=displacement, movement_direction_parent_frame=direction,
                        movement_direction_valid=direction_valid, angular_increment_rotvec=angular)
    arm_names = ['arm'] if arms == 1 else ['left', 'right']
    segments = entry.get('instruction_segments', [])
    annotated_segments = []
    for segment in segments:
        start = max(0, int(segment['start_frame_index']))
        end = min(n - 1, int(segment['end_frame_index']) - 1)
        if end < start:
            continue
        delta_position = poses[end, :, :3] - poses[start, :, :3]
        distance = np.linalg.norm(delta_position, axis=-1)
        valid = distance > 0.001
        annotated_segments.append({
            'source_segment': segment, 'start_frame_index': start, 'end_frame_index_inclusive': end,
            'displacement_m': delta_position.tolist(),
            'movement_direction_parent_frame': np.where(valid[:, None], delta_position / np.maximum(distance[:, None], 1e-8), 0).tolist(),
            'movement_direction_valid': valid.tolist(), 'endpoint_pose_xyzw': poses[end].tolist(),
            'alignment_source': 'source_dataset_instruction_segment', 'semantic_atom_verified': False})
    source_hash = digest(path)
    episode_meta = {'episode_id': uid, 'domain': domain, 'split': entry['split'],
                    'source_root': str(unit), 'source_numeric_file': str(path), 'source_numeric_sha256': source_hash,
                    'source_episode_index': eid, 'frames': n, 'joint_action_equals_state_fraction': copied_fraction,
                    'action_supervision_available': action_enabled, 'instruction_segments_with_motion': annotated_segments,
                    'semantic_atom_verified': False, 'pilot_episode': entry['pilot_episode'],
                    'selection_slot': entry['slot'], 'seconds': time.time() - started}
    write_json(output / 'episode_metadata' / f'{uid}.json', episode_meta)
    records = []
    for t in indices:
        t = int(t); future = t + delta
        pos, rot, mask, future_indices = geometry_targets(poses, t, future)
        target = output / 'targets' / f'{uid}__{t:06d}.npz'
        np.savez_compressed(target, position_delta_m=pos, rotation_6d=rot, pose_mask=mask, future_indices=future_indices)
        delta_position = pos[:, -1]; distance = np.linalg.norm(delta_position, axis=-1)
        valid = (distance > 0.001) & mask[:, -1]
        records.append({
            'id': f'{uid}__{t:06d}', 'episode_id': uid, 'source_episode_index': eid, 'domain': domain,
            'domain_type': 'real' if domain == 'droid' else 'simulation', 'split': entry['split'],
            'source_root': str(unit), 'source_numeric_file': str(path), 'source_numeric_sha256': source_hash,
            'frame_index': t, 'future_frame_index': future, 'fps': fps, 'delta_seconds': delta / fps,
            'instruction': ' '.join(str(x) for x in row['tasks'] if str(x).strip()),
            'images': [images[c, t] for c in cameras], 'future_images': [images[c, future] for c in cameras],
            'image_provenance': [image_meta[c, t] for c in cameras],
            'future_image_provenance': [image_meta[c, future] for c in cameras],
            'flow_camera': cameras[0], 'episode_file': str(epfile), 'pose_target': str(target),
            'program': motion_program(poses, sg, future, arm_names),
            'program_label_source': 'trajectory_derived_motion_window_v2',
            'gripper_program_target_source': 'future_observed_gripper_state',
            'semantic_atom_verified': False, 'program_replay_verified': False,
            'source_instruction_segments': [s for s in segments if int(s['start_frame_index']) <= future and int(s['end_frame_index']) > t],
            'movement_direction_parent_frame': np.where(valid[:, None], delta_position / np.maximum(distance[:, None], 1e-8), 0).tolist(),
            'movement_direction_valid': valid.tolist(), 'movement_displacement_m': delta_position.tolist(),
            'action_supervision_available': action_enabled,
            'action_representation': 'native_absolute_joint_target_plus_gripper',
            'action_exclusion_reason': None if action_enabled else 'GenieSim action fields reproduce observation states; control targets not verified',
            'pose_reference': 'native parent frame retained; standardized local EEF axes; no cross-domain origin calibration'})
    write_json(output / 'episode_manifests' / f'{uid}.json', records)
    return {'episode': episode_meta, 'records': records}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--version', default='expanded_v2')
    p.add_argument('--workers', type=int, default=3)
    a = p.parse_args()
    control = a.root / 'expansions' / a.version
    output = a.root / 'data' / a.version
    if (output / 'CONVERSION_READY.json').exists():
        raise SystemExit('Conversion already complete; do not overwrite an immutable version.')
    for directory in ('images', 'episodes', 'targets', 'episode_manifests', 'episode_metadata'):
        (output / directory).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(control / 'tensorboard/data_selection'))
    conversion_writer = SummaryWriter(str(control / 'tensorboard/conversion'))
    def progress(phase, **values):
        write_json(control / 'evidence/progress.json', {'phase': phase, 'updated_at': time.time(), **values})
    pilot = {}
    for split in ('train', 'validation', 'test'):
        for line in (a.root / 'data/pilot_v1' / f'{split}.jsonl').read_text().splitlines():
            row = json.loads(line); pilot[row['episode_id']] = row
    source = Path('/opt/atomvla/dataset/lerobot_v3.0_v1')
    sim_units = sorted(p for p in (source / 'clean/geniesim3').iterdir() if (p / 'meta/info.json').exists())
    if len(sim_units) != 6:
        raise ValueError('Simulation source set changed; review selection configuration.')
    units = [('droid', source / 'droid_1_0_1/chunk-000', 500)] + [
        ('geniesim_' + unit.name, unit, 84 if i < 2 else 83) for i, unit in enumerate(sim_units)]
    rng = np.random.default_rng(20261031)
    entries, reserves, sources = [], {}, []
    for domain, unit, quota in units:
        progress('metadata_selection', domain=domain, selected=len(entries), target=1000)
        info = json.loads((unit / 'meta/info.json').read_text())
        rows = metadata(unit)
        candidates = [r for r in rows if any(str(x).strip() for x in r.get('tasks', []))
                      and int(r['length']) - round(float(info['fps']) * 0.5) >= 8]
        rng.shuffle(candidates)
        required_pilot = {int(r['source_episode_index']): r['split'] for r in pilot.values() if r['domain'] == domain}
        by_id = {int(r['episode_index']): r for r in candidates}
        if not set(required_pilot).issubset(by_id):
            raise ValueError(f'Pilot source episodes missing or no longer eligible: {domain}')
        selected = [by_id[i] for i in sorted(required_pilot)]
        remaining = [r for r in candidates if int(r['episode_index']) not in required_pilot]
        selected.extend(remaining[:quota - len(selected)])
        if len(selected) != quota:
            raise ValueError(f'Insufficient episodes: {domain}')
        reserves[domain] = remaining[quota - len(required_pilot):quota - len(required_pilot) + 100]
        counts = Counter(required_pilot.values())
        targets = {'validation': round(quota * 0.1), 'test': round(quota * 0.1)}
        annotations = {}
        annotation_path = unit / 'meta/geniesim3_annotations.jsonl'
        if annotation_path.exists():
            for line in annotation_path.read_text().splitlines():
                item = json.loads(line); annotations[int(item['episode_index'])] = item.get('instruction_segments', [])
        for row in selected:
            eid = int(row['episode_index'])
            split = required_pilot.get(eid)
            if split is None:
                split = next((s for s in ('validation', 'test') if counts[s] < targets[s]), 'train')
                counts[split] += 1
            entries.append({'domain': domain, 'source_root': str(unit), 'output': str(output), 'info': info,
                            'row': row, 'split': split, 'pilot_episode': eid in required_pilot,
                            'instruction_segments': annotations.get(eid, []), 'slot': len(entries)})
        sources.append({'domain': domain, 'root': str(unit), 'metadata_episodes_scanned': len(rows),
                        'eligible_episodes': len(candidates), 'selected_episodes': quota,
                        'info_sha256': digest(unit / 'meta/info.json'),
                        'coordinate_plan_sha256': digest(unit / 'meta/coordinate_plan.json'),
                        'annotations_sha256': digest(annotation_path) if annotation_path.exists() else None,
                        'action_supervision_available': domain == 'droid'})
        writer.add_scalar('episodes/selected', len(entries), len(entries)); writer.flush()
    plan = {'seed': 20261031, 'sources': sources, 'selection': entries, 'reserves': reserves,
            'old_episode_split_policy': 'Preserve all 112 pilot episode assignments',
            'selection_policy': 'Random eligible episodes from full metadata; balanced real/simulation and simulation tasks',
            'code_sha256': digest(__file__)}
    write_json(control / 'configs/selection_plan.json', plan)
    writer.add_scalar('stage/conversion_started', 1, 1000); writer.close()
    records, episodes, errors = [], [], []
    # Numeric shard order reduces source reads; selected splits are independent of processing order.
    entries.sort(key=lambda x: (x['domain'], x['row']['data/chunk_index'], x['row']['data/file_index'], x['row']['episode_index']))
    progress('conversion', episodes=0, planned=1000, samples=0)
    pending_entries = iter(entries)
    by_type = Counter()
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        pending = {}
        for _ in range(a.workers):
            entry = next(pending_entries, None)
            if entry: pending[pool.submit(convert, entry)] = entry
        while pending:
            done, _ = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
            for future in done:
                entry = pending.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    error = {'domain': entry['domain'], 'episode_index': entry['row']['episode_index'],
                             'split': entry['split'], 'error_type': type(exc).__name__, 'error': str(exc)}
                    errors.append(error)
                    write_json(control / 'evidence/rejected_episodes.json', errors)
                    print(json.dumps({'event': 'episode_rejected', **error}), flush=True)
                    if entry['pilot_episode'] or len(errors) >= 50 or not reserves[entry['domain']]:
                        raise RuntimeError('Bounded rejection limit or changed pilot source; stop for inspection') from exc
                    replacement = {**entry, 'row': reserves[entry['domain']].pop(0), 'pilot_episode': False, 'instruction_segments': []}
                    ap = Path(entry['source_root']) / 'meta/geniesim3_annotations.jsonl'
                    if ap.exists():
                        for line in ap.read_text().splitlines():
                            item = json.loads(line)
                            if int(item['episode_index']) == int(replacement['row']['episode_index']):
                                replacement['instruction_segments'] = item.get('instruction_segments', []); break
                    pending[pool.submit(convert, replacement)] = replacement
                    continue
                records.extend(result['records']); episodes.append(result['episode'])
                kind = 'real' if entry['domain'] == 'droid' else 'simulation'; by_type[kind] += 1
                count = len(episodes)
                progress('conversion', episodes=count, planned=1000, samples=len(records), by_type=dict(by_type))
                for tag, value in [('episodes/completed', count), ('episodes/target', 1000), ('progress/fraction', count / 1000),
                                   ('samples/completed', len(records)), ('episodes/rejected', len(errors)),
                                   ('episodes/real', by_type['real']), ('episodes/simulation', by_type['simulation'])]:
                    conversion_writer.add_scalar(tag, value, count)
                conversion_writer.flush()
                print(json.dumps({'event': 'episode_converted', 'episode_id': result['episode']['episode_id'],
                                  'episodes': count, 'samples': len(records), 'seconds': result['episode']['seconds']}), flush=True)
                next_entry = next(pending_entries, None)
                if next_entry: pending[pool.submit(convert, next_entry)] = next_entry
    records.sort(key=lambda r: r['id'])
    for split in ('train', 'validation', 'test'):
        with (output / f'{split}.jsonl').open('w') as stream:
            for row in records:
                if row['split'] == split: stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    train = [ep for ep in episodes if ep['domain'] == 'droid' and ep['split'] == 'train']
    state_values, action_values = [], []
    for ep in train:
        with np.load(output / 'episodes' / f"{ep['episode_id']}.npz") as arrays:
            state_values.append(arrays['state']); action_values.append(arrays['action'])
    states, actions = np.concatenate(state_values), np.concatenate(action_values)
    write_json(output / 'normalization.json', {'droid': {
        'state_mean': states.mean(0).tolist(), 'state_std': np.maximum(states.std(0), 0.01).tolist(),
        'action_mean': actions.mean(0).tolist(), 'action_std': np.maximum(actions.std(0), 0.01).tolist(),
        'training_episodes': len(train)}})
    write_json(output / 'provenance.json', {k: v for k, v in plan.items() if k not in ('selection', 'reserves')} | {
        'accepted_episodes': episodes, 'rejected_episodes': errors,
        'program_change_from_pilot': 'Gripper endpoint now uses future observed state, matching observed EEF endpoint',
        'direction_policy': 'Net displacements <=1mm have zero direction and valid=false; native parent coordinates retained'})
    result = {'episodes': len(episodes), 'samples': len(records), 'by_type': dict(by_type),
              'split_samples': dict(Counter(r['split'] for r in records)), 'flow_labels_ready': False,
              'semantic_atoms_verified': False, 'stage2_sources': ['droid'], 'pilot_splits_preserved': True,
              'manifest_sha256': {s: digest(output / f'{s}.jsonl') for s in ('train', 'validation', 'test')}}
    write_json(output / 'CONVERSION_READY.json', result)
    conversion_writer.add_scalar('progress/complete', 1, len(episodes)); conversion_writer.close()
    progress('conversion_complete', **result)
    print(json.dumps({'event': 'conversion_complete', **result}), flush=True)


if __name__ == '__main__':
    main()
