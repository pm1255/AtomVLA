"""Build a traceable pilot from actual LeRobot v3 episodes; never edit sources.

Programs are explicitly marked trajectory-derived motion-window labels. They
are not human-verified semantic atoms or proof of successful program execution.
"""
import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time

import av
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
from PIL import Image
from scipy.spatial.transform import Rotation
from torch.utils.tensorboard import SummaryWriter


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temp.replace(path)


@lru_cache(maxsize=256)
def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def episode_rows(unit, maximum):
    for path in sorted((unit / 'meta/episodes').glob('*/*.parquet')):
        f = pq.ParquetFile(path)
        cols = [k for k in f.schema_arrow.names if not k.startswith('stats/')]
        count = 0
        for batch in f.iter_batches(batch_size=256, columns=cols):
            for row in batch.to_pylist():
                yield row
                count += 1
                if count >= maximum:
                    return


@lru_cache(maxsize=2)
def numeric_table(path):
    f = pq.ParquetFile(path)
    allowed = ['episode_index', 'frame_index', 'timestamp', 'task_index',
               'observation.state.eef_pose', 'observation.state.joint_position',
               'observation.state.gripper', 'action.joint_position', 'action.gripper',
               'language_instruction', 'language_instruction_2', 'language_instruction_3']
    return f.read(columns=[k for k in allowed if k in f.schema_arrow.names])


def exact_video_frames(path, timestamps, fps):
    """One sequential decode per episode/view, checked against decoded PTS."""
    timestamps = sorted(set(float(t) for t in timestamps))
    output = {}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'
        stream.codec_context.thread_count = 2
        container.seek(max(0, int(timestamps[0] / stream.time_base)), stream=stream, backward=True)
        wanted = iter(timestamps)
        target = next(wanted)
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            actual = float(frame.pts * stream.time_base)
            if actual < target - 0.45 / fps:
                continue
            if abs(actual - target) > 0.55 / fps:
                raise ValueError(f'video PTS mismatch: requested={target}, actual={actual}, path={path}')
            image = frame.to_image().convert('RGB').resize((224, 224), Image.Resampling.BILINEAR)
            output[target] = (image, actual)
            target = next(wanted, None)
            if target is None:
                break
    if len(output) != len(timestamps):
        raise ValueError(f'missing video frames: {len(output)}/{len(timestamps)}')
    return output


def geometry_targets(poses, t, future, steps=8):
    arms = poses.shape[1]
    indices = np.rint(np.linspace(t + 1, future, steps)).astype(int)
    position = np.zeros((2, steps, 3), np.float32)
    rotation = np.zeros((2, steps, 6), np.float32)
    mask = np.zeros((2, steps), bool)
    for arm in range(arms):
        now = poses[t, arm]
        sequence = poses[indices, arm]
        position[arm] = sequence[:, :3] - now[:3]
        relative = (Rotation.from_quat(now[3:]).inv() * Rotation.from_quat(sequence[:, 3:])).as_matrix()
        # First column then second column; do not mix this with interleaved encodings.
        rotation[arm] = relative[:, :, :2].transpose(0, 2, 1).reshape(-1, 6)
        mask[arm] = True
    return position, rotation, mask, indices


def motion_program(poses, gripper, future, arm_names):
    calls = []
    for i, arm in enumerate(arm_names):
        p = poses[future, i]
        calls.append('move_eef(arm=' + json.dumps(arm) + ', xyz=' +
                     json.dumps([round(float(x),4) for x in p[:3]]) + ', quat_xyzw=' +
                     json.dumps([round(float(x),5) for x in p[3:]]) + ')')
        calls.append('set_gripper(arm=' + json.dumps(arm) + ', target=' +
                     str(round(float(gripper[future, i]), 5)) + ')')
    return '\n'.join(calls)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--real-episodes', type=int, default=64)
    p.add_argument('--sim-per-task', type=int, default=8)
    p.add_argument('--windows', type=int, default=8)
    p.add_argument('--seed', type=int, default=20261031)
    p.add_argument('--version', default='pilot_v1')
    args = p.parse_args()
    output = args.root / 'data' / args.version
    if (output / 'READY.json').exists():
        raise SystemExit('Existing immutable pilot is ready; choose a new version to rebuild.')
    for d in ['images', 'episodes', 'targets', 'episode_manifests']:
        (output / d).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(args.root / 'tensorboard/data_conversion'))
    units = [('droid', Path('/opt/atomvla/dataset/lerobot_v3.0_v1/droid_1_0_1/chunk-000'), args.real_episodes)]
    sim_root = Path('/opt/atomvla/dataset/lerobot_v3.0_v1/clean/geniesim3')
    units += [('geniesim_' + u.name, u, args.sim_per_task) for u in sorted(sim_root.iterdir()) if (u/'meta/info.json').exists()]
    rng = np.random.default_rng(args.seed)
    records, errors, provenance = [], [], []
    completed = 0
    planned = sum(n for _, _, n in units)
    for domain, unit, required in units:
        info = json.loads((unit / 'meta/info.json').read_text())
        fps = float(info['fps'])
        delta = max(1, round(fps * 0.5))
        keys = list(info['features'])
        camera = 'observation.images.cam_ext_0' if domain == 'droid' else 'observation.images.cam_head_0'
        wrist = 'observation.images.cam_wrist_left'
        cameras = [camera] + ([wrist] if wrist in keys else [])
        candidates = list(episode_rows(unit, 2048 if domain == 'droid' else 4096))
        candidates = [r for r in candidates if any(str(x).strip() for x in r.get('tasks', [])) and r['length'] > delta + 2]
        rng.shuffle(candidates)
        annotation = {}
        ap = unit / 'meta/geniesim3_annotations.jsonl'
        if ap.exists():
            for line in ap.read_text().splitlines():
                row = json.loads(line); annotation[row['episode_index']] = row.get('instruction_segments', [])
        provenance.append({'domain':domain, 'root':str(unit), 'resolved_root':str(unit.resolve()),
                           'info_sha256':digest(unit/'meta/info.json'), 'fps':fps,
                           'coordinate_plan_sha256':digest(unit/'meta/coordinate_plan.json'),
                           'episode_selection':'seeded pilot, DROID first 2048 metadata episodes; not a full-dataset sample',
                           'action_supervision_enabled':domain == 'droid'})
        accepted = 0
        for row in candidates:
            if accepted >= required:
                break
            eid = int(row['episode_index'])
            uid = f'{domain}__{eid:06d}'
            try:
                cached = output/'episode_manifests'/f'{uid}.json'
                if cached.exists():
                    records.extend(json.loads(cached.read_text()))
                    accepted += 1; completed += 1
                    continue
                episode_records = []
                path = unit / info['data_path'].format(chunk_index=row['data/chunk_index'], file_index=row['data/file_index'])
                table = numeric_table(str(path))
                data = table.filter(pc.equal(table['episode_index'], eid)).to_pydict()
                n = len(data['frame_index'])
                if n != int(row['length']) or not np.array_equal(data['frame_index'], np.arange(n)):
                    raise ValueError('episode length/frame index mismatch')
                if not np.allclose(data['timestamp'], np.arange(n) / fps, atol=0.005):
                    raise ValueError('numeric timestamp mismatch')
                poses = np.asarray(data['observation.state.eef_pose'], np.float32).reshape(n, -1, 7)
                if poses.shape[1] not in [1,2] or not np.isfinite(poses).all():
                    raise ValueError('invalid EEF poses')
                norms = np.linalg.norm(poses[..., 3:], axis=-1)
                if np.max(np.abs(norms - 1)) > 0.01:
                    raise ValueError('invalid source quaternion norms')
                poses[..., 3:] /= norms[..., None]
                state_joint = np.asarray(data['observation.state.joint_position'], np.float32)
                state_grip = np.asarray(data['observation.state.gripper'], np.float32)
                action_joint = np.asarray(data['action.joint_position'], np.float32)
                action_grip = np.asarray(data['action.gripper'], np.float32)
                raw_state = np.concatenate([state_joint, state_grip], -1)
                raw_action = np.concatenate([action_joint, action_grip], -1)
                if not np.isfinite(raw_state).all() or not np.isfinite(raw_action).all():
                    raise ValueError('nonfinite state/action')
                indices = np.unique(np.rint(np.linspace(0, n - delta - 1, args.windows)).astype(int))
                images = {}
                image_meta = {}
                for cam in cameras:
                    prefix = 'videos/' + cam
                    vp = unit / info['video_path'].format(video_key=cam, chunk_index=row[prefix+'/chunk_index'], file_index=row[prefix+'/file_index'])
                    start = float(row[prefix+'/from_timestamp'])
                    wanted = sorted(set(indices.tolist() + (indices + delta).tolist()))
                    times = [start + t / fps for t in wanted]
                    decoded = exact_video_frames(vp, times, fps)
                    for frame, seconds in zip(wanted, times):
                        target = output/'images'/f'{uid}__{cam.rsplit(".",1)[-1]}__{frame:06d}.png'
                        decoded[seconds][0].save(target)
                        images[(cam,frame)] = str(target)
                        image_meta[(cam,frame)] = {'source_video':str(vp), 'requested_timestamp':seconds, 'decoded_timestamp':decoded[seconds][1]}
                epfile = output/'episodes'/f'{uid}.npz'
                np.savez_compressed(epfile, poses_xyzw=poses, state=raw_state, action=raw_action,
                                    timestamp=np.asarray(data['timestamp']), original_frame_index=np.asarray(data['frame_index']),
                                    displacement_m=np.diff(poses[..., :3], axis=0, prepend=poses[:1,...,:3]),
                                    angular_increment_rotvec=np.concatenate([np.zeros((1,poses.shape[1],3)),
                                      np.stack([(Rotation.from_quat(poses[:-1,a,3:]).inv()*Rotation.from_quat(poses[1:,a,3:])).as_rotvec() for a in range(poses.shape[1])],axis=1)],axis=0))
                # Split whole episodes. Keep at least one held-out episode per sim task.
                split = 'validation' if accepted == 0 else 'test' if accepted == 1 else 'train'
                if domain == 'droid':
                    heldout = max(1, round(required / 8))
                    split = 'validation' if accepted < heldout else 'test' if accepted < 2 * heldout else 'train'
                arm_names = ['arm'] if poses.shape[1] == 1 else ['left','right']
                for t in indices:
                    t = int(t);future = t + delta
                    pos, rot, mask, future_indices = geometry_targets(poses,t,future)
                    target_file = output/'targets'/f'{uid}__{t:06d}.npz'
                    np.savez_compressed(target_file, position_delta_m=pos, rotation_6d=rot, pose_mask=mask, future_indices=future_indices)
                    segments = [s for s in annotation.get(eid,[]) if int(s['start_frame_index']) <= t < int(s['end_frame_index'])]
                    displacement = pos[:, -1]
                    distance = np.linalg.norm(displacement, axis=-1)
                    direction = displacement / np.maximum(distance[:,None], 1e-8)
                    episode_records.append({'id':f'{uid}__{t:06d}', 'episode_id':uid, 'source_episode_index':eid,
                                    'domain':domain, 'split':split, 'source_root':str(unit), 'source_numeric_file':str(path),
                                    'source_numeric_sha256':digest(path),
                                    'frame_index':t, 'future_frame_index':future, 'fps':fps, 'delta_seconds':delta/fps,
                                    'instruction':' '.join(str(x) for x in row['tasks'] if str(x).strip()),
                                    'images':[images[(cam,t)] for cam in cameras], 'future_images':[images[(cam,future)] for cam in cameras],
                                    'image_provenance':[image_meta[(cam,t)] for cam in cameras],
                                    'future_image_provenance':[image_meta[(cam,future)] for cam in cameras],
                                    'flow_camera':cameras[0], 'episode_file':str(epfile), 'pose_target':str(target_file),
                                    'program':motion_program(poses,action_grip,future,arm_names),
                                    'program_label_source':'trajectory_derived_motion_window_v1', 'semantic_atom_verified':False,
                                    'program_replay_verified':False, 'source_instruction_segments':segments,
                                    'movement_direction_parent_frame':direction.tolist(),
                                    'movement_direction_valid':(distance > 0.001).tolist(),
                                    'movement_displacement_m':displacement.tolist(),
                                    'action_supervision_available':domain == 'droid',
                                    'action_representation':'native_absolute_joint_target_plus_gripper',
                                    'pose_reference':'native parent frame retained; standardized local EEF axes; no cross-domain origin calibration'})
                write_json(cached, episode_records)
                records.extend(episode_records)
                accepted += 1;completed += 1
                writer.add_scalar('episodes/converted',completed,completed)
                writer.add_scalar('episodes/planned',planned,completed)
                writer.add_scalar('samples/converted',len(records),completed)
                writer.add_scalar('progress/episode_fraction',completed/planned,completed)
                writer.flush()
                write_json(args.root/'evidence/conversion_progress.json',{'episodes':completed,'planned':planned,'samples':len(records),'updated_at':time.time(),'last_episode':uid})
                print(json.dumps({'event':'episode_converted','episode':uid,'episodes':completed,'samples':len(records)}),flush=True)
            except Exception as e:
                errors.append({'episode_id':uid,'error_type':type(e).__name__,'error':str(e)})
                print(json.dumps({'event':'episode_rejected',**errors[-1]}),flush=True)
                if len(errors) >= 25 and completed == 0:
                    raise RuntimeError('No convertible episode after 25 errors; stop instead of retrying indefinitely')
        if accepted < required:
            raise RuntimeError(f'{domain}: only {accepted}/{required} episodes converted')
    for split in ['train','validation','test']:
        with (output/f'{split}.jsonl').open('w') as f:
            for r in records:
                if r['split'] == split:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    stats = {}
    for domain in sorted(set(r['domain'] for r in records)):
        episodes = sorted(set(r['episode_file'] for r in records if r['domain']==domain and r['split']=='train' and r['action_supervision_available']))
        if episodes:
            a = np.concatenate([np.load(p)['action'] for p in episodes]);s=np.concatenate([np.load(p)['state'] for p in episodes])
            stats[domain]={'action_mean':a.mean(0).tolist(),'action_std':np.maximum(a.std(0),0.01).tolist(),
                           'state_mean':s.mean(0).tolist(),'state_std':np.maximum(s.std(0),0.01).tolist(),
                           'training_episodes':len(episodes)}
    write_json(output/'normalization.json',stats)
    write_json(output/'provenance.json',{'units':provenance,'errors':errors,'seed':args.seed,'code_sha256':digest(__file__),
                                      'conversion_args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}})
    counts = dict(Counter(r['split'] for r in records))
    write_json(output/'READY.json',{'episodes':completed,'samples':len(records),'split_samples':counts,'flow_labels_ready':False,
                                   'semantic_atoms_verified':False,'stage2_sources':list(stats),
                                   'manifest_sha256':{s:digest(output/f'{s}.jsonl') for s in counts}})
    writer.add_scalar('progress/complete',1,completed);writer.close()


if __name__ == '__main__':
    main()
