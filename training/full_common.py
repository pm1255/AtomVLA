"""AtomVLA full_v3: lossless numeric trajectories, honest masks, tar records."""
import hashlib, io, json, os, tarfile
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from prepare_data import geometry_targets, motion_program

def dump(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2));tmp.replace(path)

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        while b:=f.read(8*1024*1024):h.update(b)
    return h.hexdigest()

def add(tar,name,data):
    item=tarfile.TarInfo(name);item.size=len(data);item.mtime=0
    tar.addfile(item,io.BytesIO(data))

def add_json(tar,name,value):
    add(tar,name,json.dumps(value,ensure_ascii=False,allow_nan=False).encode())

def add_npz(tar,name,**arrays):
    b=io.BytesIO();np.savez_compressed(b,**arrays);add(tar,name,b.getvalue())

def split_for(uid,old):
    if uid in old:return old[uid]
    n=int(hashlib.sha256(('atomvla_full_v3:'+uid).encode()).hexdigest()[:8],16)%100
    return 'validation' if n<10 else 'test' if n<20 else 'train'

def labels(tar,uid,poses,state,action,gripper,frame_index,timestamp,meta,images,windows=8):
    """images(local_indices) returns camera -> {index: (PNG bytes, provenance)}."""
    n=len(poses);arms=poses.shape[1]
    if poses.shape!=(n,arms,7) or arms not in (1,2):raise ValueError('unsupported pose shape')
    for value in (poses,state,action,gripper):
        if not np.isfinite(value).all():raise ValueError('nonfinite numeric values')
    norms=np.linalg.norm(poses[...,3:],axis=-1)
    if np.max(np.abs(norms-1))>0.015:raise ValueError('nonunit source quaternion')
    poses=poses.copy();poses[...,3:]/=norms[...,None]
    d=np.diff(poses[...,:3],axis=0,prepend=poses[:1,...,:3]);distance=np.linalg.norm(d,axis=-1)
    valid=distance>0.001;direction=np.where(valid[...,None],d/np.maximum(distance[...,None],1e-8),0)
    angular=np.zeros((n,arms,3),np.float32)
    if n>1:
        for arm in range(arms):angular[1:,arm]=(Rotation.from_quat(poses[:-1,arm,3:]).inv()*Rotation.from_quat(poses[1:,arm,3:])).as_rotvec()
    delta=max(1,round(meta['fps']*.5)) if meta.get('fps') else 16
    # Partition every observed row, without calling heuristic partitions semantic atoms.
    boundaries=[0]
    for i in range(1,n):
        start=boundaries[-1]
        grip_change=np.max(np.abs(gripper[i]-gripper[i-1]))>0.2
        turn=(i-start>=4 and np.any(valid[i]&valid[i-1]&(np.sum(direction[i]*direction[i-1],axis=-1)<0.707)))
        if grip_change or turn or i-start>=delta:boundaries.append(i)
    if boundaries[-1]!=n-1:boundaries.append(n-1)
    segments=[];names=['arm'] if arms==1 else ['left','right']
    for start,end in zip(boundaries[:-1],boundaries[1:]):
        displacement=poses[end,:,:3]-poses[start,:,:3];dist=np.linalg.norm(displacement,axis=-1);ok=dist>0.001
        segments.append({'start_row':start,'end_row_inclusive':end,'endpoint_pose_xyzw':poses[end].tolist(),
                         'displacement_m':displacement.tolist(),'movement_direction_parent_frame':np.where(ok[:,None],displacement/np.maximum(dist[:,None],1e-8),0).tolist(),
                         'direction_valid':ok.tolist(),'program':motion_program(poses,gripper,end,names),
                         'alignment_source':'observed_gripper_transition_direction_change_or_max_horizon','semantic_atom_verified':False})
    future=np.minimum(np.arange(n)+delta,n-1)
    add_npz(tar,uid+'/trajectory.npz',poses_xyzw=poses,state=state,action=action,gripper=gripper,
            timestamp=np.asarray(timestamp),original_frame_index=np.asarray(frame_index),
            displacement_m=d,movement_direction_parent_frame=direction,movement_direction_valid=valid,
            angular_increment_rotvec=angular,future_endpoint_row=future,future_endpoint_pose_xyzw=poses[future])
    add_json(tar,uid+'/program_segments.json',segments)
    indices=np.unique(np.rint(np.linspace(0,max(0,n-delta-1),min(windows,max(0,n-delta)))).astype(int)) if n>delta else np.array([],int)
    wanted=sorted(set(indices.tolist()+np.minimum(indices+delta,n-1).tolist()))
    decoded=images(wanted) if wanted else {}
    refs={};image_hashes={}
    for camera,frames in decoded.items():
        for t,(raw,provenance) in frames.items():
            name=f'{uid}/images/{camera}_{t:06d}.png';add(tar,name,raw)
            refs[camera,t]={**provenance,'member':name};image_hashes[camera,t]=hashlib.sha256(raw).hexdigest()
    camera_names=list(decoded);records=[]
    for t in indices:
        t=int(t);end=min(n-1,t+delta);pos,rot,mask,fi=geometry_targets(poses,t,end)
        name=f'{uid}/targets/{t:06d}.npz';add_npz(tar,name,position_delta_m=pos,rotation_6d=rot,pose_mask=mask,future_indices=fi)
        records.append({'id':uid+f'__{t:06d}','episode_id':uid,'frame_index':t,'future_frame_index':end,
                        'delta_seconds':(end-t)/meta['fps'] if meta.get('fps') else None,'timebase_verified':bool(meta.get('fps')),
                        'images':[refs[c,t] for c in camera_names],'future_images':[refs[c,end] for c in camera_names],
                        'image_sha256':[image_hashes[c,t] for c in camera_names],'future_image_sha256':[image_hashes[c,end] for c in camera_names],
                        'pose_target_member':name,'trajectory_member':uid+'/trajectory.npz',
                        'program':motion_program(poses,gripper,end,names),
                        'flow_label_status':'waiting','action_supervision_available':meta['action_supervision_available'],
                        'action_row_offset':meta.get('action_row_offset',0),'action_representation':meta['action_representation'],
                        'gripper_program_target_source':meta.get('gripper_program_target_source','future_observed_gripper_state'),
                        'instruction':meta['instruction'],'instruction_available':bool(meta['instruction']),
                        'domain':meta['domain'],'domain_type':meta['domain_type'],'split':meta['split']})
    meta={**meta,'episode_id':uid,'frames':n,'windows':len(records),'image_count':len(refs),'segments':len(segments),
          'semantic_atom_verified':False,'program_replay_verified':False,'program_label_source':'trajectory_derived_motion_partition_v3',
          'direction_reference':'native parent coordinate frame; not cross-domain calibrated',
          'continuous_window_sampling':'all trajectory rows and full source image locators retained; cached RGB/flow anchors are a subset',
          'trajectory_content_sha256':hashlib.sha256(np.ascontiguousarray(np.concatenate([poses.reshape(n,-1),state,action],axis=-1),dtype='<f4').tobytes()).hexdigest()}
    add_json(tar,uid+'/metadata.json',meta);add_json(tar,uid+'/windows.json',records)
    return meta
