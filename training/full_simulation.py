"""Official HDF5 adapters; preserve native control spaces and RGB conventions."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse, io, json, tarfile,time,traceback,zipfile
from pathlib import Path
import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image
from full_common import dump,sha,split_for,labels,add_npz
from full_hf_download import download
from full_lerobot import ROOT,CONTROL,DATA

def png(a,flip=False):
    image=Image.fromarray(np.asarray(a,dtype=np.uint8)) if not isinstance(a,(bytes,np.bytes_)) else Image.open(io.BytesIO(bytes(a)))
    image=image.convert('RGB')
    if flip:image=image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    b=io.BytesIO();image.resize((224,224),Image.Resampling.BILINEAR).save(b,format='PNG');return b.getvalue()

def convert_episode(h,uid,meta,archive,source,member=None,episode_name=None):
    kind=meta['source_family']
    if kind=='libero':
        obs=h['obs'];n=len(h['actions']);p=np.asarray(obs['ee_pos']);ori=np.asarray(obs['ee_ori'])
        pose=np.concatenate([p,Rotation.from_rotvec(ori).as_quat()],-1)[:,None,:].astype(np.float32)
        state=np.concatenate([obs['joint_states'][()],obs['gripper_states'][()]],axis=-1).astype(np.float32)
        action=np.asarray(h['actions'],np.float32)
        grip=np.sum(np.abs(obs['gripper_states'][()]),axis=-1,keepdims=True).astype(np.float32)
        cameras=[k for k in obs if k.endswith('_rgb')]
        meta.update({'action_supervision_available':True,'action_representation':'libero_OSC_POSE_normalized_delta_xyz_axisangle_gripper',
                     'gripper_program_target_source':'observed_finger_aperture_sum_abs_qpos_m; requires embodiment adapter, not normalized command',
                     'action_row_offset':1,'action_alignment_note':'source records observation after env.step(action[j]); from observation[t] supervise action[t+1], mask final row',
                     'pose_convention':'world position; source ee_ori axis-angle converted to xyzw; native robosuite EEF axes retained',
                     'image_transform':'source OpenGL RGB vertical flip then resize to 224x224'})
        def images(indices):
            return {cam:{t:(png(obs[cam][t],flip=True),{'source_hdf5':str(source),'dataset':episode_name+'/obs/'+cam,'source_row':t,'vertical_flip':True}) for t in indices} for cam in cameras}
        ts=np.arange(n)/meta['fps']
    elif 'joint_action/vector' in h:
        n=len(h['joint_action/vector']);pose=np.stack([h['endpose/left_endpose'][()],h['endpose/right_endpose'][()]],axis=1).astype(np.float32)
        # RoboTwin uses transforms3d mat2quat / SAPIEN wxyz in its saved endpose.
        pose=pose[...,[0,1,2,4,5,6,3]]
        state=np.asarray(h['joint_action/vector'],np.float32);action=np.zeros((n,0),np.float32)
        grip=np.stack([h['endpose/left_gripper'][()],h['endpose/right_gripper'][()]],axis=-1).astype(np.float32)
        obs=h['observation'];cameras=[k for k in ['head_camera','left_camera','right_camera','front_camera'] if k in obs and 'rgb' in obs[k]]
        meta.update({'action_supervision_available':False,'action_representation':'no_verified_control_target; joint_action/vector is measured qpos in source get_obs',
                     'action_exclusion_reason':'source labels are measured joints; do not silently treat same-frame state as control action',
                     'pose_convention':'native RoboTwin transformed endpose in world coordinates; wxyz converted to xyzw; native local axes retained',
                     'gripper_program_target_source':'observed_normalized_gripper_state',
                     'image_transform':'decode original JPEG/PNG with RGB conversion, resize to 224x224',
                     'source_timebase':'not embedded in HDF5; use explicit row horizons until sampling rate verified'})
        def images(indices):
            return {cam:{t:(png(obs[cam]['rgb'][t]),{'source_zip':str(source),'source_member':member,'dataset':'observation/'+cam+'/rgb','source_row':t}) for t in indices} for cam in cameras}
        ts=np.arange(n,dtype=np.float64) # row coordinate; explicitly not seconds
    elif all(k in h for k in ['state/left_ee_poses','state/right_ee_poses','action','vision','data_format_version']):
        if h['data_format_version'].asstr()[()]!='v1.0' or h.attrs.get('source_format')!='RoboTwin':raise ValueError('Unverified standardized RoboTwin schema')
        pose=np.stack([h['state/left_ee_poses'][()],h['state/right_ee_poses'][()]],axis=1).astype(np.float32)
        n=len(pose);pose=pose[...,[0,1,2,4,5,6,3]]
        grip=np.concatenate([h['state/left_ee_joint_states'][()].reshape(n,1),h['state/right_ee_joint_states'][()].reshape(n,1)],axis=1).astype(np.float32)
        # Some native joint columns stay zero; their control semantics need replay. Use EEF/gripper
        # state for stage 1 and retain every original state/action column separately.
        state=np.concatenate([pose[:,0],grip[:,:1],pose[:,1],grip[:,1:]],axis=1).astype(np.float32)
        action=np.zeros((n,0),np.float32)
        for key in h['state']:
            if key in h['action']:
                current=h['state'][key][()];future=h['action'][key][()]
                if current.shape!=future.shape or len(current)!=n:raise ValueError('Standardized state/action shape mismatch')
                if not np.allclose(future[:-1],current[1:],atol=2e-6,rtol=0):raise ValueError('Standardized next-state alignment mismatch')
        raw_instruction=json.loads(h['instructions'].asstr()[()])
        if not isinstance(raw_instruction,list):raise ValueError('Unexpected standardized instruction encoding')
        meta['instruction']=next((str(x) for x in raw_instruction if str(x)),meta.get('instruction',''))
        meta['source_instruction_record']=raw_instruction
        camera_map={'head_camera':'cam_head','left_camera':'cam_left_wrist','right_camera':'cam_right_wrist','front_camera':'cam_third_view'}
        obs=h['vision'];cameras=[k for k,v in camera_map.items() if v in obs and 'colors' in obs[v]]
        declared=float(h['additional_info/frequency'][()]) if 'additional_info/frequency' in h else None
        meta.update({'fps':None,'source_schema':'RoboTwin_XPolicyLab_v1.0','state_representation':'current_left_eef_xyzw_gripper_right_eef_xyzw_gripper',
                     'action_supervision_available':False,'action_representation':'next_recorded_state_not_verified_control_target',
                     'action_exclusion_reason':'exporter action is next recorded state; zero-valued joint fields and executable control semantics need replay validation',
                     'pose_convention':'native RoboTwin world EEF, source transforms3d wxyz converted to xyzw',
                     'schema_source_revision':'ea8b21121ebb3cd201ff5b3fe361944ac94eda3f',
                     'schema_source_exporter_sha256':'c19a77f69d9b5c624c73f0162c2fee7b5f5979bdfddf9fdef3dda020912371bf',
                     'schema_source_robot_sha256':'3dcd80acc8cab489a4c5edb507cc460dab1724be0226b8d5c4c1b218dee605cb',
                     'source_declared_frequency':declared,
                     'source_timebase':'exporter writes save_freq as frequency; collector uses it as physics-step interval; boundaries add extra samples, so seconds are unverified',
                     'gripper_program_target_source':'observed_normalized_gripper_state','image_transform':'decode original encoded colors to RGB, resize to 224x224'})
        def images(indices):
            return {cam:{t:(png(obs[camera_map[cam]]['colors'][t]),{'source_zip':str(source),'source_member':member,'dataset':'vision/'+camera_map[cam]+'/colors','source_row':t}) for t in indices} for cam in cameras}
        ts=np.arange(n,dtype=np.float64)
    else:raise ValueError('Unsupported RoboTwin HDF5 schema')
    meta['source_image_locators']={'path':str(source),'member':member,'episode_group':episode_name,'cameras':cameras,'all_source_rows_retained':True}
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w') as ep:
        result=labels(ep,uid,pose,state,action,grip,np.arange(n),ts,meta,images)
        if kind=='libero':add_npz(ep,uid+'/source_numeric.npz',states=h['states'][()],robot_states=h['robot_states'][()],rewards=h['rewards'][()],dones=h['dones'][()],ee_ori_source=ori)
        else:
            calibration={}
            for cam in cameras:
                original=obs[camera_map[cam]] if meta.get('source_schema')=='RoboTwin_XPolicyLab_v1.0' else obs[cam]
                for k in ['intrinsic_cv','extrinsic_cv','cam2world_gl','intrinsic_matrix','extrinsics_matrix']:
                    if k in original:calibration[cam+'__'+k]=original[k][()]
            if calibration:add_npz(ep,uid+'/camera_calibration.npz',**calibration)
            if meta.get('source_schema')=='RoboTwin_XPolicyLab_v1.0':
                numeric={group+'__'+key:h[group][key][()] for group in ['state','action'] for key in h[group]}
                add_npz(ep,uid+'/source_numeric.npz',**numeric)
    buffer.seek(0)
    with tarfile.open(fileobj=buffer,mode='r') as ep:
        for m in ep:archive.addfile(m,ep.extractfile(m))
    return result

def run(item,repo,revision,sample_only=False):
    kind='libero' if 'LIBERO' in repo else 'robotwin'
    sid=kind+'__'+item['path'].replace('/','__').replace('.hdf5','').replace('.zip','')
    target=DATA/'shards'/kind/(sid+'.tar');receipt=target.with_suffix('.json');target.parent.mkdir(parents=True,exist_ok=True)
    if receipt.exists():return json.loads(receipt.read_text())
    progress=CONTROL/'worker_progress'/(sid+'.json')
    src=download(item,repo,revision,DATA/'sources',progress)
    errors=[];records=[];start=time.time();temp=target.with_suffix('.partial.tar')
    if kind=='libero':
        with h5py.File(src,'r') as f,tarfile.open(temp,'w') as tar:
            data=f['data'];attrs={k:str(v) for k,v in data.attrs.items()}
            info=json.loads(attrs['problem_info']);env=json.loads(attrs['env_args']);fps=env['env_kwargs'].get('control_freq')
            if fps is None:raise ValueError('LIBERO source control_freq missing; verify explicit timebase')
            names=sorted(data,key=lambda x:int(x.split('_')[-1]))
            for name in names:
                uid='libero__'+item['path'].replace('/','__').removesuffix('_demo.hdf5')+'__'+name
                meta={'source_family':kind,'domain':'libero_'+item['path'].split('/')[0],'domain_type':'simulation','split':split_for(uid,{}),
                      'fps':fps,'instruction':info['language_instruction'],'source_repo':repo,'source_revision':revision,'source_sha256':item['lfs']['oid'],'source_hdf5':str(src),
                      'source_group':name,'source_attributes':attrs,'model_file':str(data[name].attrs.get('model_file',''))}
                try:records.append(convert_episode(data[name],uid,meta,tar,src,episode_name='data/'+name))
                except Exception as e:errors.append({'episode_id':uid,'error':str(e),'traceback':traceback.format_exc()})
                dump(progress,{'phase':'convert','time':time.time(),'episodes':len(records),'errors':len(errors),'total':len(names)})
    else:
        with zipfile.ZipFile(src) as z,tarfile.open(temp,'w') as tar:
            names=sorted(n for n in z.namelist() if n.endswith('.hdf5') and '/data/' in n)
            if not names:raise ValueError('No episode HDF5 in official archive')
            text_by_id={}
            for name in z.namelist():
                if name.endswith('.json') and any(x in name for x in ['/instructions/','/instruction/']):
                    text_by_id[Path(name).stem]=json.loads(z.read(name))
            for name in names:
                eid=Path(name).stem;uid=sid+'__'+eid
                instruction_record=text_by_id.get(eid,{})
                if isinstance(instruction_record,dict):
                    instr=instruction_record.get('seen',instruction_record.get('instruction',[]))
                else:instr=instruction_record
                instruction=instr if isinstance(instr,str) else str(instr[0]) if isinstance(instr,list) and instr else ''
                meta={'source_family':kind,'domain':'robotwin_'+Path(item['path']).stem,'domain_type':'simulation','split':split_for(uid,{}),'fps':None,
                      'instruction':instruction,'source_instruction_record':instruction_record,'source_repo':repo,'source_revision':revision,'source_sha256':item['lfs']['oid'],
                      'source_member':name,'task':item['path'].split('/')[1],'robot_variant':Path(item['path']).stem,'horizon_rows':16}
                try:
                    with h5py.File(io.BytesIO(z.read(name)),'r') as h:records.append(convert_episode(h,uid,meta,tar,src,member=name))
                except Exception as e:errors.append({'episode_id':uid,'error':str(e),'traceback':traceback.format_exc()})
                dump(progress,{'phase':'convert','time':time.time(),'episodes':len(records),'errors':len(errors),'total':len(names)})
    temp.replace(target);result={'id':sid,'source_family':kind,'tar':str(target),'sha256':sha(target),'bytes':target.stat().st_size,'episodes':records,'errors':errors,'attempted':len(names),'seconds':time.time()-start}
    dump(receipt,result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--item',required=True);a=p.parse_args();d=json.loads(Path(a.item).read_text());r=run(d['item'],d['repo'],d['revision']);print(json.dumps({'id':r['id'],'episodes':len(r['episodes']),'errors':r['errors']}))
