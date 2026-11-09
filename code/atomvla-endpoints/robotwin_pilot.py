"""Instrument native RoboTwin primitive execution, reconstruct measured endpoints.

Source and replay use the same native cuRobo planner. Each native
take_dense_action call is one aligned segment. Capture includes both arms even
when one is stationary. Reference controls never enter endpoint executors.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import traceback
import numpy as np
import imageio.v2 as imageio
import yaml
from geometry import annotate


def dump(path,value):
    def encode(v):
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,np.generic):return v.item()
        raise TypeError(str(type(v)))
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,default=encode,allow_nan=False));tmp.replace(p)


def config(root,out):
    args=yaml.safe_load((root/'task_config/demo_clean.yml').read_text())
    emb=root/yaml.safe_load((root/'task_config/_embodiment_config.yml').read_text())['aloha-agilex']['file_path']
    rob=yaml.safe_load((emb/'config.yml').read_text())
    cam=yaml.safe_load((root/'task_config/_camera_config.yml').read_text())[args['camera']['head_camera_type']]
    args.update(task_name='adjust_bottle',task_config='demo_clean',left_robot_file=str(emb),right_robot_file=str(emb),
                left_embodiment_config=deepcopy(rob),right_embodiment_config=deepcopy(rob),dual_arm_embodied=True,
                head_camera_h=cam['h'],head_camera_w=cam['w'],render_freq=0,save_freq=None,
                save_data=False,collect_data=False,eval_video_log=False,eval_mode=True,
                save_path=str(out/'scratch'))
    return args


def poses(env):
    return np.array([env.robot.get_left_ee_pose(),env.robot.get_right_ee_pose()],dtype=float)


def fingerprint(env):
    return sorted([{'name':x.name,'pose':np.r_[x.get_pose().p,x.get_pose().q].tolist()}
                   for x in env.scene.get_all_actors()],key=lambda x:(x['name'],x['pose']))


class SceneRecorder:
    """Transparent scene proxy: physics still advances in the native scene."""
    def __init__(self,scene,callback):self.original=scene;self.callback=callback
    def __getattr__(self,key):return getattr(self.original,key)
    def step(self):
        self.original.step();self.callback()


class Capture:
    def __init__(self,env,path):
        self.env=env;self.path=Path(path);self.path.mkdir(parents=True,exist_ok=True)
        self.poses=[poses(env)];self.segments=[];self.controls=[];self.step_count=0
        self.scene=env.scene;self.original_dense=env.take_dense_action
        self.writer=imageio.get_writer(str(self.path/'video.mp4'),fps=10,codec='libx264',quality=6,macro_block_size=2)
        env.scene=SceneRecorder(self.scene,self.capture_step)
        env.take_dense_action=self.dense
        self.render()

    def render(self):
        rgb=self.env.get_obs()['observation']['head_camera']['rgb']
        self.writer.append_data(np.asarray(rgb)[...,:3])

    def capture_step(self):
        self.poses.append(poses(self.env));self.step_count+=1
        if self.step_count%25==0:self.render()

    def dense(self,seq,save_freq=-1):
        start=self.step_count; self.controls.append(deepcopy(seq))
        result=self.original_dense(seq,save_freq=None)
        end=self.step_count
        if end<=start:raise ValueError('Empty native segment')
        kinds={}
        for arm in ['left','right']:
            if seq[arm+'_arm'] is not None:kinds[arm]='move'
            elif seq[arm+'_gripper'] is not None:
                kinds[arm]='grasp' if float(seq[arm+'_gripper']['result'][-1])<.5 else 'release'
            else:kinds[arm]='hold'
        self.segments.append({'start':start,'end':end,'primitive_by_arm':kinds,
                              'boundary_source':'native_take_dense_action_call','semantic_verified':False})
        return result

    def finish(self):
        self.writer.close();self.env.scene=self.scene;self.env.take_dense_action=self.original_dense
        p=np.asarray(self.poses)
        np.savez_compressed(self.path/'trace.npz',positions=p[:,:,:3],quaternions_wxyz=p[:,:,3:7])
        dump(self.path/'segments.json',self.segments)
        return {'steps_250hz':self.step_count,'final_native_success':bool(self.env.check_success()),
                'plan_success':bool(self.env.plan_success),'segments':len(self.segments)}


def run(a):
    root=Path(a.robotwin).resolve();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
    os.chdir(root);sys.path[:0]=[str(root),str(root/'description/utils'),str(root/'envs/curobo/src')]
    import torch
    torch.set_num_threads(2)
    import sapien
    import sapien.render
    # Use one explicit raster backend for every condition on the assigned GPU.
    orig_shader=sapien.render.set_camera_shader_dir
    sapien.render.set_camera_shader_dir=lambda *args,**kwargs:orig_shader('default')
    def create_scene(engine,scene_config=None):
        sapien.physx.set_scene_config(scene_config or sapien.SceneConfig())
        return sapien.Scene([sapien.physx.PhysxCpuSystem(),sapien.render.RenderSystem('cuda:0')])
    sapien.Engine.create_scene=create_scene
    from envs.adjust_bottle import adjust_bottle
    from envs.utils.action import ArmTag
    args=config(root,out);completed=0;records=[]
    def progress(finished=False,**kw):
        dump(out/'robotwin_progress.json',{'finished':finished,'metrics':{
            'status/waiting_for_environment':0,'evaluation/running':int(not finished),
            'evaluation/completed_episodes':completed,'evaluation/planned_rollouts':len(a.seeds)*4,
            'evaluation/completion_fraction':completed/(len(a.seeds)*4),**kw}})
    dump(out/'robotwin_protocol.json',{'task':'adjust_bottle','seeds':a.seeds,'source':'new native play_once demonstrations',
        'planner':'native cuRobo, same in source and endpoint variants',
        'renderer':'raster','frame':'RoboTwin get_*_ee_pose world frame, native TCP convention',
        'boundaries':'each native take_dense_action call; not seven-verb human labels',
        'endpoint_xyz_payload':'arm, native primitive, measured terminal XYZ; gripper open/close intrinsic to verb',
        'endpoint_xyz_orientation':'current orientation at call start',
        'endpoint_pose_extra':'terminal quaternion WXYZ',
        'duration':'native planner timing; source durations not passed to endpoint variants',
        'all_steps_physical':True,'no_state_reset_after_episode_start':True,
        'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    progress()
    for seed in a.seeds:
        rec={'seed':seed,'task':'adjust_bottle','conditions':{},'status':'running'}
        epout=out/'robotwin'/f'adjust_bottle__seed{seed}'
        reference_controls=None;reference_fingerprint=None;programs={}
        for mode in ['source','raw_control_replay','endpoint_xyz','endpoint_pose']:
            env=None;capture=None
            try:
                random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
                env=adjust_bottle();env.setup_demo(now_ep_num=0,seed=seed,is_test=True,**deepcopy(args))
                initial=fingerprint(env)
                if reference_fingerprint is None:
                    reference_fingerprint=initial;dump(epout/'initial_scene.json',initial)
                elif ([x['name'] for x in initial]!=[x['name'] for x in reference_fingerprint] or
                      not np.allclose([x['pose'] for x in initial],[x['pose'] for x in reference_fingerprint],atol=1e-5)):
                    raise ValueError('Paired reset scene mismatch')
                capture=Capture(env,epout/mode)
                if mode=='source':env.play_once()
                elif mode=='raw_control_replay':
                    for seq in reference_controls:capture.dense(deepcopy(seq))
                else:
                    payload=json.loads((epout/('program_xyz.json' if mode=='endpoint_xyz' else 'program_pose.json')).read_text())
                    for group in payload:
                        actions=[]
                        for call in group:
                            arm=ArmTag(call['arm']);kind=call['primitive']
                            if kind=='move':
                                current=env.get_arm_pose(arm)
                                quat=call.get('target_quaternion_wxyz',np.asarray(current)[3:7])
                                actions.append(env.move_to_pose(arm,[*call['target_xyz_m'],*quat]))
                            elif kind=='grasp':actions.append(env.close_gripper(arm))
                            elif kind=='release':actions.append(env.open_gripper(arm))
                        if actions:env.move(*actions)
                        if not env.plan_success:break
                saved_controls=deepcopy(capture.controls)
                result=capture.finish();capture=None;completed+=1
                if mode=='source':
                    # Read the artifact, keeping the execution payload independent of source controls.
                    trace=np.load(epout/mode/'trace.npz');segments=json.loads((epout/mode/'segments.json').read_text())
                    # Captured control arrays are retained only for the raw-control baseline.
                    reference_controls=saved_controls
                    p=trace['positions'];q=trace['quaternions_wxyz']
                    labels=annotate(p,segments,frame='robotwin_native_eef_world',arm_names=['left','right'],
                                    dt=1/250,horizon=25,quaternions_xyzw=q[:,:,[1,2,3,0]])
                    dump(epout/'direction_annotations.json',labels)
                    xyz=[];pose=[]
                    for s in segments:
                        gx=[];gp=[]
                        for j,arm in enumerate(['left','right']):
                            kind=s['primitive_by_arm'][arm]
                            if kind=='hold':continue
                            call={'arm':arm,'primitive':kind,'target_xyz_m':p[s['end'],j].tolist()}
                            gx.append(call);gp.append({**call,'target_quaternion_wxyz':q[s['end'],j].tolist()})
                        xyz.append(gx);pose.append(gp)
                    dump(epout/'program_xyz.json',xyz);dump(epout/'program_pose.json',pose)
                    # Plain NPZ/JSON, no executable pickle artifacts.
                    dump(epout/'reference_controls.json',reference_controls)
                if mode=='raw_control_replay':
                    src=np.load(epout/'source/trace.npz')['positions'];now=np.load(epout/mode/'trace.npz')['positions']
                    result['max_measured_position_difference_m']=float(np.linalg.norm(src-now,axis=-1).max()) if src.shape==now.shape else None
                rec['conditions'][mode]=result;dump(epout/mode/'result.json',result)
                progress(**{mode+'/native_success':int(result['final_native_success']),
                            'selection/completed_episodes':len(records)+1})
            except Exception as e:
                rec.update(status='error',error=str(e),traceback=traceback.format_exc())
                print(rec['traceback'],flush=True);break
            finally:
                if capture is not None:
                    try:capture.finish()
                    except Exception:pass
                if env is not None:
                    try:env.close_env()
                    except Exception:pass
        if rec['status']!='error':
            rec['status']='complete'
            rec['valid_reference']=bool(rec['conditions']['source']['final_native_success'] and
                rec['conditions']['raw_control_replay']['final_native_success'] and
                rec['conditions']['raw_control_replay'].get('max_measured_position_difference_m',1)<1e-4)
        records.append(rec);dump(out/'robotwin_results.json',records)
        print(json.dumps({'seed':seed,'status':rec['status'],'conditions':rec['conditions']}),flush=True)
    progress(finished=True,**{'evaluation/error_cases':sum(r['status']=='error' for r in records),
                              'evaluation/valid_reference_cases':sum(r.get('valid_reference',False) for r in records)})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--robotwin',required=True);p.add_argument('--out',required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[100000,100001,100002]);run(p.parse_args())
