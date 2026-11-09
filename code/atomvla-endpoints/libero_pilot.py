"""Paired endpoint reconstruction in native LIBERO goal environments.

This is a SCRIPTED, instrumented demonstration pilot, not a public demonstration
dataset evaluation. Semantic boundaries are recorded when source primitives run.
No learned policy or model is trained. Only the source generator sees objects.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
from scipy.spatial.transform import Rotation
import imageio.v2 as imageio
from geometry import annotate, endpoint_program
from monitor import Monitor


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


class Episode:
    def __init__(self, bddl, seed, out):
        from libero.libero.envs import OffScreenRenderEnv
        self.out = Path(out); self.out.mkdir(parents=True, exist_ok=True)
        self.env = OffScreenRenderEnv(bddl_file_name=str(bddl), use_camera_obs=False,
                                     camera_heights=256, camera_widths=256,
                                     control_freq=20, horizon=4000, ignore_done=True)
        self.env.seed(seed); self.env.reset()
        self.sim = self.env.sim
        self.robot = self.env.robots[0]
        self.eef_id = self.robot.eef_site_id
        self.initial_xml = self.sim.model.get_xml()
        self.initial_state = self.env.get_sim_state().copy()
        self.initial_gripper = self.robot.gripper.current_action.copy()
        self.mode = None

    def pose(self):
        p = self.sim.data.site_xpos[self.eef_id].copy()
        r = self.sim.data.site_xmat[self.eef_id].reshape(3,3).copy()
        return p, r

    def begin(self, mode):
        self.env.reset_from_xml_string(self.initial_xml)
        # XML reset replaces the simulation and robot objects.
        self.sim = self.env.sim
        self.robot = self.env.robots[0]
        self.eef_id = self.robot.eef_site_id
        self.sim.reset(); self.env.set_init_state(self.initial_state)
        self.robot.gripper.current_action = self.initial_gripper.copy()
        self.robot.controller.update(force=True); self.robot.controller.reset_goal()
        self.robot.controller.use_delta = True
        if not np.allclose(self.env.get_sim_state(), self.initial_state, atol=1e-8):
            raise ValueError('Initial simulator state mismatch')
        self.mode = mode; self.grip = -1.; self.actions = []; self.positions = []; self.rotations = []
        self.successes = []; self.segments = []; self.segment_results = []
        self.writer = imageio.get_writer(str(self.out / (mode+'.mp4')), fps=5, codec='libx264',
                                         quality=6, macro_block_size=2)
        self._state()
        self.writer.append_data(self.sim.render(height=256,width=320,camera_name='agentview')[::-1])

    def _state(self):
        p,r = self.pose(); self.positions.append(p); self.rotations.append(Rotation.from_matrix(r).as_quat())

    def step(self, action):
        if len(self.actions) >= 4000:
            raise RuntimeError('Bounded episode step budget reached')
        self.env.step(action)
        self.actions.append(np.asarray(action).copy()); self._state()
        self.successes.append(bool(self.env.check_success()))
        if len(self.actions) % 4 == 0:
            self.writer.append_data(self.sim.render(height=256,width=320,camera_name='agentview')[::-1])

    def move(self, primitive, target, rotation, *, hold_steps=None, max_steps=160):
        target = np.asarray(target); start = len(self.actions); stable = 0
        if primitive == 'grasp': self.grip = 1.
        elif primitive == 'release': self.grip = -1.
        for _ in range(hold_steps or max_steps):
            p,r = self.pose()
            error = target-p
            rerr = Rotation.from_matrix(rotation @ r.T).as_rotvec()
            stable = stable+1 if np.linalg.norm(error)<0.006 and np.linalg.norm(rerr)<0.10 else 0
            if hold_steps is None and stable >= 4:
                break
            # Native OSC_POSE maps unit translations to 0.05 m and rotations to 0.5 rad.
            action = np.r_[np.clip(error/.05,-.30,.30), np.clip(rerr/.5,-.5,.5), self.grip]
            self.step(action)
        end = len(self.actions)
        if end == start:
            raise ValueError('Empty primitive segment')
        p,r = self.pose()
        self.segments.append({'start':start,'end':end,'primitive':primitive,
                              'boundary_source':'instrumented_script_call', 'semantic_verified':False})
        self.segment_results.append({'primitive':primitive,'steps':end-start,
                                     'target_position':target.tolist(),
                                     'position_error_m':float(np.linalg.norm(p-target)),
                                     'rotation_error_deg':float(np.degrees(Rotation.from_matrix(rotation@r.T).magnitude()))})

    def object_position(self, name):
        return self.sim.data.body_xpos[self.env.env.obj_body_id[name]].copy()

    def source(self, obj, target, offset, yaw):
        """Oracle state used only to generate reference demonstration."""
        p,r = self.pose(); self.move('wait',p,r,hold_steps=10)
        rot = Rotation.from_euler('z',yaw,degrees=True).as_matrix() @ np.diag([-1.,1.,-1.])
        self.move('align',self.pose()[0],rot)
        grasp = self.object_position(obj)+offset
        self.move('approach',grasp+[0,0,.18],rot)
        self.move('approach',grasp,rot)
        self.move('grasp',self.pose()[0],rot,hold_steps=18)
        self.move('lift',grasp+[0,0,.18],rot)
        place = self.object_position(target)+[0,0,.045]+self.pose()[0]-self.object_position(obj)
        self.move('transport',place+[0,0,.14],rot)
        self.move('place',place,rot)
        self.move('release',self.pose()[0],rot,hold_steps=18)
        self.move('retreat',place+[0,0,.20],rot)
        self.move('wait',self.pose()[0],rot,hold_steps=30)

    def finish(self):
        self.writer.close()
        result = {'mode':self.mode,'steps':len(self.actions),
                  'final_native_success':bool(self.env.check_success()),
                  'ever_native_success':any(self.successes),
                  'segment_results':self.segment_results}
        np.savez_compressed(self.out/(self.mode+'_trace.npz'), positions=np.asarray(self.positions),
                            quaternions_xyzw=np.asarray(self.rotations), actions=np.asarray(self.actions),
                            success=np.asarray(self.successes))
        dump(self.out/(self.mode+'_result.json'),result)
        return result

    def execute(self, program, pose=False):
        # The executor sees only the serialized program and current robot state.
        # It does not inspect reference trajectories or object positions.
        for call in program:
            rot = Rotation.from_quat(call['target_quaternion_xyzw']).as_matrix() if pose else self.pose()[1]
            kind = call['primitive']
            hold = 18 if kind in ['grasp','release'] else 30 if kind == 'wait' else None
            self.move(kind,call['target_xyz_m'],rot,hold_steps=hold)


def run(a):
    out = Path(a.out); out.mkdir(parents=True,exist_ok=True)
    metrics = Monitor(out,'libero')
    tasks = [
        ('put_the_bowl_on_the_plate','akita_black_bowl_1','plate_1',[.045,0,.025],0),
        ('put_the_cream_cheese_in_the_bowl','cream_cheese_1','akita_black_bowl_1',[0,0,.005],90)]
    protocol = {'experiment':'scripted_demonstration_endpoint_pilot',
                'dataset':'new scripted trajectories in unperturbed LIBERO goal BDDL, from LIBERO-PRO checkout',
                'seeds':a.seeds, 'tasks':[t[0] for t in tasks],
                'boundaries':'instrumented script calls; not existing human semantic alignments',
                'conditions':['source','raw_action_replay','endpoint_xyz','endpoint_pose'],
                'endpoint_xyz_inputs':['primitive','terminal measured XYZ'],
                'endpoint_xyz_orientation':'hold current EEF orientation at each call',
                'gripper_policy':'grasp closes; release opens; other verbs retain previous command',
                'executor':'closed-loop bounded native OSC_POSE; no source timing in endpoint conditions',
                'endpoint_max_steps':160, 'fixed_gripper_steps':18, 'fixed_wait_steps':30,
                'evaluation':'official check_success at episode end; ever success separately',
                'no_model_training':True, 'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    dump(out/'libero_protocol.json',protocol)
    completed=0; records=[]
    total=len(tasks)*len(a.seeds)*4
    metrics.write(1, **{'status/waiting_for_environment':0,'evaluation/running':1,
                        'evaluation/planned_rollouts':total,'evaluation/completed_episodes':0})
    for task,obj,target,offset,yaw in tasks:
        bddl=Path(a.libero)/'libero/libero/bddl_files/libero_goal'/(task+'.bddl')
        for seed in a.seeds:
            epout=out/'libero'/f'{task}__seed{seed}'; env=None
            record={'task':task,'seed':seed,'status':'running','conditions':{}}
            try:
                env=Episode(bddl,seed,epout)
                (epout/'initial_model.xml').write_text(env.initial_xml)
                np.save(epout/'initial_state.npy',env.initial_state)
                dump(epout/'source_provenance.json',{'bddl':str(bddl), 'bddl_sha256':hashlib.sha256(bddl.read_bytes()).hexdigest(),
                                                    'initial_model_sha256':hashlib.sha256(env.initial_xml.encode()).hexdigest()})
                env.begin('source'); env.source(obj,target,np.asarray(offset),yaw)
                actions=np.asarray(env.actions).copy(); positions=np.asarray(env.positions).copy()
                rotations=np.asarray(env.rotations).copy(); segments=list(env.segments)
                result=env.finish(); record['conditions']['source']=result; completed+=1
                labels=annotate(positions,segments,frame='libero_world',arm_names=['panda'],
                                quaternions_xyzw=rotations[:,None,:])
                dump(epout/'direction_annotations.json',labels)
                program=endpoint_program(positions,segments)
                dump(epout/'program_xyz.json',program)
                pose_program=[{**c,'target_quaternion_xyzw':rotations[s['end']].tolist()}
                              for c,s in zip(program,segments)]
                dump(epout/'program_pose.json',pose_program)
                metrics.write(completed,**{'selection/completed_episodes':len(records)+1,
                                           'annotation/completed_segments':len(segments),
                                           'evaluation/completed_episodes':completed,
                                           'evaluation/running':1})
                for mode in ['raw_action_replay','endpoint_xyz','endpoint_pose']:
                    env.begin(mode)
                    if mode=='raw_action_replay':
                        for action in actions: env.step(action)
                    else:
                        # Reload execution input from disk; no direction sidecar is passed.
                        payload=json.loads((epout/('program_pose.json' if mode=='endpoint_pose' else 'program_xyz.json')).read_text())
                        env.execute(payload,pose=mode=='endpoint_pose')
                    result=env.finish(); completed+=1
                    if mode=='raw_action_replay':
                        result['max_measured_position_difference_m']=float(np.linalg.norm(np.asarray(env.positions)-positions,axis=1).max())
                        dump(epout/(mode+'_result.json'),result)
                    else:
                        measured=np.asarray(env.positions)[[s['end'] for s in env.segments]]
                        expected=positions[[s['end'] for s in segments]]
                        result['mean_endpoint_error_m']=float(np.linalg.norm(measured-expected,axis=1).mean())
                        dump(epout/(mode+'_result.json'),result)
                    record['conditions'][mode]=result
                    metrics.write(completed,**{'evaluation/completed_episodes':completed,
                                               'evaluation/completion_fraction':completed/total,
                                               'evaluation/running':1,mode+'/native_success':result['final_native_success']})
                record['status']='complete'
                record['valid_reference']=bool(record['conditions']['source']['final_native_success'] and
                                               record['conditions']['raw_action_replay']['final_native_success'] and
                                               record['conditions']['raw_action_replay']['max_measured_position_difference_m']<1e-5)
            except Exception as e:
                record.update(status='error',error=str(e),traceback=traceback.format_exc())
                print(record['traceback'],flush=True)
            finally:
                if env is not None:
                    try: env.writer.close(); env.env.close()
                    except Exception: pass
            records.append(record); dump(out/'libero_results.json',records)
            print(json.dumps({'task':task,'seed':seed,'status':record['status'],
                              'valid_reference':record.get('valid_reference'),
                              'success':{k:v['final_native_success'] for k,v in record['conditions'].items()}}),flush=True)
    metrics.write(completed+1,**{'evaluation/running':0,'evaluation/complete':1,
                                  'evaluation/completed_episodes':completed,
                                  'evaluation/error_cases':sum(x['status']=='error' for x in records),
                                  'evaluation/valid_reference_cases':sum(x.get('valid_reference',False) for x in records)})
    metrics.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--libero',required=True); p.add_argument('--out',required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[11,23,37]); a=p.parse_args(); run(a)
