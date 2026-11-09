"""Decode saved MuJoCo qpos without rendering or executing untrusted XML.
This is state extraction, NOT simulation/contact replay. Alignment is measured
against recorded robot joints, not assumed from dataset name.
"""
import hashlib,xml.etree.ElementTree as ET
import numpy as np

WIDTH={'free':(7,6),'ball':(4,3),'slide':(1,1),'hinge':(1,1)}

def layout(xml):
 root=ET.fromstring(xml)
 if root.findall('.//include') or root.findall('.//composite') or root.findall('.//flexcomp'):raise ValueError('expanded MJCF required')
 world=root.find('worldbody')
 if world is None:raise ValueError('missing worldbody')
 # Ambiguous inherited joint types fail closed rather than guessing offsets.
 defaults=[x for x in root.findall('.//default/joint') if x.get('type') not in (None,'hinge')]
 if defaults:raise ValueError('non-hinge default requires compiled joint layout')
 joints=[];nq=nv=0
 for body in world.iter('body'):
  for j in body:
   if j.tag not in ('joint','freejoint'):continue
   kind='free' if j.tag=='freejoint' else j.get('type','hinge')
   if kind not in WIDTH:raise ValueError('unsupported joint type')
   name=j.get('name')
   if not name or any(x['name']==name for x in joints):raise ValueError('missing or duplicate joint identity')
   q,v=WIDTH[kind];joints.append({'name':name,'body':body.get('name'),'type':kind,'qpos_start':nq,'qpos_width':q,'qvel_start':nv,'qvel_width':v,'range':j.get('range')});nq+=q;nv+=v
 return joints,nq,nv

def extract(xml,states,observed_joints,observed_gripper,tolerance=1e-6):
 joints,nq,nv=layout(xml);s=np.asarray(states,float);obs=np.asarray(observed_joints,float);g=np.asarray(observed_gripper,float)
 if s.ndim!=2 or s.shape[1]!=1+nq+nv:raise ValueError('unrecognized flattened state layout (time + qpos + qvel only)')
 if len(s)!=len(obs) or len(g)!=len(s) or not np.isfinite(s).all():raise ValueError('invalid state shape/values')
 if len(s)<3:raise ValueError('insufficient rows to establish alignment')
 robot=[j for j in joints if j['name'].startswith('robot0_joint')];fingers=[j for j in joints if j['name'].startswith('gripper0_finger_joint')]
 if len(robot)!=obs.shape[1] or g.ndim!=2 or len(fingers)!=g.shape[1]:raise ValueError('observation joint mapping mismatch')
 checks=robot+fingers;recorded=np.concatenate([obs,g],axis=1);qpos=s[:,1:1+nq]
 values=np.column_stack([qpos[:,j['qpos_start']] for j in checks]);candidates={}
 for offset in (-1,0,1):
  rows=np.arange(max(0,-offset),min(len(s),len(s)-offset));err=float(np.max(abs(values[rows+offset]-recorded[rows])));candidates[offset]=err
 matches=[k for k,v in candidates.items() if v<=tolerance]
 if len(matches)!=1:raise ValueError('ambiguous or absent state/observation alignment: '+str(candidates))
 offset=matches[0];valid=(np.arange(len(s))+offset>=0)&(np.arange(len(s))+offset<len(s));objects={};articulations={}
 for j in joints:
  if j in checks:continue
  arr=np.full((len(s),j['qpos_width']),np.nan);arr[valid]=qpos[np.arange(len(s))[valid]+offset,j['qpos_start']:j['qpos_start']+j['qpos_width']]
  if j['type']=='free':
   quat=arr[valid,3:]
   if np.max(abs(np.linalg.norm(quat,axis=1)-1))>.015:raise ValueError('invalid object quaternion')
   objects[j['name']]={'body':j['body'],'xyz':arr[:,:3],'quat_wxyz':arr[:,3:]}
  elif j['type'] in ('hinge','slide'):articulations[j['name']]={**j,'position':arr[:,0]}
 return {'objects':objects,'articulations':articulations,'valid':valid,'alignment':{'state_row_offset':offset,'max_joint_errors':candidates,'compared_channels':len(checks),'xml_sha256':hashlib.sha256(xml.encode()).hexdigest(),'flattened_state_width':s.shape[1],'contact_evidence_available':False,'state_layout':'time_qpos_qvel','unavailable_observation_rows':np.flatnonzero(~valid).tolist()},'joints':joints}
