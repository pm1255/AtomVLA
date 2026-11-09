"""Per-arm change points from measured trajectory, not action-class labels.
Partitions every source transition. Thresholds are in seconds/metres/radians;
gripper thresholds require declared source units. Quiet spans remain candidates.
"""
from dataclasses import dataclass,asdict
import numpy as np

@dataclass(frozen=True)
class Config:
 translation_speed:float=.015
 rotation_speed:float=.15
 minimum_run_seconds:float=.15
 gripper_change_normalized:float=.12
 gripper_change_metres:float=.006
 object_speed:float=.01
 articulation_speed_slide:float=.01
 articulation_speed_hinge:float=.1


def true_runs(mask):
 a=np.asarray(mask,bool);edges=np.diff(np.r_[False,a,False].astype(int));return list(zip(np.flatnonzero(edges==1).tolist(),np.flatnonzero(edges==-1).tolist()))

def smooth_short(mask,min_rows):
 """Remove isolated short activation bursts; never infer an operation from quiet."""
 result=np.asarray(mask,bool).copy()
 for s,e in true_runs(result):
  if e-s<min_rows:result[s:e]=False
 return result

def build(pose,gripper,timestamps,gripper_units,scene=None,config=None):
 c=config or Config();p=np.asarray(pose,float);g=np.asarray(gripper,float);t=np.asarray(timestamps,float);n=len(p)
 if n<2 or p.ndim!=3 or p.shape[2]!=7 or p.shape[1] not in (1,2) or g.shape!=p.shape[:2] or t.shape!=(n,):raise ValueError('trajectory dimensions')
 if not np.isfinite(p).all() or not np.isfinite(g).all() or not np.isfinite(t).all() or np.any(np.diff(t)<=0):raise ValueError('nonfinite or nonmonotonic trajectory')
 q=p[:,:,3:];norm=np.linalg.norm(q,axis=-1)
 if np.max(abs(norm-1))>.015:raise ValueError('pose quaternion invalid')
 q=q/norm[...,None];dt=np.diff(t);speed=np.linalg.norm(np.diff(p[:,:,:3],axis=0),axis=-1)/dt[:,None];rotation=2*np.arccos(np.clip(abs((q[1:]*q[:-1]).sum(-1)),0,1))/dt[:,None]
 min_rows=max(1,int(np.ceil(c.minimum_run_seconds/np.median(dt))));names=['left','right'][:p.shape[1]];events=[];boundaries={a:{0,n-1} for a in names}
 def add(arm,kind,start,end,**details):
  if not 0<=start<end<n:return
  events.append({'id':f'e{len(events):05d}','arm':arm,'kind':kind,'start':int(start),'end':int(end),'evidence':details,'semantic_verified':False});boundaries[arm].update((int(start),int(end)))
 for i,arm in enumerate(names):
  for kind,signal,threshold in [('translation',speed[:,i],c.translation_speed),('rotation',rotation[:,i],c.rotation_speed)]:
   for s,e in true_runs(smooth_short(signal>threshold,min_rows)):add(arm,kind,s,e,peak_speed=float(signal[s:e].max()),threshold=threshold)
  threshold={'normalized_open_positive':c.gripper_change_normalized,'aperture_metres':c.gripper_change_metres}.get(gripper_units)
  if threshold is None:continue
  # Hysteresis over aperture extrema: event from last reversal to threshold crossing;
  # extend to plateau. No episode-wise normalization of constant/noisy grippers.
  delta=np.diff(g[:,i]);epsilon=threshold/10;direction=np.where(delta>epsilon,1,np.where(delta<-epsilon,-1,0))
  for sign in (-1,1):
   # Close short pauses within a gradual aperture transition, but not reversals.
   mask=direction==sign
   for s,e in true_runs(~mask):
    if s>0 and e<len(mask) and e-s<=min_rows and not np.any(direction[s:e]==-sign):mask[s:e]=True
   for s,e in true_runs(mask):
    change=float(g[e,i]-g[s,i])
    if abs(change)>=threshold:add(arm,'gripper_opening' if sign>0 else 'gripper_closing',s,e,delta=change,units=gripper_units)
 if scene is not None:
  valid=np.asarray(scene['valid'],bool)
  if len(valid)!=n:raise ValueError('scene timeline mismatch')
  for object_id,o in scene['objects'].items():
   xyz=o['xyz'];v=np.linalg.norm(np.diff(xyz,axis=0),axis=-1)/dt;active=smooth_short((v>c.object_speed)&valid[:-1]&valid[1:],min_rows)
   for s,e in true_runs(active):
    for arm in names:add(arm,'object_motion',s,e,object=object_id,displacement_xyz=(xyz[e]-xyz[s]).tolist(),peak_speed=float(v[s:e].max()),contact_proven=False)
  for joint_id,j in scene['articulations'].items():
   values=j['position'];v=np.diff(values)/dt;threshold=c.articulation_speed_slide if j['type']=='slide' else c.articulation_speed_hinge
   for sign in (-1,1):
    for s,e in true_runs(smooth_short((v*sign>threshold)&valid[:-1]&valid[1:],min_rows)):
     for arm in names:add(arm,'articulation_change',s,e,joint=joint_id,body=j['body'],joint_type=j['type'],delta=float(values[e]-values[s]),semantic_direction='requires_object_contract',contact_proven=False)
 parts=[]
 for arm in names:
  b=sorted(boundaries[arm])
  for s,e in zip(b,b[1:]):parts.append({'id':f'{arm}:{s}:{e}','arm':arm,'start':s,'end':e,'event_ids':[v['id'] for v in events if v['arm']==arm and max(v['start'],s)<min(v['end'],e)],'semantic_status':'unresolved'})
 return {'schema':'atomvla_motion_evidence_v1','source_frames':n,'physical_arms':names,'events':events,'partitions':parts,'config':asdict(c),'full_transition_partition':True,'semantic_annotation_complete':False,'gripper_units':gripper_units}
