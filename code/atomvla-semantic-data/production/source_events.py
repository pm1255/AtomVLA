"""Conservative language-bound weak events for single-arm DROID/LIBERO.
Language references are NOT scene-verified object IDs. Ambiguous instructions
and unsupported manipulations are rejected instead of assigning generic objects.
"""
import re
import numpy as np
from annotate import runs,rest_starts

def binding(instruction):
 s=instruction.lower().strip().rstrip('.')
 if any(re.search(r'\b'+w+r'\b',s) for w in ['open','close','push','pull','turn','stack','pour','insert','then','two','three','both']):return None
 patterns=[r'pick up (.+?) and (?:place|put) it (.+)',r'(?:put|place|move) (.+?) (on top of .+|on .+|in .+|into .+|onto .+|to .+)']
 match=next((m for p in patterns if (m:=re.fullmatch(p,s))),None)
 if not match:return None
 obj,target=match.groups()
 if ' and ' in obj or ' and ' in target:return None
 def symbol(x):return re.sub(r'[^a-z0-9]+','_',re.sub(r'^the\s+','',x)).strip('_')
 return {'object':symbol(obj),'target':symbol(target),'source_text':instruction,'method':'single_object_instruction_reference','scene_binding_verified':False}

def infer(pose,grip,source,instruction):
 b=binding(instruction)
 if b is None:return [],['missing_or_ambiguous_instruction_binding']
 if pose.shape[1]!=1:return [],['single_arm_adapter_only']
 g=np.asarray(grip[:,0],float);p=pose[:,0];n=len(p)
 if source=='droid':
  if np.any((g<-.05)|(g>1.05)):return [],['invalid_droid_gripper_range']
  # Source custom.gripper declares normalized open-positive.
  normalized=g;normalization='source_declared_normalized_open_positive'
 elif source=='libero':
  lo,hi=np.quantile(g,[.05,.95])
  if hi-lo<.015:return [],['insufficient_gripper_aperture_change']
  normalized=np.clip((g-lo)/(hi-lo),0,1);normalization='episode_aperture_quantile_candidate_only'
 else:raise ValueError('source not allowed')
 rests=rest_starts(p);out=[];issues=[]
 for cs,ce in runs(normalized<.25):
  if cs<4 or ce-cs<3:continue
  opens=np.flatnonzero(normalized[:cs]>.9)
  if not len(opens):continue
  onset=int(opens[-1]);before=[t for t in rests if t<onset-1]
  if len(before)>=2 and np.linalg.norm(p[before[-2],:3]-p[before[-1],:3])>=.005:
   pre,contact=before[-2:];premethod='two_observed_stops'
  else:
   contact=onset
   # Geometry-dependent approach event, not uniformly sampled timestamps.
   dist=np.linalg.norm(p[:contact,:3]-p[contact,:3],axis=-1)
   approach=np.flatnonzero(dist>=.05)
   if not len(approach):issues.append('no_observable_approach_5cm');continue
   pre=int(approach[-1]);premethod='last_5cm_approach_crossing_before_closure_candidate'
  if np.linalg.norm(p[pre,:3]-p[contact,:3])<.005:continue
  start=next((t for t in reversed(rests) if t<pre),0)
  lc=np.flatnonzero(p[cs:ce,2]-p[contact,2]>=.025)
  if not len(lc):continue
  cross=cs+int(lc[0])
  if cross<=cs:continue
  closed=cs+int(np.argmin(g[cs:cross]));ls=[t for t in rests if cross<=t<ce and p[t,2]-p[contact,2]>=.025]
  if not ls:continue
  lift=ls[0]
  if not start<pre<closed<lift:continue
  ev={'method':'language_reference_gripper_motion_stop_weak_annotation','binding':b,'gripper_normalization':normalization,'pregrasp_method':premethod,'physical_completion_verified':False,'thresholds':{'rest_step_m':.001,'minimum_lift_m':.025,'minimum_contact_distance_m':.005}}
  def atom(verb,begin,end,events,target,completion):
   return {'verb':verb,'arguments':{'object':b['object'],'target':target,'arm':'left','completion':completion},'start':int(begin),'end':int(end),'events':[{'role':role,'row':int(t)} for role,t in events],'quality':'silver','evidence':ev,'review_reasons':['language_binding_not_scene_verified','event_semantics_not_verified','native_single_arm_stored_in_left_slot']}
  out.append(atom('pick',start,lift,[('pregrasp',pre),('closed',closed),('lifted',lift)],'above_support','held_clear'))
  op=np.flatnonzero(normalized[lift:]>.9)
  if not len(op):continue
  release=lift+int(op[0]);on=np.flatnonzero(normalized[lift:release]>.1);onset2=lift+int(on[0]) if len(on) else release
  ds=[t for t in rests if lift+1<t<onset2]
  if len(ds)>=2:
   preplace,support=ds[-2:]
   if preplace<support<release:out.append(atom('place',lift,release,[('preplace',preplace),('supported',support),('released',release)],b['target'],'released_supported'))
 if not out:issues.append('no_observable_complete_pick_event_sequence')
 return out,sorted(set(issues))
