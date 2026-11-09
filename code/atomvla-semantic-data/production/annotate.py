"""Task-bound event proposals. Silver labels never certify physical success."""
import numpy as np
from task_schema import object_for
EXCLUDE=('click','press','beat','open','close','push','pull','rotate','turn','shake','handover','dump','pour','adjust','ranking','stack','insert','scan','stamp')

def runs(mask):
 a=np.flatnonzero(np.diff(np.r_[False,mask,False].astype(int)))
 return [(int(a0),int(b)) for a0,b in zip(a[::2],a[1::2])]

def rest_starts(pose,threshold=.001):
 speed=np.linalg.norm(np.diff(pose[:,:3],axis=0),axis=-1)
 # Two small increments (three observations) establish a candidate stop.
 return [a+1 for a,b in runs(speed<threshold) if b-a>=2]

def proposals(pose,gripper,task,schema,scene,min_lift=.025):
 n,arms,_=pose.shape
 if any(t in task for t in EXCLUDE):return [],['task_requires_specific_event_schema']
 obj,reason=object_for(schema,scene)
 if reason:return [],[reason]
 result=[];issues=[]
 for arm in range(arms):
  g=gripper[:,arm];p=pose[:,arm]
  if np.any((g<-.05)|(g>1.05)):issues.append('non_normalized_gripper');continue
  rests=rest_starts(p)
  for close_start,close_end in runs(g<.25):
   if close_start<4 or close_end-close_start<3:continue
   before_open=np.flatnonzero(g[:close_start]>.95)
   if not len(before_open):continue
   closure_onset=int(before_open[-1])
   # last two distinct stops before closure: pregrasp then contact. A single
   # continuous approach has no observable pregrasp stop and needs review.
   before=[t for t in rests if t<closure_onset-1]
   if len(before)<2:issues.append('no_distinct_pregrasp_stop');continue
   contact=before[-1];pre=before[-2]
   if np.linalg.norm(p[pre,:3]-p[contact,:3])<.005:
    issues.append('pregrasp_contact_not_separated');continue
   earlier=[t for t in rests if t<pre]
   start=earlier[-1] if earlier else 0
   if start>=pre:continue
   # Closure endpoint: min gripper aperture before significant lift. This
   # accommodates contact-limited grippers that never reach numeric zero.
   lift_candidates=np.flatnonzero(p[close_start:close_end,2]-p[contact,2]>=min_lift)
   if not len(lift_candidates):issues.append('closure_without_lift_evidence');continue
   lift_cross=close_start+int(lift_candidates[0])
   if lift_cross<=close_start:continue
   closed_row=close_start+int(np.argmin(g[close_start:lift_cross]))
   lifts=[t for t in rests if lift_cross<=t<close_end and p[t,2]-p[contact,2]>=min_lift]
   if not lifts:issues.append('no_stable_lift_endpoint');continue
   lift=lifts[0]
   if not start<pre<closed_row<lift:issues.append('ambiguous_event_order');continue
   common={'object':obj,'arm':['left','right'][arm]}
   result.append({'verb':'pick','arguments':{**common,'target':'above_support','completion':'held_clear'},
    'start':start,'end':lift,'events':[{'role':'pregrasp','row':pre},{'role':'closed','row':closed_row},{'role':'lifted','row':lift}],
    'quality':'silver','evidence':{'method':'source_scene_binding_motion_stops_gripper_lift','object_attachment_verified':False,
      'completion_verified':False,'contact_candidate_row':contact,'thresholds':{'open':.95,'closed':.25,'lift_m':min_lift,'rest_step_m':.001,'rest_increments':2}},
    'review_reasons':['pregrasp_stop_semantics_unverified','object_attachment_and_completion_unverified']})
   later_open=np.flatnonzero(g[lift:]>.95)
   if not len(later_open):continue
   release=lift+int(later_open[0])
   onset_candidates=np.flatnonzero(g[lift:release]>.05)
   onset=lift+int(onset_candidates[0]) if len(onset_candidates) else release
   destinations=[t for t in rests if lift+1<t<onset]
   # Separate transport endpoint from the final placement descent, when both
   # stable preplace and support candidates are observable.
   if len(destinations)<2:issues.append('no_distinct_preplace_support_stops');continue
   preplace,support=destinations[-2:]
   target=schema.get('target_symbol') or 'task_goal'
   if np.linalg.norm(p[preplace,:3]-p[lift,:3])>.02:
    result.append({'verb':'move','arguments':{**common,'target':target,'completion':'arrived'},
      'start':lift,'end':preplace,'events':[{'role':'arrived','row':preplace}],
      'quality':'silver','evidence':{'method':'stable_transport_stop_before_descent','completion_verified':False},
      'review_reasons':['target_binding_and_goal_arrival_unverified']})
   if preplace<support<release:
    # Reuse transport start so the preplace event is strictly in the future.
    # Pick/move/place can overlap in the candidate pool; final gold sequences
    # require an explicitly reviewed non-overlapping segmentation.
    result.append({'verb':'place','arguments':{**common,'target':target,'completion':'released_supported'},
      'start':lift,'end':release,'events':[{'role':'preplace','row':preplace},{'role':'supported','row':support},{'role':'released','row':release}],
      'quality':'silver','evidence':{'method':'preplace_stop_descent_stop_and_observed_release','support_contact_verified':False,'completion_verified':False},
      'review_reasons':['support_contact_unverified','stable_object_after_release_unverified','overlaps_transport_candidate']})
 return result,sorted(set(issues))
