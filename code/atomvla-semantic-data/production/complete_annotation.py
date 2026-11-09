"""Full-episode annotation contract, separate from weak candidate record formats.
Every state transition must be accounted for on each physical arm. Unresolved
intervals remain explicit; they are never relabelled as move/idle automatically.
"""
import collections
VOCABULARY=('approach','move','pick','place','flip','rotate','toggle','open','close','push','pull','insert','press','click','strike','handover')
# Allowed event templates, not inferred labels. Select using observed execution.
EVENT_SCHEMAS={
 'approach': [('arrived',)], 'move':[('arrived',)],
 'pick':[('closed','lifted'),('pregrasp','closed','lifted')],
 'place':[('released',),('supported','released'),('preplace','supported','released')],
 'flip':[('completed',),('turning','completed')], 'rotate':[('completed',),('turning','completed')],
 'toggle':[('changed',)], 'open':[('opened',),('engaged','opened')], 'close':[('closed',),('engaged','closed')],
 'push':[('arrived',),('contact','arrived')], 'pull':[('arrived',),('contact','arrived')],
 'insert':[('inserted',),('aligned','inserted')],
 'press':[('pressed',),('pressed','released')], 'click':[('triggered',),('triggered','released')],
 'strike':[('contact',),('contact','retracted')],
 'handover':[('receiver_grasped','giver_released'),('aligned','receiver_grasped','giver_released')]
}

def evidence_valid(e):
 return isinstance(e,dict) and e.get('accepted') is True and all(isinstance(e.get(k),str) and bool(e[k].strip()) for k in ('reviewer','evidence_ref','reviewed_at'))

def intervals(mask):
 result=[];start=None
 for i,v in enumerate(list(mask)+[False]):
  if v and start is None:start=i
  elif not v and start is not None:result.append({'start':start,'end':i});start=None
 return result

def audit(n,arms,annotations,non_action=None):
 """Half-open transition ranges [start,end), corresponding state endpoints.
 Non-action intervals require independent evidence: no motion is not sufficient
 to infer idle because holds/contact maintenance can be meaningful operations.
 """
 if type(n)is not int or n<2:raise ValueError('at least two state rows required')
 if not arms or len(set(arms))!=len(arms):raise ValueError('physical arm mapping required')
 count={arm:[0]*(n-1) for arm in arms};resolved={arm:[0]*(n-1) for arm in arms};errors=[];verbs=collections.Counter()
 for i,a in enumerate(annotations):
  try:
   s,e=a['start'],a['end'];arm=a['arguments']['arm'];targets=list(arms) if arm=='both' else [arm]
   if type(s)is not int or type(e)is not int or not 0<=s<e<n or any(x not in arms for x in targets):raise ValueError('invalid interval/arm')
   verb=a['verb'];events=a['events'];roles=tuple(x['role'] for x in events);rows=[x['row'] for x in events]
   if verb not in EVENT_SCHEMAS or roles not in EVENT_SCHEMAS[verb]:raise ValueError('unsupported verb/event schema')
   if not rows or any(type(t)is not int for t in rows) or any(b<=a for a,b in zip([s]+rows,rows)) or rows[-1]!=e:raise ValueError('event ordering/end mismatch')
   args=a['arguments']
   if not all(isinstance(args.get(k),str) and args[k].strip() for k in ('object','target','arm','completion')):raise ValueError('missing string binding')
   verified=evidence_valid(a.get('verification')) and all(evidence_valid(a.get(k)) for k in ('binding_verification','boundary_verification')) and all(evidence_valid(x.get('verification')) for x in events)
   verbs[verb]+=1
   for x in targets:
    for t in range(s,e):count[x][t]+=1;resolved[x][t]+=int(verified)
  except (ValueError,KeyError,TypeError) as exc:errors.append({'annotation_index':i,'error':str(exc)})
 for i,a in enumerate(non_action or []):
  try:
   s,e=a['start'],a['end'];arm=a['arm']
   if a.get('kind') not in ('idle','sensor_padding','unavailable_observation'):raise ValueError('unknown non-action kind')
   if not evidence_valid(a.get('verification')):raise ValueError('non-action classification requires evidence')
   if type(s)is not int or type(e)is not int or not 0<=s<e<n or arm not in arms:raise ValueError('invalid non-action range')
   for t in range(s,e):count[arm][t]+=1;resolved[arm][t]+=1
  except (ValueError,KeyError,TypeError) as exc:errors.append({'non_action_index':i,'error':str(exc)})
 by_arm={}
 for arm in arms:
  c=count[arm];v=resolved[arm]
  by_arm[arm]={'total_transitions':n-1,'candidate_covered_transitions':sum(x>0 for x in c),'verified_covered_transitions':sum(x>0 for x in v),
   'unannotated_intervals':intervals(x==0 for x in c),'overlap_intervals':intervals(x>1 for x in c),'unverified_intervals':intervals(x==0 for x in v)}
 complete=not errors and all(not x['unannotated_intervals'] and not x['overlap_intervals'] and not x['unverified_intervals'] for x in by_arm.values())
 return {'complete_annotation':complete,'arms':by_arm,'errors':errors,'candidate_atom_counts':dict(verbs),'scope':'full_available_source_episode','interval_convention':'transition [start,end), terminal state row=end'}
