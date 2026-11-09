"""Local semantic decisions over measured intervals; bounded refinement queue.
The classifier never emits coordinates, programs or arbitrary frame indices.
The converter assembles and validates string programs + measured endpoint targets.
"""
import hashlib,json,re
import numpy as np
from complete_annotation import EVENT_SCHEMAS
from records import validate_annotation
from common import write_json


def windows(evidence,fps,max_seconds=4.,minimum_seconds=.5):
 """Merge fine change-point partitions into bounded local inspection windows.
These windows are proposals, not primitive boundaries. Strong gripper/object
changes are preserved; a classifier can request splitting at observed frames.
"""
 n=evidence['source_frames'];result=[]
 for arm in evidence['physical_arms']:
  points={0,n-1}
  for ev in evidence['events']:
   if ev['arm']==arm and ev['kind'] in ('gripper_closing','gripper_opening','articulation_change'):
    points.update((ev['start'],ev['end']))
  # Coalesce nearby changes to avoid a one-frame "primitive" at every threshold.
  selected=[0]
  for x in sorted(points-{0,n-1}):
   if x-selected[-1]>=max(2,round(minimum_seconds*fps)) and n-1-x>=max(2,round(minimum_seconds*fps)):selected.append(x)
  selected.append(n-1);bounded=[0]
  for s,e in zip(selected,selected[1:]):
   count=max(1,int(np.ceil((e-s)/max(2,round(max_seconds*fps)))))
   bounded.extend(np.linspace(s,e,count+1,dtype=int)[1:].tolist())
  for s,e in zip(bounded,bounded[1:]):result.append({'arm':arm,'start':s,'end':e,'reason':'bounded_event_context_not_final_primitive'})
 return result

def inspection_rows(window,n,evidence,dense=False):
 s,e=window['start'],window['end'];inside=set(np.linspace(s,e,min(e-s+1,12 if dense else 6),dtype=int).tolist())
 for ev in evidence['events']:
  if ev['arm']==window['arm']:
   inside.update(x for x in (ev['start'],ev['end']) if s<=x<=e)
 # Keep evidence frame endpoints, cap only redundant uniform frames.
 context={max(0,s-3),min(n-1,e+3)}
 return sorted(inside|context)

def request(window,rows,evidence,instruction,scene):
 s,e=window['start'],window['end'];tokens={f'F{i}':r for i,r in enumerate(rows)}
 relevant=[x for x in evidence['events'] if x['arm']==window['arm'] and max(s,x['start'])<min(e,x['end'])]
 objects=list(scene['objects']) if scene else [];articulations=list(scene['articulations']) if scene else []
 text=f'''Classify ONLY this local robot interval: arm={window['arm']}, start row={s}, end row={e}. Images outside this interval are context only. Task text is context, not proof: {json.dumps(instruction)}.
Observed frame tokens: {json.dumps(tokens)}. Measured changes: {json.dumps(relevant)}.
Known simulator object IDs: {json.dumps(objects)}; articulated joint IDs: {json.dumps(articulations)}. If using a known object, copy its exact ID.
Return one JSON object with ONLY fields verb, object, target, events, evidence. Example: {{"verb":"move","object":"gripper","target":"above_cup","events":{{"arrived":"F5"}},"evidence":"Empty gripper visibly moves above the cup."}}.
Allowed verb/event-role sequences: {json.dumps(EVENT_SCHEMAS)}.
Choose one observed operation; object and target must be string identifiers. Events maps each role to an observed frame token. Events occur strictly after interval start; last event is exactly interval end. move/approach need only their endpoint. pick requires gripper closure AND actual acquisition/lift; place requires visible supported release; stationary alone does not mean idle. open/close refer to articulated objects. Never infer success from the task text.
If there are multiple operations or the boundary cuts an operation, return {{"verb":"unresolved","object":"unknown","target":"unknown","events":{{}},"evidence":"Explain which operation/boundary is uncertain","split":"F3"}} with an internal observed frame token if a split helps. Otherwise omit split. For visible idle or sustained hold, use verb idle/hold, empty events and specific visual evidence. Do not force an unsupported operation (e.g. wiping) into move. No coordinates and no program code.'''
 return text,tokens

def assemble(raw,window,tokens,evidence,n,scene=None):
 text=raw.strip();text=re.sub(r'^```(?:json)?\s*','',text);text=re.sub(r'\s*```$','',text);d=json.loads(text)
 if not isinstance(d,dict):raise ValueError('local result must be object')
 verb=d.get('verb');observation=d.get('evidence');s,e=window['start'],window['end'];arm=window['arm']
 if not isinstance(observation,str) or len(observation.strip())<8:raise ValueError('missing specific visual evidence')
 if verb in ('unresolved','idle','hold'):
  if verb=='idle' and any(v['arm']==arm and max(s,v['start'])<min(e,v['end']) for v in evidence['events']):
   raise ValueError('idle contradicts measured motion or gripper/object change')
  result={'kind':verb,'arm':arm,'start':s,'end':e,'observation':observation}
  if 'split' in d:
   row=tokens.get(d['split'])
   if row is None or not s<row<e:raise ValueError('invalid internal split token')
   result['split_row']=row
  return result
 if verb not in EVENT_SCHEMAS:raise ValueError('unknown/unsupported operation')
 if not isinstance(d.get('events'),dict):raise ValueError('events must map roles to frame tokens')
 events=[]
 for role,token in d['events'].items():
  if not isinstance(token,str) or token not in tokens:raise ValueError('invented frame token')
  events.append({'role':role,'row':tokens[token]})
 events.sort(key=lambda x:x['row'])
 args={'object':d.get('object'),'target':d.get('target'),'arm':arm,'completion':events[-1]['role'] if events else ''}
 if any(args[k] in ('unknown','unspecified','none') for k in ('object','target')):raise ValueError('unbound symbolic argument')
 a={'kind':'primitive','verb':verb,'start':s,'end':e,'arguments':args,'events':events,'quality':'silver','observation':observation,'evidence':{'method':'local_visual_with_measured_events','semantic_verified':False},'review_reasons':['local_auto_label_requires_independent_review','proposal_boundary_requires_review']}
 validate_annotation(a,n)
 relevant=[v for v in evidence['events'] if v['arm']==arm and max(s,v['start'])<min(e,v['end'])]
 if verb=='pick' and not any(v['kind']=='gripper_closing' for v in relevant):raise ValueError('pick lacks measured gripper closing evidence')
 if verb=='place' and not any(v['kind']=='gripper_opening' for v in relevant):raise ValueError('place lacks measured gripper opening evidence')
 if scene:
  if verb in ('press','click','strike','push','pull','insert','handover'):
   raise ValueError('contact-dependent operation requires independent contact/interaction evidence')
  if verb=='place' and args['object'] not in scene['objects']:
   raise ValueError('place object not bound to simulator state')
  if verb=='pick':
   obj=scene['objects'].get(args['object'])
   if obj is None:raise ValueError('pick object not bound to simulator state')
   if not scene['valid'][s] or not scene['valid'][e]:raise ValueError('missing aligned object endpoint')
   if float(obj['xyz'][e,2]-obj['xyz'][s,2])<.015:raise ValueError('pick lacks observed object lift')
  if verb in ('open','close','toggle') and not any(v['kind']=='articulation_change' and args['object'] in (v['evidence']['joint'],v['evidence']['body']) for v in relevant):raise ValueError('articulated operation lacks bound joint change')
 return a

def _subtract_reserved(window, reserved):
 """Return uncovered half-open pieces of a visual window."""
 pieces=[(window['start'],window['end'])]
 for rs,re in sorted(reserved):
  out=[]
  for s,e in pieces:
   if re<=s or e<=rs:out.append((s,e));continue
   if s<rs:out.append((s,rs))
   if re<e:out.append((re,e))
  pieces=out
 return [dict(window,start=s,end=e,reason='visual_gap_after_source_contract')
         for s,e in pieces if s<e]


def annotate(pose,grip,evidence,scene,instruction,decode,model,folder,fps,deadline=None,seed_atoms=None,source='unknown'):
 """Two views of each interval, then at most two split levels; never drops gaps."""
 import time
 try:
  from full_visual_convert import sheets
 except ModuleNotFoundError as exc:
  # Unit tests and deterministic state-only passes do not need HDF5.  Keep the
  # image-sheet helper local so a missing optional decoder dependency cannot
  # turn a valid unresolved receipt into a process crash.
  if exc.name not in ('h5py', 'PIL'):
   raise
  from PIL import Image,ImageDraw
  import io
  def sheets(decoded,rows):
   cams=sorted(decoded)
   out=[]
   for row in rows:
    canvas=Image.new('RGB',(224*max(1,len(cams)),202),'white');draw=ImageDraw.Draw(canvas)
    for j,cam in enumerate(cams):
     im=Image.open(io.BytesIO(decoded[cam][row][0])).convert('RGB');im.thumbnail((224,158));canvas.paste(im,(j*224,22));draw.text((j*224+3,3),f'row {row} | {cam}',fill='black')
    out.append(canvas)
   return out
 from visual_annotation import validate_prediction
 folder.mkdir(parents=True,exist_ok=True)
 seed_atoms=list(seed_atoms or [])
 reserved=[(a['start'],a['end']) for a in seed_atoms]
 pending=[]
 for base in windows(evidence,fps):
  pending.extend(dict(w,depth=0) for w in _subtract_reserved(base,reserved))
 accepted=list(seed_atoms);queue=[];traces=[];observed=set()
 for a in seed_atoms:
  observed.update([a['start'],a['end']]);observed.update(x['row'] for x in a.get('events',[]))
 merge_done=False
 while pending or (queue and not merge_done):
  if not pending:
   # Revisit adjacent unresolved windows together when an operation was cut by
   # a motion boundary. Never merge across an accepted span or between arms.
   merge_done=True;old=sorted(queue,key=lambda x:(x['arm'],x['start']));queue=[];i=0
   while i<len(old):
    a=old[i]
    if i+1<len(old) and old[i+1]['arm']==a['arm'] and old[i+1]['start']==a['end'] and old[i+1]['end']-a['start']<=8*fps:
     pending.append(dict(a,end=old[i+1]['end'],depth=2,reason='merged_adjacent_unresolved'));i+=2
    else:queue.append(a);i+=1
   if not pending:break
  w=pending.pop(0);resolved=None;reasons=[]
  if deadline is not None and time.time()>=deadline:
   queue.extend([dict(x,reason='run_time_budget') for x in [w]+pending]);break
  candidates=[];candidate_reasons=[]
  for attempt in range(2):
   rows=inspection_rows(w,len(pose),evidence,dense=bool(attempt));observed.update(rows);query,tokens=request(w,rows,evidence,instruction,scene);key=f"{w['arm']}_{w['start']}_{w['end']}_{attempt}";qhash=hashlib.sha256(query.encode()).hexdigest();cache=folder/(key+'.json')
   if cache.exists():
    saved=json.loads(cache.read_text())
    if saved['prompt_sha256']!=qhash or saved['model']!=model.path:raise ValueError('local prediction cache mismatch')
    raw=saved['response']
   else:
    images=sheets(decode(rows),rows)
    from PIL import ImageDraw
    for i,img in enumerate(images):ImageDraw.Draw(img).text((2,2),f'F{i}',fill='red')
    raw=model.generate(query,images);write_json(cache,{'prompt':query,'prompt_sha256':qhash,'model':model.path,'rows':rows,'response':raw})
   try:
    # Canonicalize symbolic bindings and event rows against measured source
    # events before validating the model response.  This removes harmless
    # sampled-frame/name variation while preserving the model's operation vote.
    from deterministic_semantics import canonicalize_prediction
    raw_obj=json.loads(raw.strip().removeprefix('```json').removesuffix('```').strip())
    raw= json.dumps(canonicalize_prediction(raw_obj,w,tokens,evidence,instruction,scene,source))
    value=assemble(raw,w,tokens,evidence,len(pose),scene)
    if value['kind']!='unresolved':
     candidates.append(value)
     continue
    reasons.append(value['observation'])
    if 'split_row' in value and w['depth']<2:
     x=value['split_row'];pending[:0]=[dict(w,end=x,depth=w['depth']+1),dict(w,start=x,depth=w['depth']+1)];resolved='split';break
   except (ValueError,KeyError,TypeError) as exc:reasons.append(str(exc))
  if candidates:
   keys=[(x.get('kind'),x.get('verb'),json.dumps(x.get('arguments',{}),sort_keys=True),json.dumps(x.get('events',[]),sort_keys=True)) for x in candidates]
   if len(candidates)>=2 and len(set(keys))==1:
    resolved=candidates[0]
    physical=any(x['arm']==w['arm'] and max(w['start'],x['start'])<min(w['end'],x['end']) and x['kind']=='object_motion' for x in evidence['events'])
    # A visual vote cannot certify acquisition/release without an independent
    # object-motion channel.  Keep the whole interval quarantined when that
    # physical condition is absent.
    if resolved.get('verb') in ('pick','place') and not physical:
     candidate_reasons.append('pick_place_missing_independent_object_motion')
     resolved=None
    else:
     resolved['auto_policy']='atomvla_auto_only_v1';resolved['auto_decision']={'status':'auto_accepted','confidence':0.79,'reason':'two_independent_local_observations_agree','channels':['sparse','dense'],'physical_evidence':physical}
   else:
    candidate_reasons.append('independent_local_observations_disagree_or_missing')
  reasons.extend(candidate_reasons)
  traces.append({'window':w,'reasons':reasons,'resolved':resolved is not None})
  if resolved=='split':continue
  if resolved is None:queue.append(dict(w,reason='; '.join(reasons),next_action='auto_quarantine_or_reprocess'))
  else:accepted.append(resolved)
 doc=validate_prediction({'segments':accepted},len(pose),evidence['physical_arms'],sorted(observed),{'method':'trajectory_local_semantics_v1','model':model.path,'independent_verification':False})
 doc.update(review_queue=queue,local_attempts=traces);return doc
