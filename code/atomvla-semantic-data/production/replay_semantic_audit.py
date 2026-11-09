"""Automatic audit using compiled MuJoCo contacts, independent of VLM votes.
No rendering or simulated action rollout. Restores saved states with mj_forward.
Reports necessary-condition conformance, NOT full semantic ground-truth accuracy.
"""
import argparse, collections, hashlib, json, re, time, traceback
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

PROTOCOL = {
 'version':'saved_state_contact_audit_v1', 'selection':'demo_25 from each of 130 sorted LIBERO task shards',
 'minimum_lift_m':.015, 'consecutive_frames':3, 'boundary_tolerance_frames':4,
 'max_eef_reconstruction_error_m':.001, 'contact_margin_m':.001, 'support_normal_abs_z':.5, 'post_release_span_m':.01,
 'truth_scope':'compiled-contact/state necessary conditions; goal compatibility is not observed execution',
 'full_semantic_accuracy':None, 'no_human_review':True,
 'repair_evaluation':'same physical evidence reused; before/after is conformance, not independent generalization accuracy'
}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False));tmp.replace(p)
def sexpr(text):
 tokens=re.findall(r'\(|\)|[^\s()]+',re.sub(r';[^\n]*','',text));stack=[[]]
 for t in tokens:
  if t=='(':v=[];stack[-1].append(v);stack.append(v)
  elif t==')':
   if len(stack)<2:raise ValueError('unbalanced BDDL')
   stack.pop()
  else:stack[-1].append(t)
 if len(stack)!=1:raise ValueError('unbalanced BDDL')
 return stack[0][0]
def contract(path):
 tree=sexpr(Path(path).read_text());sections={v[0]:v[1:] for v in tree if isinstance(v,list) and v}
 objects=[]
 for section in (':objects',':fixtures'):
  vals=sections.get(section,[]);group=[];i=0
  while i<len(vals):
   if vals[i]=='-':objects.extend(group);group=[];i+=2
   else:group.append(vals[i]);i+=1
  objects.extend(group)
 regions={}
 for r in sections.get(':regions',[]):
  if isinstance(r,list):
   target=next((x[1] for x in r[1:] if isinstance(x,list) and x[0]==':target'),None)
   if target:regions[target+'_'+r[0]]=target
 def leaves(n):
  if not isinstance(n,list) or not n:return []
  if n[0].lower()=='and':return [v for x in n[1:] for v in leaves(x)]
  return [n]
 goals=[v for x in sections.get(':goal',[]) for v in leaves(x)]
 return {'objects':objects,'regions':regions,'goals':goals,'sha256':sha(path),'path':str(path),'language':' '.join(sections.get(':language',[]))}
def entity(name,c):
 if name in c['regions']:return c['regions'][name]
 matches=[x for x in c['objects'] if name==x or name.startswith(x+'_')]
 return max(matches,key=len) if matches else None

def restore(meta,bddl,assets,robosuite_assets):
 import h5py,mujoco
 from libero_state_evidence import extract
 with h5py.File(meta['source_hdf5']) as f:
  g=f['data'][meta['source_group']];states=g['states'][()];joint=g['obs/joint_states'][()];grip=g['obs/gripper_states'][()];eef=g['obs/ee_pos'][()]
 scene=extract(meta['model_file'],states,joint,grip)
 root=ET.fromstring(meta['model_file']);mapping=[]
 for a in root.findall('.//*[@file]'):
  old=a.get('file')
  if '/chiliocosm/assets/' in old:new=Path(assets)/old.split('/chiliocosm/assets/',1)[1]
  elif '/robosuite/' in old:new=Path(robosuite_assets)/old.rsplit('/robosuite/',1)[1]
  else:new=Path(old)
  if not new.is_file():raise FileNotFoundError(str(new))
  a.set('file',str(new));mapping.append({'old':old,'path':str(new),'sha256':sha(new)})
 model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'));data=mujoco.MjData(model)
 if states.shape[1]!=1+model.nq+model.nv:raise ValueError('compiled state dimensions mismatch')
 # Independent compiled qpos layout must agree with the XML decoder.
 for j in scene['joints']:
  ji=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,j['name'])
  if ji<0 or model.jnt_qposadr[ji]!=j['qpos_start']:raise ValueError('compiled joint layout mismatch')
 c=contract(bddl);n=len(states);objids={k:entity(k,c) for k in scene['objects']};xyz={k:v['xyz'] for k,v in scene['objects'].items()}
 held={k:np.zeros(n,bool) for k in xyz};supports={k:[set() for _ in range(n)] for k in xyz};valid=scene['valid'];off=scene['alignment']['state_row_offset'];eef_errors=[]
 geom_names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i) or '' for i in range(model.ngeom)]
 geom_entities=[entity(v,c) for v in geom_names];by_entity={v:k for k,v in objids.items() if v}
 # Kitchen table geometry uses generic table_collision in the source XML.
 tables=[x for x in c['objects'] if 'table' in x]
 for i,name in enumerate(geom_names):
  if name.startswith('table_') and len(tables)==1:geom_entities[i]=tables[0]
 sid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,'gripper0_grip_site')
 for t in np.flatnonzero(valid):
  s=states[t+off];data.time=s[0];data.qpos[:]=s[1:1+model.nq];data.qvel[:]=s[1+model.nq:];mujoco.mj_forward(model,data)
  if sid>=0:eef_errors.append(float(np.linalg.norm(data.site_xpos[sid]-eef[t])))
  fingers={k:set() for k in xyz}
  for ct in data.contact:
   if ct.dist>PROTOCOL['contact_margin_m']:continue
   for oi,other in ((ct.geom1,ct.geom2),(ct.geom2,ct.geom1)):
    ob=by_entity.get(geom_entities[oi])
    if ob is None:continue
    name=geom_names[other]
    if 'finger1' in name:fingers[ob].add(1)
    if 'finger2' in name:fingers[ob].add(2)
    target=geom_entities[other]
    if target and target!=objids[ob] and abs(ct.frame[2])>=PROTOCOL['support_normal_abs_z'] and ct.pos[2]<=xyz[ob][t,2]+.01:supports[ob][t].add(target)
  for ob in xyz:held[ob][t]=fingers[ob]=={1,2}
 if sid<0 or not eef_errors or max(eef_errors)>PROTOCOL['max_eef_reconstruction_error_m']:raise ValueError('compiled EEF mismatch: '+str(max(eef_errors,default=None)))
 return {'contract':c,'scene':scene,'xyz':xyz,'held':held,'supports':supports,'valid':valid,'object_entities':objids,'n':n,'grip':abs(grip).sum(1),
         'provenance':{'source_hdf5':meta['source_hdf5'],'source_group':meta['source_group'],'states_sha256':hashlib.sha256(states.tobytes()).hexdigest(),'xml_sha256':hashlib.sha256(meta['model_file'].encode()).hexdigest(),'bddl_sha256':c['sha256'],'asset_mapping':mapping,'state_alignment':scene['alignment'],'max_eef_error_m':max(eef_errors),'mujoco_version':mujoco.__version__}}

def first_run(mask,lo,hi,k=3):
 for t in range(lo,hi-k+2):
  if np.all(mask[t:t+k]):return t
 return None

def check_atom(a,r):
 verb=a['verb'];s,e=a['start'],a['end'];args=a['arguments'];events={v['role']:v['row'] for v in a['events']};ob=args['object'];c=r['contract'];fails=[];missing=[];checks={}
 if not 0<=s<e<r['n']:return {'status':'contradicted','failures':['invalid_bounds'],'checks':{}}
 if verb not in ('pick','place'):return {'status':'unverified','missing':['verb_contract_not_implemented'],'checks':{}}
 if ob not in r['xyz']:return {'status':'unverified','missing':['object_state_not_bound'],'checks':{}}
 ent=r['object_entities'][ob];end_valid=bool(r['valid'][e]);held=r['held'][ob];sup=r['supports'][ob];z=r['xyz'][ob][:,2]
 def check(name,value):
  checks[name]=bool(value)
  if not value:fails.append(name)
 if verb=='pick':
  close=events.get('closed');lift=events.get('lifted')
  if close is None or lift is None:missing.append('event_schema_missing')
  else:
   if r['valid'][close]:check('two_finger_contact_near_closed',bool(held[max(s+1,close-2):min(e+1,close+3)].any()))
   else:missing.append('closure_state_unavailable')
   if end_valid:
    check('held_at_lift_endpoint',held[e]);check('no_support_at_lift_endpoint',not sup[e]);check('vertical_lift',z[e]-z[s]>=.015)
   else:missing.append('endpoint_state_unavailable')
   mask=held & r['valid'] & (z-z[s]>=.015) & np.array([not x for x in sup]);boundary=first_run(mask,max(s+1,close),min(e+2,r['n']-1))
   checks['reference_lift_row']=boundary
   if boundary is None:missing.append('stable_acquisition_boundary_unavailable')
   else:
    checks['boundary_error_frames']=int(abs(lift-boundary));check('lift_boundary_within_tolerance',abs(lift-boundary)<=PROTOCOL['boundary_tolerance_frames'])
 else:
  target=entity(args['target'],c)
  check('nonself_target',target!=ent and args['target']!=ob)
  goals=[x for x in c['goals'] if len(x)==3 and x[0].lower() in ('on','in') and x[1]==ent]
  checks['goal_compatible']=bool(target and any(entity(x[2],c)==target for x in goals)) if goals else None
  if target is None:missing.append('target_not_bound_to_scene')
  sr=events.get('supported')
  if sr is None:missing.append('support_event_missing')
  elif r['valid'][sr] and target:check('target_support_at_keyframe',target in sup[sr])
  elif not r['valid'][sr]:missing.append('support_state_unavailable')
  if end_valid:
   check('not_pinched_at_release',not held[e])
   if target:check('target_support_at_release',target in sup[e])
  else:missing.append('endpoint_state_unavailable')
  check('gripper_opened',r['grip'][e]-r['grip'][s]>=.006)
  if e+2>=r['n'] or not r['valid'][e:e+3].all():missing.append('post_release_observation_unavailable')
  elif target:
   check('support_persists_after_release',all(target in x for x in sup[e:e+3]));check('post_release_position_stable',np.linalg.norm(np.ptp(r['xyz'][ob][e:e+3],axis=0))<=.01)
 return {'status':'contradicted' if fails else ('unverified' if missing else 'supported'), 'checks':checks,'failures':fails,'missing':missing,
         'qualification':'necessary_conditions_only_not_full_semantic_truth'}

def repair(a,r):
 """Generate conservative revision candidates; never promote to train eligible."""
 import copy
 out=copy.deepcopy(a);args=out['arguments'];ob=args['object'];s,e=a['start'],a['end'];log=[]
 out['train_eligible']=False;out['auto_decision']={'status':'auto_unresolved','reason':'requires_independent_validation'};out['quality']='silver'
 if ob not in r['xyz']:return out,['object_not_bound']
 if a['verb']=='place':
  # Source observed contact selects target; BDDL is only a compatibility check.
  counts=collections.Counter(x for row in r['supports'][ob][s:e+1] for x in row)
  if counts:
   ranked=counts.most_common()
   if len(ranked)==1 or ranked[0][1]>ranked[1][1]:
    target=ranked[0][0];args['target']=target;log.append('target_from_compiled_contact')
    for t in range(s+1,min(e+1,r['n']-2)):
     if r['valid'][t:t+3].all() and not r['held'][ob][t:t+3].any() and all(target in x for x in r['supports'][ob][t:t+3]) and r['grip'][t]-r['grip'][s]>=.006:
      out['end']=t;out['events']=[{'role':'released','row':t}]
      support=next((j for j in range(s+1,t) if target in r['supports'][ob][j]),None)
      if support is not None:out['events'].insert(0,{'role':'supported','row':support})
      log.append('release_from_contact_and_aperture');break
 elif a['verb']=='pick':
  close=next((x['row'] for x in a['events'] if x['role']=='closed'),s+1);z=r['xyz'][ob][:,2]
  mask=r['held'][ob]&r['valid']&(z-z[s]>=.015)&np.array([not x for x in r['supports'][ob]])
  t=first_run(mask,max(s+1,close+1),e)
  if t is not None:
   out['end']=t;out['events']=[x for x in a['events'] if x['role']!='lifted' and x['row']<t]+[{'role':'lifted','row':t}];log.append('lift_boundary_from_stable_acquisition')
 return out,log

def main():
 p=argparse.ArgumentParser();p.add_argument('--legacy',required=True);p.add_argument('--baseline',required=True);p.add_argument('--bddl-root',required=True);p.add_argument('--assets',required=True);p.add_argument('--robosuite-root',required=True);p.add_argument('--root',required=True);p.add_argument('--demo',type=int,default=25);p.add_argument('--limit',type=int);p.add_argument('--tb-root',required=True);a=p.parse_args();root=Path(a.root);root.mkdir(parents=True,exist_ok=True)
 from tensorboard.summary.writer.event_file_writer import EventFileWriter
 from tensorboard.compat.proto.event_pb2 import Event
 from tensorboard.compat.proto.summary_pb2 import Summary
 class SummaryWriter:
  def __init__(self,path):self.writer=EventFileWriter(path)
  def add_scalar(self,tag,value,step):self.writer.add_event(Event(wall_time=time.time(),step=step,summary=Summary(value=[Summary.Value(tag=tag,simple_value=value)])))
  def flush(self):self.writer.flush()
  def close(self):self.writer.close()
 w=SummaryWriter(a.tb_root);files=sorted((Path(a.legacy)/'data/full_v3/shards/libero').glob('*.json'));files=files[:a.limit] if a.limit else files
 save(root/'protocol.json',{**PROTOCOL,'demo':a.demo,'tasks':len(files),'code_sha256':sha(__file__),'started':time.time()});rows=[];errors=[];counts=collections.Counter();began=time.time();tot_trans=0;old_cov=0;new_cov=0
 for fi,pth in enumerate(files):
  m=next((v for v in json.loads(pth.read_text())['episodes'] if v['source_group']==f'demo_{a.demo}'),None)
  try:
   if m is None:raise ValueError('selected demo missing')
   uid=m['episode_id'];preds=list(Path(a.baseline).glob('shard_*/episodes/libero/'+uid+'/annotations.json'))
   if len(preds)!=1:raise ValueError('baseline missing or duplicate')
   doc=json.loads(preds[0].read_text());name=m['source_attributes']['bddl_file_name'].split('bddl_files/')[-1];bddl=Path(a.bddl_root)/name
   r=restore(m,bddl,a.assets,a.robosuite_root);n=r['n'];oldmask=np.zeros(n-1,bool);newmask=np.zeros(n-1,bool);details=[]
   for ai,atom in enumerate(doc['annotations']):
    before=check_atom(atom,r);fixed,changes=repair(atom,r);after=check_atom(fixed,r)
    counts['atoms']+=1;counts['before_'+before['status']]+=1;counts['after_'+after['status']]+=1;counts['verb_'+atom['verb']]+=1
    if before['status']=='supported':oldmask[atom['start']:atom['end']]=True
    if after['status']=='supported':newmask[fixed['start']:fixed['end']]=True
    details.append({'atom_index':ai,'original':atom,'before':before,'revision_candidate':fixed,'changes':changes,'after':after})
   counts['episodes']+=1;tot_trans+=n-1;old_cov+=int(oldmask.sum());new_cov+=int(newmask.sum())
   entry={'episode_id':uid,'task':name,'baseline_sha256':sha(preds[0]),'provenance':r['provenance'],'contract':r['contract'],'atoms':details,'source_transitions':n-1,'supported_transitions_before':int(oldmask.sum()),'supported_transitions_after':int(newmask.sum())}
   save(root/'episodes'/f'{uid}.json',entry);rows.append({'episode_id':uid,'atoms':len(details)})
  except Exception as exc:errors.append({'task_file':str(pth),'episode_id':m.get('episode_id') if m else None,'error':str(exc),'traceback':traceback.format_exc()})
  report={'counts':dict(counts),'errors':errors,'planned_tasks':len(files),'processed_tasks':fi+1,'source_transitions':tot_trans,'supported_transitions_before':old_cov,'supported_transitions_after':new_cov,'full_semantic_accuracy':None,'interpretation':'necessary-condition conformance, not semantic accuracy','training_ready':False,'elapsed_seconds':time.time()-began,'time':time.time()};save(root/'progress.json',report)
  for k,v in counts.items():w.add_scalar('audit/'+k,v,fi+1)
  w.add_scalar('progress/fraction',(fi+1)/len(files),fi+1);w.add_scalar('progress/errors',len(errors),fi+1);w.add_scalar('training/waiting',1,fi+1);w.flush()
 save(root/'report.json',report);w.close();print(json.dumps(report))
if __name__=='__main__':main()
