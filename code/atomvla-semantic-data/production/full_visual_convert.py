"""Resumable, full-episode visual labeling and aligned target materialization.
All included episodes are inventoried; unknown segments and bad sources are retained.
No source commands are synthesized for RoboTwin's observed-state action arrays.
"""
import argparse,collections,hashlib,io,json,os,tarfile,time,traceback,zipfile
from pathlib import Path
from contextlib import contextmanager
try:
 import h5py
except ModuleNotFoundError:
 h5py=None
import numpy as np
from PIL import Image,ImageDraw
from common import write_json,digest,stable_id
from convert_existing import read_instruction,get_images
from convert_archive import tar_bytes
from robotwin_adapter import read_arrays,image_png
from records import make_record
from visual_annotation import VERSION,sample_rows,prompt,parse_json,validate_prediction,compact_prompt,parse_compact


def inventory(legacy,robot):
 for family,pattern in [('libero','libero/*.json'),('droid','lerobot/droid*.json')]:
  for p in sorted((Path(legacy)/'data/full_v3/shards').glob(pattern)):
   d=json.loads(p.read_text())
   for ep in d['episodes']:
    yield {'source':family,'episode_id':ep['episode_id'],'receipt':str(p),'tar':d['tar'],'tar_bytes':d['bytes'],'meta':ep}
 for p in sorted((Path(robot)/'episodes').glob('*/*/annotations.json')):
  d=json.loads(p.read_text());m=d['source'];r=json.loads((p.parent/'receipt.json').read_text())
  yield {'source':'robotwin','episode_id':r['episode_id'],'receipt':str(p.parent/'receipt.json'),'meta':m}

@contextmanager
def read_episode(item):
 source=item['source'];m=dict(item['meta']);uid=item['episode_id']
 if source in ('droid','libero'):
  if h5py is None:raise RuntimeError('h5py is required for source episode decoding')
  if Path(item['tar']).stat().st_size!=item['tar_bytes']:raise ValueError('source shard size mismatch')
  with tarfile.open(item['tar']) as t:raw=t.extractfile(uid+'/trajectory.npz').read()
  with np.load(io.BytesIO(raw),allow_pickle=False) as a:arr={k:a[k] for k in a.files}
  m['source_family']=source;m['_timestamp']=arr['timestamp'];instruction=read_instruction(m)
  meta={'source':source,'task':m['domain'],'robot_variant':source+'_native_single','instruction':instruction,'split':m['split'],'split_group':uid,'split_group_verified':True,'coordinate_frame':source+'_native_parent_eef_xyzw_m','base_frame_calibrated':False,'source_trajectory_sha256':hashlib.sha256(raw).hexdigest(),'source_shard':item['tar'],'source_receipt':item['receipt'],'source_episode_id':uid,'action_representation':m['action_representation'],'action_row_offset':int(m.get('action_row_offset',0)),'physical_replay_verified':False,'arm_slot_mapping':{'left':'native_single_arm','right':'absent'}}
  yield arr['poses_xyzw'],arr['gripper'],arr['state'],arr['action'],bool(m.get('action_supervision_available')),meta,lambda rows:get_images(m,rows)
 else:
  if h5py is None:raise RuntimeError('h5py is required for RoboTwin episode decoding')
  with zipfile.ZipFile(m['archive']) as z:
   if z.getinfo(m['member']).file_size>2*1024**3:raise ValueError('source episode exceeds memory limit')
   with h5py.File(io.BytesIO(z.read(m['member'])),'r') as h:
    p,g,c,schema=read_arrays(h)
    state=h['joint_action/vector'][()] if 'joint_action/vector' in h else np.concatenate([p.reshape(len(p),-1),g],axis=1)
    meta={**m,'source':'robotwin','source_receipt':item['receipt'],'action_representation':'unavailable_measured_states_not_commands','action_row_offset':0,'physical_replay_verified':False}
    def images(rows):return {cam:{row:(image_png(h,ds,row),{'source_dataset':ds,'source_row':row,'archive':m['archive'],'source_member':m['member']}) for row in rows} for cam,ds in c.items()}
    yield p,g,state,None,False,meta,images

def sheets(decoded,rows):
 cams=sorted(decoded,key=lambda c:(not any(k in c for k in ('agentview','head','ext_0')),c))
 # External and wrist/complementary view, both explicitly labeled.
 cams=[cams[0]]+[next((c for c in cams[1:] if any(k in c for k in ('wrist','eye_in_hand'))),cams[1])] if len(cams)>1 else cams;out=[]
 for start in range(0,len(rows),1):
  rr=rows[start:start+1];canvas=Image.new('RGB',(224*len(cams),180*len(rr)),'white');draw=ImageDraw.Draw(canvas)
  for i,row in enumerate(rr):
   for j,cam in enumerate(cams):
    im=Image.open(io.BytesIO(decoded[cam][row][0])).convert('RGB');im.thumbnail((224,158));canvas.paste(im,(j*224,i*180+22));draw.text((j*224+3,i*180+3),f'row {row} | {cam[:23]}',fill='black')
  out.append(canvas)
 return out

class Annotator:
 def __init__(self,path):
  import torch
  from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
  self.torch=torch;self.processor=AutoProcessor.from_pretrained(path,local_files_only=True,min_pixels=256*28*28,max_pixels=1280*28*28)
  self.model=Qwen3VLForConditionalGeneration.from_pretrained(path,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval();self.path=path
 def generate(self,text,images):
  content=[{'type':'text','text':text}]+[{'type':'image','image':x} for x in images]
  messages=[{'role':'user','content':content}]
  inputs=self.processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt').to('cuda')
  with self.torch.inference_mode():output=self.model.generate(**inputs,max_new_tokens=1536,do_sample=False)
  return self.processor.batch_decode(output[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)[0]

def convert(item,root,model,max_frames=40,grounded=False):
 folder=Path(root)/'episodes'/item['source']/item['episode_id'];folder.mkdir(parents=True,exist_ok=True);receipt=folder/'receipt.json'
 identity=stable_id(VERSION,model.path,max_frames,grounded,digest(Path(__file__)),*[digest(Path(__file__).with_name(name)) for name in ['visual_annotation.py','grounded_adapter.py','local_semantics.py','trajectory_events.py','libero_state_evidence.py','deterministic_semantics.py','candidate_policy.py']],item['source'],item['episode_id'],item['receipt'],item['meta'].get('trajectory_content_sha256',item['meta'].get('source_sha256','')))
 if receipt.exists():
  r=json.loads(receipt.read_text())
  if r.get('identity')!=identity:raise ValueError('resume identity mismatch')
  if r.get('tar_sha256') and digest(folder/'sample.tar')!=r['tar_sha256']:raise ValueError('resume artifact corrupted')
  return r
 attempt=folder/'attempt.json'
 if attempt.exists() and json.loads(attempt.read_text()).get('identity')!=identity:
  raise ValueError('prediction cache belongs to different code/model/source; use a new output root')
 write_json(attempt,{'identity':identity,'model':model.path,'max_frames':max_frames})
 started=time.time()
 try:
  with read_episode(item) as (pose,grip,state,action,has_action,meta,decode):
   n=len(pose);arms=['left'] if pose.shape[1]==1 else ['left','right']
   if n<2:raise ValueError('source episode has fewer than two states; no trainable transition')
   if not np.isfinite(pose).all() or not np.isfinite(grip).all():raise ValueError('nonfinite source pose/gripper')
   if grounded:
    from grounded_adapter import evidence_for
    from deterministic_semantics import candidates as deterministic_candidates
    from local_semantics import annotate as local_annotate
    events,scene=evidence_for(item,pose,grip)
    write_json(folder/'motion_evidence.json',events)
    seed_atoms,seed_issues=deterministic_candidates(pose,grip,events,scene,meta['instruction'],item['source'])
    write_json(folder/'deterministic_semantics.json',{
      'source':item['source'],'instruction':meta['instruction'],
      'issues':seed_issues,'candidate_count':len(seed_atoms),
      'candidates':seed_atoms,'method':'source_state_task_contract_v1'})
    doc=local_annotate(pose,grip,events,scene,meta['instruction'],decode,model,folder/'local_predictions',events['fps'],deadline=time.time()+900,seed_atoms=seed_atoms,source=item['source'])
    needed=sorted({a['start'] for a in doc['annotations']}|{e['row'] for a in doc['annotations'] for e in a['events']})
    decoded=decode(needed) if needed else {}
   else:
    rows=sample_rows(pose,grip,max_frames);decoded=decode(rows);images=sheets(decoded,rows)
    for i,img in enumerate(images):img.save(folder/f'contact_sheet_{i:02d}.jpg')
    query=compact_prompt(n,arms,meta['instruction'],rows);(folder/'prompt.txt').write_text(query)
    response=folder/'prediction.txt'
    if response.exists():raw=response.read_text()
    else:raw=model.generate(query,images);response.write_text(raw)
    evidence={'method':'video_language_state_joint_auto_annotation','model':model.path,'sampled_rows':rows,'cameras':list(decoded),'prompt_sha256':hashlib.sha256(query.encode()).hexdigest(),'prediction_sha256':hashlib.sha256(raw.encode()).hexdigest(),'quality':'silver','independent_verification':False}
    try:doc=validate_prediction(parse_compact(raw),n,arms,rows,evidence)
    except (ValueError,KeyError,TypeError) as e:
     # Retain all source rows even when model formatting/semantics fail.
     doc=validate_prediction({'segments':[]},n,arms,rows,evidence);doc['prediction_error']=str(e)
   from candidate_policy import quarantine
   doc=quarantine(doc,n,arms)
   doc.update(episode_id=item['episode_id'],source_frames=n,physical_arms=arms,provenance=meta)
   write_json(folder/'annotations.json',doc)
   pp=np.zeros((n,2,7),np.float32);pp[:,:,6]=1;pp[:,:pose.shape[1]]=pose
   gg=np.zeros((n,2),np.float32);gg[:,:grip.shape[1]]=grip
   records=[]
   for atom in doc['annotations']:
    for t in sorted({atom['start']}|{x['row'] for x in atom['events'][:-1]}):
     rec=make_record(item['episode_id'],atom,pp,gg,meta,t);rec['schema']=VERSION;rec['inputs']['native_state']=state[t].tolist()
     offset=meta['action_row_offset'];ar=np.arange(t,min(t+8,atom['end']))+offset
     available=bool(has_action and action is not None and np.isfinite(action).all())
     if available:
      ar=ar[(ar>=0)&(ar<len(action))];chunk=np.zeros((8,action.shape[1]),np.float32);mask=np.zeros(8,bool);chunk[:len(ar)]=action[ar];mask[:len(ar)]=True
      available=bool(len(ar));rec['targets'].update(actions=chunk.tolist(),action_mask=mask.tolist(),action_source_rows=ar.tolist())
     rec['targets']['action_supervision_available']=available
     rec['targets']['action_contract']='source_native_audited_alignment_not_new_replay' if available else 'unavailable'
     # Quarantine until a separately audited release exists. V1 votes do not prove accuracy.
     auto_ok=False
     rec['train_eligible']=auto_ok;rec['quality']='auto_gold_candidate' if auto_ok else 'auto_unresolved';rec['auto_policy']='atomvla_auto_only_v1';rec['loss_weights']={'program':.5 if auto_ok else 0.,'terminal_pose':1. if auto_ok else 0.,'future_rgb':.5 if auto_ok else 0.,'action':1. if available and auto_ok else 0.};records.append(rec)
   needed=sorted({x['inputs']['current_row'] for x in records}|{e['row'] for x in records for e in x['targets']['keyframes']})
   refs={};tmp=folder/'sample.partial.tar'
   with tarfile.open(tmp,'w') as tar:
    for cam in decoded:
     for row in needed:
      pixels,provenance=decoded[cam][row];member=f'images/{cam}/{row:06d}.png';tar_bytes(tar,member,pixels);refs[cam,row]={'member':member,'sha256':hashlib.sha256(pixels).hexdigest(),**provenance}
    for rec in records:
     rec['inputs']['images']={c:refs[c,rec['inputs']['current_row']] for c in decoded}
     for e in rec['targets']['keyframes']:e['images']={c:refs[c,e['row']] for c in decoded}
    tar_bytes(tar,'records.jsonl',''.join(json.dumps(x,allow_nan=False)+'\n' for x in records).encode());tar_bytes(tar,'episode.json',json.dumps(meta).encode());tar_bytes(tar,'annotations.json',json.dumps(doc).encode())
   os.replace(tmp,folder/'sample.tar')
   r={'identity':identity,'schema':VERSION,'source':item['source'],'episode_id':item['episode_id'],'source_frames':n,'status':'auto_labeled_complete' if doc['automatically_labeled_complete'] else 'partially_labeled','records':len(records),'action_records':sum(x['targets']['action_supervision_available'] for x in records),'atoms':dict(collections.Counter(x['verb'] for x in doc['annotations'])),'verified_complete':False,'full_timeline_accounted':True,**doc['counts'],'tar_sha256':digest(folder/'sample.tar'),'seconds':time.time()-started,'time':time.time()}
   write_json(receipt,r);return r
 except Exception as e:
  r={'identity':identity,'source':item['source'],'episode_id':item['episode_id'],'status':'error','error':str(e),'traceback':traceback.format_exc(),'seconds':time.time()-started,'time':time.time()};write_json(folder/'error.json',r);return r

def main():
 p=argparse.ArgumentParser();p.add_argument('--legacy',required=True);p.add_argument('--robot',required=True);p.add_argument('--root',required=True);p.add_argument('--model',required=True);p.add_argument('--inventory');p.add_argument('--limit',type=int);p.add_argument('--hours',type=float,default=12);p.add_argument('--frames',type=int,default=40);p.add_argument('--grounded',action='store_true');p.add_argument('--source',choices=['droid','libero','robotwin']);p.add_argument('--shard-index',type=int,default=0);p.add_argument('--shard-count',type=int,default=1);a=p.parse_args()
 if a.shard_count<1 or not 0<=a.shard_index<a.shard_count:raise SystemExit('invalid shard index/count')
 root=Path(a.root);root.mkdir(parents=True,exist_ok=True)
 from torch.utils.tensorboard import SummaryWriter
 w=SummaryWriter(str(Path(a.legacy)/'tensorboard/semantic_full_visual'/root.parent.name/root.name));began=time.time();counts={s:collections.Counter() for s in ['droid','libero','robotwin']}
 # Persistent full inventory independent of any diagnostic slice.
 items=[json.loads(line) for line in Path(a.inventory).read_text().splitlines() if line.strip()] if a.inventory else list(inventory(a.legacy,a.robot));write_json(root/'inventory_summary.json',{'counts':dict(collections.Counter(x['source'] for x in items)),'excluded':['geniesim'],'total':len(items),'shard_count':a.shard_count,'shard_index':a.shard_index})
 with (root/'inventory.jsonl').open('w') as f:
  for x in items:f.write(json.dumps({k:v for k,v in x.items() if k!='meta'})+'\n')
 if a.source:
  items=[x for x in items if x['source']==a.source]
 if a.shard_count>1:
  items=[x for i,x in enumerate(items) if i % a.shard_count == a.shard_index]
 if a.limit:
  groups={s:[x for x in items if x['source']==s] for s in counts};chosen=[]
  # Spread LIBERO across distinct tasks; RoboTwin across task names; DROID across shards.
  for s,g in groups.items():
   seen=set()
   for x in g:
    key=x['meta'].get('task',x['receipt'])
    if key in seen:continue
    seen.add(key);chosen.append(x)
    if len(seen)>=a.limit:break
  items=chosen
 def progress(phase,current=None):
  d={'phase':phase,'counts':{s:dict(v) for s,v in counts.items()},'planned_episodes':len(items),'elapsed_seconds':time.time()-began,'current':current,'full_dataset_ready':False,'updated_at':time.time()};write_json(root/'progress.json',d)
  step=sum(c['processed'] for c in counts.values())
  for s,c in counts.items():
   for k,v in c.items():w.add_scalar(s+'/'+k,v,step)
  w.add_scalar('progress/planned_episodes',len(items),step);w.add_scalar('progress/processed',step,step);w.add_scalar('progress/fraction',step/max(1,len(items)),step);w.add_scalar('training/waiting',1,step);w.flush();return d
 progress('loading_model');model=Annotator(a.model)
 for item in items:
  if time.time()-began>a.hours*3600:break
  progress('annotating',{'source':item['source'],'episode_id':item['episode_id']});r=convert(item,root,model,a.frames,grounded=a.grounded);c=counts[item['source']];c['processed']+=1;c[r['status']]+=1
  for k in ('records','action_records','total_arm_transitions','resolved_candidate_transitions','unresolved_transitions'):c[k]+=r.get(k,0)
  for k,v in r.get('atoms',{}).items():c['atom_'+k]+=v
  progress('annotating')
 d=progress('finished_bounded_run');write_json(root/'run_result.json',d);w.close();print(json.dumps(d))
if __name__=='__main__':main()
