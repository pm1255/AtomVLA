"""Inventory every existing episode, including those producing zero candidates.
Writes an explicit per-arm gap queue; never fabricates labels to fill gaps.
"""
import argparse,collections,json,os,time
from pathlib import Path
from common import write_json
from complete_annotation import audit,EVENT_SCHEMAS
from torch.utils.tensorboard import SummaryWriter

def main():
 p=argparse.ArgumentParser();p.add_argument('--legacy',required=True);p.add_argument('--multisource',required=True);p.add_argument('--robotwin',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 old=Path(a.legacy);multi=Path(a.multisource);robot=Path(a.robotwin);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
 writer=SummaryWriter(str(old/'tensorboard/semantic_complete_annotation'/out.name));counts={s:collections.Counter() for s in ('droid','libero','robotwin')};start=time.time()
 write_json(out/'contract.json',{'sources':list(counts),'excluded':['geniesim'],'event_schemas':EVENT_SCHEMAS,'requirements':['every available source transition on every arm accounted for','no overlapping primitive intervals on same arm','no unresolved intervals','event/binding/boundary evidence','query and event image targets aligned'],'not_an_annotation_generator':True})
 queue=(out/'unresolved_intervals.jsonl').open('w');details=(out/'episode_coverage.jsonl').open('w');index=(out/'fully_annotated_episodes.jsonl').open('w')
 def update():
  state={'counts':{s:dict(c) for s,c in counts.items()},'elapsed_seconds':time.time()-start,'updated_at':time.time(),'full_dataset_ready':False}
  write_json(out/'progress.json',state);step=int(time.time()-start)
  for s,c in counts.items():
   for k,v in c.items():writer.add_scalar(s+'/'+k,v,step)
  writer.add_scalar('training/full_joint_waiting',1,step);writer.flush()
 def one(source,uid,n,arms,annpath,locator):
  c=counts[source];c['source_episodes']+=1
  try:
   if annpath.exists():
    doc=json.loads(annpath.read_text());annotations=doc.get('annotations',[]);non_action=doc.get('non_action',[]);c['candidate_processing_attempted']+=1
   else:annotations=[];non_action=[];c['not_yet_processed']+=1
   result=audit(n,arms,annotations,non_action);result.update(source=source,episode_id=uid,source_locator=locator,annotation_path=str(annpath))
   details.write(json.dumps(result)+'\n')
   if annotations:c['episodes_with_candidates']+=1
   c['candidate_atoms']+=sum(result['candidate_atom_counts'].values())
   c['total_transitions']+=sum(v['total_transitions'] for v in result['arms'].values())
   c['candidate_covered_transitions']+=sum(v['candidate_covered_transitions'] for v in result['arms'].values())
   c['unannotated_intervals']+=sum(len(v['unannotated_intervals']) for v in result['arms'].values())
   c['episodes_with_gaps']+=int(any(v['unannotated_intervals'] for v in result['arms'].values()))
   c['episodes_with_overlaps']+=int(any(v['overlap_intervals'] for v in result['arms'].values()))
   if result['complete_annotation']:
    c['fully_annotated_episodes']+=1;index.write(json.dumps({'source':source,'episode_id':uid,'annotation_path':str(annpath)})+'\n')
   else:
    c['incomplete_episodes']+=1;queue.write(json.dumps(result)+'\n')
  except Exception as exc:
   c['audit_errors']+=1;queue.write(json.dumps({'source':source,'episode_id':uid,'error':str(exc),'source_locator':locator})+'\n')
  if sum(x['source_episodes'] for x in counts.values())%500==0:update()
 try:
  for source,pattern in [('droid','lerobot/droid*.json'),('libero','libero/*.json')]:
   for p in sorted((old/'data/full_v3/shards').glob(pattern)):
    d=json.loads(p.read_text())
    for ep in d['episodes']:
     uid=ep['episode_id'];one(source,uid,ep['frames'],['left'],multi/source/'episodes'/source/uid/'annotations.json',{'legacy_receipt':str(p),'source_shard':d['tar'],'native_arm_mapping':'left_slot_is_native_single_arm','images':ep.get('source_image_locators')})
  for p in sorted((robot/'episodes').glob('*/*/receipt.json')):
   d=json.loads(p.read_text());uid=d['episode_id'];one('robotwin',uid,d['source_frames'],['left','right'],p.parent/'annotations.json',{'receipt':str(p),'source_sha256':d['source_sha256'],'member':d['member']})
 finally:
  for f in [queue,details,index]:f.flush();os.fsync(f.fileno());f.close()
  update();writer.close()
 report={'counts':{s:dict(c) for s,c in counts.items()},'seconds':time.time()-start,'full_dataset_ready':False,'status':'coverage_audit_completed_not_annotation_completed','time':time.time(),'snapshot_note':'conversion may continue; missing artifacts at audit time are counted as not_yet_processed'}
 write_json(out/'report.json',report);print(json.dumps(report))
if __name__=='__main__':main()
