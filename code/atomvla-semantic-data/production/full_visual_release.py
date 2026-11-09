"""Reconcile every inventory entry and validate target artifacts before publication.
A partial corpus can be inspected, but never receives a full-dataset-ready claim.
"""
import argparse,collections,hashlib,io,json,tarfile,time
from pathlib import Path
import numpy as np
from common import write_json,digest
from visual_annotation import VERSION

def validate_tar(path):
 with tarfile.open(path) as t:
  names=set(t.getnames());records=[json.loads(x) for x in t.extractfile('records.jsonl').read().splitlines()];doc=json.load(t.extractfile('annotations.json'));meta=json.load(t.extractfile('episode.json'))
  for r in records:
   if r['schema']!=VERSION:raise ValueError('tier/schema mismatch')
   if r['train_eligible'] and r.get('auto_policy')!='atomvla_auto_only_v1':raise ValueError('train record missing automatic policy')
   a=r['annotation'];target=r['targets'];current=r['inputs']['current_row']
   if not a['start']<=current<a['end'] or target['terminal_row']!=a['end']:raise ValueError('time alignment')
   pose=np.asarray(target['terminal_pose_xyzw']);mask=np.asarray(target['arm_mask'],bool)
   if pose.shape!=(2,7) or not mask.any() or not np.isfinite(pose).all() or not np.allclose(np.linalg.norm(pose[mask,3:],axis=1),1,atol=.015):raise ValueError('endpoint pose')
   if np.any(pose[~mask]):raise ValueError('inactive arm target')
   frames=[(current,r['inputs']['images'])]+[(ev['row'],ev['images']) for ev in target['keyframes']]
   for row,cameras in frames:
    if not cameras:raise ValueError('missing images')
    for ref in cameras.values():
     if ref['source_row']!=row or ref['member'] not in names:raise ValueError('image row/ref mismatch')
     if hashlib.sha256(t.extractfile(ref['member']).read()).hexdigest()!=ref['sha256']:raise ValueError('image integrity')
   if target['action_supervision_available']:
    ar=target['action_source_rows'];am=np.asarray(target['action_mask'],bool);actions=np.asarray(target['actions']);expected=list(range(current+meta['action_row_offset'],current+meta['action_row_offset']+len(ar)))
    if ar!=expected or len(ar)>min(8,a['end']-current) or am.sum()!=len(ar) or not am[:len(ar)].all() or am[len(ar):].any() or not np.isfinite(actions).all():raise ValueError('action alignment')
   elif r['loss_weights']['action']!=0:raise ValueError('missing action target used')
  return records,doc

def publish(root):
 root=Path(root);out=root/'release';out.mkdir(exist_ok=True);counts=collections.defaultdict(collections.Counter);errors=[];groups={};paths={}
 inventory=[json.loads(x) for x in (root/'inventory.jsonl').read_text().splitlines()]
 for split in ['train','validation','test']:paths[split]=(out/(split+'.jsonl')).open('w')
 try:
  for item in inventory:
   source,uid=item['source'],item['episode_id'];c=counts[source];c['inventory_episodes']+=1;folder=root/'episodes'/source/uid;p=folder/'receipt.json'
   if not p.exists():c['error_episodes' if (folder/'error.json').exists() else 'not_processed']+=1;continue
   try:
    receipt=json.loads(p.read_text());tar=folder/'sample.tar'
    if digest(tar)!=receipt['tar_sha256']:raise ValueError('tar SHA mismatch')
    records,doc=validate_tar(tar)
    if len(records)!=receipt['records']:raise ValueError('record count mismatch')
    c['converted_episodes']+=1;c['automatically_labeled_complete']+=int(doc['automatically_labeled_complete']);c['unresolved_transitions']+=doc['counts']['unresolved_transitions']
    for i,r in enumerate(records):
     split={'val':'validation'}.get(r['split'],r['split']);group=source+':'+r['split_group']
     if split not in paths:raise ValueError('unknown split')
     if group in groups and groups[group]!=split:raise ValueError('split leakage')
     groups[group]=split
     row={'id':r['id'],'episode_id':uid,'source':source,'split':split,'split_group':group,'tar':str(tar),'tar_sha256':receipt['tar_sha256'],'record_index':i,'quality':'silver','policy':'stage1_visual_candidates_v1','full_episode_auto_labeled':doc['automatically_labeled_complete'],'verified_complete':False}
     paths[split].write(json.dumps(row)+'\n');c['records']+=1
   except Exception as exc:errors.append({'source':source,'episode_id':uid,'error':str(exc)});c['artifact_errors']+=1
 finally:
  for f in paths.values():f.close()
 full=not errors and all(c['inventory_episodes']==c['automatically_labeled_complete'] for c in counts.values())
 report={'schema':VERSION,'counts':{s:dict(c) for s,c in counts.items()},'errors':errors,'all_episodes_automatically_labeled':full,'all_episodes_independently_verified':False,'full_joint_training_ready':False,'policy':'stage1_visual_candidates_v1','note':'These are automatically proposed labels, not verified ground truth. Incomplete episodes remain in the source inventory; candidate manifests do not establish full completion.','time':time.time()}
 write_json(out/'report.json',report);return report
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();print(json.dumps(publish(a.root)))
