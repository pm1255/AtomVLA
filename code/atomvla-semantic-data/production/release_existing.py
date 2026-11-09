"""Audit and publish weak-event existing-source records with native actions."""
import argparse,collections,hashlib,json,os,tarfile,time
from pathlib import Path
import numpy as np
from common import digest,write_json
from records import validate_annotation
from loader import load_episode,collate

def publish(root):
 root=Path(root);out=root/'release';tmp=root/'release.partial'
 if out.exists():raise ValueError('immutable release already exists')
 tmp.mkdir(exist_ok=False);counts=collections.Counter();errors=[];groups={};seen=set();files={s:(tmp/(s+'.jsonl')).open('w') for s in ['train','validation','test']}
 for path in sorted((root/'episodes').glob('*/*/receipt.json')):
  d=json.loads(path.read_text());counts['scanned_episodes']+=1
  if not d.get('records'):counts['needs_annotation']+=1;continue
  try:
   tarpath=path.parent/'sample.tar'
   if digest(tarpath)!=d['tar_sha256']:raise ValueError('tar hash mismatch')
   with tarfile.open(tarpath) as tar:
    records=[json.loads(x) for x in tar.extractfile('records.jsonl')]
    for r in records:
     if r['provenance']['source'] not in ('droid','libero'):raise ValueError('excluded source')
     validate_annotation(r['annotation'],r['targets']['terminal_row']+1)
     if r['id'] in seen:raise ValueError('duplicate record id')
     if groups.get(r['split_group'],r['split'])!=r['split']:raise ValueError('split leak')
     for frame in [r['inputs']['images']]+[e['images'] for e in r['targets']['keyframes']]:
      for ref in frame.values():
       if hashlib.sha256(tar.extractfile(ref['member']).read()).hexdigest()!=ref['sha256']:raise ValueError('image hash')
     tg=r['targets'];ac=np.asarray(tg['actions']);mask=np.asarray(tg['action_mask'],bool)
     if ac.ndim!=2 or ac.shape[0]!=8 or mask.shape!=(8,) or not np.isfinite(ac).all():raise ValueError('action dimensions')
     expected=list(range(r['inputs']['current_row']+r['provenance']['action_row_offset'],r['inputs']['current_row']+r['provenance']['action_row_offset']+int(mask.sum())))
     if tg['action_supervision_available'] and tg['action_source_rows']!=expected:raise ValueError('action alignment')
    samples=list(load_episode(tarpath,allow_silver=True));collate(samples[:1])
    for r in records:
     seen.add(r['id']);groups[r['split_group']]=r['split'];entry={'tar':str(tarpath),'tar_sha256':d['tar_sha256'],'id':r['id'],'episode_id':r['episode_id'],'split':r['split'],'policy':'stage1_multisource_weak_events_v1','annotation_quality':'silver','source':r['provenance']['source'],
       'loss_weights':{'program':.5,'terminal_pose':1.,'future_rgb':.5,'action':1. if r['targets']['action_supervision_available'] else 0.},'robot_variant':r['robot_variant'],'coordinate_frame':r['coordinate_frame']}
     files[r['split']].write(json.dumps(entry)+'\n');counts['records']+=1;counts[r['split']+'_records']+=1;counts['action_records']+=r['targets']['action_supervision_available']
    counts['converted_episodes']+=1
  except Exception as e:errors.append({'path':str(path),'error':str(e)})
 for f in files.values():f.close()
 report={'counts':dict(counts),'errors':errors,'policy':'stage1_multisource_weak_events_v1','physical_semantics_verified':False,'full_joint_training_ready':False,'time':time.time(),'limitations':['native action contract inherited from audited legacy source; not newly replayed','language object references not scene-verified','only conservative single-object pick/place automatic proposals','native parent coordinate frames; no cross-robot calibration']}
 write_json(tmp/'report.json',report);os.replace(tmp,out);return report
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();print(json.dumps(publish(a.root)))
