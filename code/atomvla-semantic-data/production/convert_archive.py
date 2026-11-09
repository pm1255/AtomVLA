"""Atomic per-episode outputs, complete source accounting, immutable gold/silver tiers."""
import argparse,fcntl,hashlib,io,json,os,tarfile,time,traceback,zipfile
from pathlib import Path
from common import VERSION,digest,write_json,stable_id,split_for
from robotwin_adapter import read_arrays,image_png,metadata
from task_schema import infer_schema
from annotate import proposals
from records import make_record,validate_annotation

def tar_bytes(tar,name,data):
 item=tarfile.TarInfo(name);item.size=len(data);item.mtime=0;tar.addfile(item,io.BytesIO(data))

def run(archive,relative_path,output,sources,source_hash,review_root=None,progress=None):
 import h5py
 output=Path(output);task=relative_path.split('/')[1];variant=Path(relative_path).stem
 task_source=Path(sources)/(task+'.py');schema=infer_schema(task_source) if task_source.exists() else {}
 with zipfile.ZipFile(archive) as z:
  names=sorted(n for n in z.namelist() if n.endswith('.hdf5') and '/data/' in n)
  outcome={'archive':str(archive),'relative_path':relative_path,'source_sha256':source_hash,'episodes':0,'silver_records':0,'gold_records':0,'review_episodes':0,'failed':0,'no_hdf5':not bool(names)}
  for member in names:
   uid=stable_id('robotwin',source_hash,member);folder=output/'episodes'/task/uid;receipt=folder/'receipt.json'
   if receipt.exists():
    r=json.loads(receipt.read_text())
    if r['source_sha256']!=source_hash:raise RuntimeError('immutable identity conflict')
    if r.get('tar_sha256') and (not (folder/'sample.tar').exists() or digest(folder/'sample.tar')!=r['tar_sha256']):raise RuntimeError('previous artifact corrupted')
   else:
    folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'write.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    if receipt.exists():r=json.loads(receipt.read_text());lock.close()
    else:
     try:
      if z.getinfo(member).file_size>2*1024**3:raise ValueError('episode exceeds 2 GiB decompression budget')
      with h5py.File(io.BytesIO(z.read(member)),'r') as h:
       pose,grip,cameras,source_schema=read_arrays(h);eid,scene,seed,instruction=metadata(z,member,h)
       split,group=split_for(task,seed,uid)
       meta={'source':'RoboTwin2.0','source_sha256':source_hash,'archive':str(archive),'member':member,
             'task':task,'robot_variant':variant,'seed':seed,'episode_number':eid,'instruction':instruction,
             'split':split,'split_group':group,'split_group_verified':seed is not None,
             'source_schema':source_schema,'coordinate_frame':'robotwin_native_world_eef_xyzw_m',
             'base_frame_calibrated':False,'source_task_sha256':schema.get('source_sha256'),
             'source_task_revision':'ea8b21121ebb3cd201ff5b3fe361944ac94eda3f',
             'source_task_matches_collector_version':False,'scene_info':scene}
       annotations,issues=proposals(pose,grip,task,schema,scene)
       if not instruction:issues.append('missing_instruction')
       # Review sidecars explicitly bind the exact source episode and pose basis.
       review=Path(review_root)/(uid+'.json') if review_root else None
       if review and review.exists():
        d=json.loads(review.read_text())
        if d['source_sha256']!=source_hash or d['member']!=member:raise ValueError('review source mismatch')
        if d.get('coordinate_frame')!=meta['coordinate_frame']:raise ValueError('review coordinate mismatch')
        annotations=d['annotations']
        for a in annotations:validate_annotation(a,len(pose))
       records=[]
       for a in annotations:
        # At primitive start and after observed events: variable event-role labels,
        # not a fixed uniform window grid. Completion itself is never an input.
        for t in sorted(set([a['start']]+[e['row'] for e in a['events'][:-1]])):
         rec=make_record(uid,a,pose,grip,meta,t)
         rec['train_eligible'] &= bool(instruction) and seed is not None
         records.append(rec)
       needed={r['inputs']['current_row'] for r in records}
       needed.update(e['row'] for r in records for e in r['targets']['keyframes'])
       images={};temp=folder/'sample.partial.tar'
       with tarfile.open(temp,'w') as tar:
        for row in sorted(needed):
         for cam,dataset in cameras.items():
          raw=image_png(h,dataset,row);name=f'images/{cam}/{row:06d}.png';tar_bytes(tar,name,raw)
          images[cam,row]={'member':name,'sha256':hashlib.sha256(raw).hexdigest(),'source_dataset':dataset,'source_row':row,'valid':True}
        for rec in records:
         rec['inputs']['images']={cam:images[cam,rec['inputs']['current_row']] for cam in cameras}
         for event in rec['targets']['keyframes']:event['images']={cam:images[cam,event['row']] for cam in cameras}
        body=''.join(json.dumps(v,ensure_ascii=False,allow_nan=False)+'\n' for v in records).encode();tar_bytes(tar,'records.jsonl',body)
        tar_bytes(tar,'episode.json',json.dumps(meta,ensure_ascii=False,allow_nan=False).encode())
       os.replace(temp,folder/'sample.tar')
       write_json(folder/'annotations.json',{'episode_id':uid,'annotations':annotations,'review_reasons':issues,'source':meta})
       r={'version':VERSION,'episode_id':uid,'source_sha256':source_hash,'member':member,'task':task,'variant':variant,
          'source_frames':len(pose),'records':len(records),'silver_records':sum(x['quality']=='silver' for x in records),
          'gold_records':sum(x['quality']=='gold' and x['train_eligible'] for x in records),'split':split,'split_group':group,
          'split_group_verified':seed is not None,'review_required':bool(issues) or any(x['quality']=='silver' for x in records),
          'status':'converted' if records else 'needs_review','tar_sha256':digest(folder/'sample.tar'),'timestamp':time.time()}
       write_json(receipt,r)
     except Exception as e:
      r={'episode_id':uid,'source_sha256':source_hash,'member':member,'status':'error','error':str(e),'traceback':traceback.format_exc(),'timestamp':time.time()}
      write_json(folder/'error.json',r)
     finally:lock.close()
   outcome['episodes']+=1
   outcome['silver_records']+=r.get('silver_records',0);outcome['gold_records']+=r.get('gold_records',0)
   outcome['review_episodes']+=int(r.get('review_required',r.get('status')=='needs_review'));outcome['failed']+=int(r['status']=='error')
   if progress:write_json(progress,{**outcome,'member':member,'updated_at':time.time()})
  return outcome

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--relative-path',required=True);p.add_argument('--output',required=True);p.add_argument('--sources',required=True);p.add_argument('--sha256',required=True);p.add_argument('--review-root');p.add_argument('--progress');a=p.parse_args()
 if digest(a.archive)!=a.sha256:raise SystemExit('source SHA mismatch')
 result=run(a.archive,a.relative_path,a.output,a.sources,a.sha256,a.review_root,a.progress);write_json(Path(a.output)/'archive_receipts'/(stable_id(a.relative_path)+'.json'),result)
 print(json.dumps(result));raise SystemExit(2 if result['failed'] else 0)
