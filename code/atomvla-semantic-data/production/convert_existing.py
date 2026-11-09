"""New event format from existing DROID/LIBERO numeric shards; no downloads.
Preserves native action dimensions and observation/action alignment, with source
contract provenance distinct from physical replay verification.
"""
import argparse,collections,hashlib,io,json,os,tarfile,time,traceback
from pathlib import Path
import numpy as np
import h5py
from PIL import Image
from common import write_json,digest,stable_id,VERSION
from source_events import infer
from records import make_record
from convert_archive import tar_bytes

def read_instruction(meta):
 text=meta.get('instruction','').strip()
 if text or meta.get('domain')!='droid':return text
 # Current metadata may have corrected the empty historical task text. Resolve
 # exact episode identity, never borrow language from a neighbouring trajectory.
 try:
  from expand_dataset import metadata
  unit=meta['source_root']
  if unit not in read_instruction.cache:read_instruction.cache[unit]={int(r['episode_index']):r.get('tasks',[]) for r in metadata(Path(unit))}
  values=read_instruction.cache[unit].get(int(meta['source_episode_index']),[])
  return ' '.join(str(x).strip() for x in values if str(x).strip())
 except (OSError,KeyError,ValueError):return ''
read_instruction.cache={}

def get_images(meta,rows):
 out={}
 if meta['source_family']=='libero':
  with h5py.File(meta['source_hdf5'],'r') as f:
   obs=f['data'][meta['source_group']]['obs']
   for cam in [k for k in obs if k.endswith('_rgb')]:
    out[cam]={}
    for row in rows:
     im=Image.fromarray(obs[cam][row]).convert('RGB').transpose(Image.Transpose.FLIP_TOP_BOTTOM).resize((256,256),Image.Resampling.LANCZOS)
     b=io.BytesIO();im.save(b,format='PNG');out[cam][row]=(b.getvalue(),{'source_row':row,'vertical_flip':True,'source_hdf5':meta['source_hdf5']})
 else:
  from prepare_data import exact_video_frames
  for loc in meta['source_image_locators']:
   cam=loc['camera'].split('.')[-1];times=[loc['video_from_timestamp']+float(meta['_timestamp'][t]) for t in rows]
   frames=exact_video_frames(loc['source_video'],times,meta['fps']);out[cam]={}
   for row,ts in zip(rows,times):
    im=frames[ts][0].convert('RGB').resize((256,256),Image.Resampling.LANCZOS);b=io.BytesIO();im.save(b,format='PNG')
    out[cam][row]=(b.getvalue(),{'source_row':row,'source_video':loc['source_video'],'requested_timestamp':ts,'decoded_timestamp':frames[ts][1]})
 return out

def convert(receipt,root,limit=None):
 source=json.loads(Path(receipt).read_text());tarpath=Path(source['tar']);root=Path(root)
 if not tarpath.exists() or tarpath.stat().st_size!=source['bytes']:raise ValueError('source shard missing or wrong size')
 summary=collections.Counter();reasons=collections.Counter()
 with tarfile.open(tarpath) as tar:
  for meta0 in source['episodes'][:limit]:
   family=meta0.get('source_family') or ('droid' if meta0.get('domain')=='droid' else 'geniesim')
   if family not in ('droid','libero'):raise ValueError('excluded source: '+family)
   uid=meta0['episode_id'];folder=root/'episodes'/family/uid;done=folder/'receipt.json'
   if done.exists():r=json.loads(done.read_text())
   else:
    folder.mkdir(parents=True,exist_ok=True)
    try:
     meta={**meta0,'source_family':family};instruction=read_instruction(meta)
     raw=tar.extractfile(uid+'/trajectory.npz').read();arr=np.load(io.BytesIO(raw),allow_pickle=False)
     pose=arr['poses_xyzw'];grip=arr['gripper'];actions=arr['action'];state=arr['state'];n=len(pose)
     annotations,issues=infer(pose,grip,family,instruction)
     if not instruction:issues.append('missing_instruction')
     # Do not create a fake left arm: native single arm is explicitly mapped to slot0.
     poses=np.zeros((n,2,7),np.float32);poses[:,:,6]=1;poses[:,0]=pose[:,0]
     grippers=np.zeros((n,2),np.float32);grippers[:,0]=grip[:,0]
     recordmeta={'source':family,'task':meta.get('domain',family),'robot_variant':family+'_franka_single','instruction':instruction,'split':meta['split'],
      'split_group':uid,'split_group_verified':True,'coordinate_frame':family+'_native_parent_eef_xyzw_m','base_frame_calibrated':False,
      'source_trajectory_sha256':hashlib.sha256(raw).hexdigest(),'source_shard':str(tarpath),'source_shard_receipt':str(receipt),
      'source_episode_id':uid,'source_pose_convention':meta['pose_convention'],'arm_slot_mapping':{'left':'native_single_arm','right':'absent'},
      'action_representation':meta['action_representation'],'action_row_offset':int(meta.get('action_row_offset',0)),'physical_replay_verified':False}
     records=[]
     for atom in annotations:
      for t in sorted(set([atom['start']]+[e['row'] for e in atom['events'][:-1]])):
       rec=make_record(uid,atom,poses,grippers,recordmeta,t);rec['inputs']['native_state']=state[t].tolist()
       offset=recordmeta['action_row_offset'];rows=np.arange(t,min(t+8,atom['end']))+offset
       valid=(rows<len(actions)) & (rows<n);rows=rows[valid]
       chunk=np.zeros((8,actions.shape[1]),np.float32);mask=np.zeros(8,bool)
       available=bool(meta.get('action_supervision_available')) and np.isfinite(actions).all() and len(rows)>0
       if available:chunk[:len(rows)]=actions[rows];mask[:len(rows)]=True
       rec['targets'].update(actions=chunk.tolist(),action_mask=mask.tolist(),action_source_rows=rows.tolist() if available else [],action_supervision_available=available,action_contract='inherited_audited_source_native_action_not_new_replay')
       rec['train_eligible']=False;records.append(rec)
     needed=sorted({r['inputs']['current_row'] for r in records}|{e['row'] for r in records for e in r['targets']['keyframes']})
     if records:
      meta['_timestamp']=arr['timestamp'];decoded=get_images(meta,needed);refs={};tmp=folder/'sample.partial.tar'
      with tarfile.open(tmp,'w') as out:
       for cam,frames in decoded.items():
        for row,(pixels,provenance) in frames.items():
         name=f'images/{cam}/{row:06d}.png';tar_bytes(out,name,pixels);refs[cam,row]={'member':name,'sha256':hashlib.sha256(pixels).hexdigest(),**provenance}
       for rec in records:
        rec['inputs']['images']={c:refs[c,rec['inputs']['current_row']] for c in decoded}
        for e in rec['targets']['keyframes']:e['images']={c:refs[c,e['row']] for c in decoded}
       tar_bytes(out,'records.jsonl',''.join(json.dumps(x,allow_nan=False)+'\n' for x in records).encode());tar_bytes(out,'episode.json',json.dumps(recordmeta).encode())
      os.replace(tmp,folder/'sample.tar')
     write_json(folder/'annotations.json',{'episode_id':uid,'annotations':annotations,'issues':issues,'provenance':recordmeta})
     r={'episode_id':uid,'source':family,'status':'converted_silver' if records else 'needs_annotation','records':len(records),'issues':issues,'split':meta['split'],
       'action_records':sum(x['targets']['action_supervision_available'] for x in records),'tar_sha256':digest(folder/'sample.tar') if records else None,'source_trajectory_sha256':recordmeta['source_trajectory_sha256'],'time':time.time()}
     write_json(done,r)
    except Exception as e:
     r={'episode_id':uid,'source':family,'status':'error','error':str(e),'traceback':traceback.format_exc()};write_json(folder/'error.json',r)
   summary['episodes']+=1;summary[r['status']]+=1;summary['records']+=r.get('records',0);summary['action_records']+=r.get('action_records',0);reasons.update(r.get('issues',[]))
   write_json(root/'workers'/(stable_id(receipt)+'.json'),{'counts':dict(summary),'reasons':dict(reasons),'updated_at':time.time()})
 result={'source_receipt':str(receipt),'counts':dict(summary),'reasons':dict(reasons),'time':time.time()};write_json(root/'shards'/(stable_id(receipt)+'.json'),result);return result

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--receipt',required=True);p.add_argument('--root',required=True);p.add_argument('--limit',type=int);a=p.parse_args();print(json.dumps(convert(a.receipt,a.root,a.limit)))
