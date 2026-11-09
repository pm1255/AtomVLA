"""Gold-default image/pose/text loader. Silver requires an explicit opt-in."""
import io,json,tarfile
import numpy as np
from PIL import Image
from common import VERSION

def load_episode(path,allow_silver=False):
 with tarfile.open(path,'r') as tar:
  for line in tar.extractfile('records.jsonl'):
   r=json.loads(line)
   if r['schema']!=VERSION:raise ValueError('record version mismatch')
   if not r['train_eligible'] and not (allow_silver and r['quality']=='silver'):continue
   def pixels(ref):
    im=Image.open(io.BytesIO(tar.extractfile(ref['member']).read())).convert('RGB');return np.asarray(im)
   r['current_rgb']={c:pixels(ref) for c,ref in r['inputs']['images'].items()}
   r['future_rgb']=[{c:pixels(ref) for c,ref in e['images'].items()} for e in r['targets']['keyframes']]
   yield r

def collate(records):
 if not records:raise ValueError('empty batch')
 cameras=sorted(set(c for r in records for frame in r['future_rgb'] for c in frame));length=max(len(r['future_rgb']) for r in records)
 first=next(iter(records[0]['future_rgb'][0].values()));h,w,_=first.shape
 rgb=np.zeros((len(records),length,len(cameras),h,w,3),dtype=np.uint8);mask=np.zeros(rgb.shape[:3],bool)
 for b,r in enumerate(records):
  for k,frame in enumerate(r['future_rgb']):
   for j,c in enumerate(cameras):
    if c in frame:
     if frame[c].shape!=(h,w,3):raise ValueError('resize heterogeneous cameras explicitly before batching')
     rgb[b,k,j]=frame[c];mask[b,k,j]=True
 return {'future_rgb':rgb,'future_mask':mask,'terminal_pose':np.asarray([r['targets']['terminal_pose_xyzw'] for r in records]),
         'arm_mask':np.asarray([r['targets']['arm_mask'] for r in records]),'program_text':[r['targets']['program_text'] for r in records],
         'event_roles':[[e['role'] for e in r['targets']['keyframes']] for r in records],
         'current_rgb':[r['current_rgb'] for r in records],
         'instruction':[r['inputs']['instruction'] for r in records],
         'current_pose':np.asarray([r['inputs']['eef_pose_xyzw'] for r in records],dtype=np.float32),
         'current_gripper':np.asarray([r['inputs']['gripper'] for r in records],dtype=np.float32),
         'robot_variant':[r['robot_variant'] for r in records],
         'coordinate_frame':[r['coordinate_frame'] for r in records],
         'loss_weights':[r.get('loss_weights') for r in records]}

def load_release(manifest,accepted_policy):
 """Read released indices only; explicit policy prevents silent silver opt-in."""
 from collections import defaultdict
 from common import digest
 by_tar=defaultdict(dict)
 with open(manifest) as f:
  for line in f:
   e=json.loads(line)
   if e['policy']!=accepted_policy:raise ValueError('release supervision policy must be explicitly accepted')
   if e['id'] in by_tar[e['tar']]:raise ValueError('duplicate manifest record')
   by_tar[e['tar']][e['id']]=e
 for path,entries in by_tar.items():
  checksums={e['tar_sha256'] for e in entries.values()}
  if len(checksums)!=1 or digest(path) not in checksums:raise ValueError('release artifact changed')
  found=set()
  for r in load_episode(path,allow_silver=True):
   if r['id'] in entries:
    e=entries[r['id']];r['loss_weights']=e['loss_weights'];r['release_policy']=e['policy'];found.add(r['id']);yield r
  if found!=set(entries):raise ValueError('release IDs missing from artifact')
