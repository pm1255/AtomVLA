"""Read official native and XPolicyLab HDF5 without unpickling archive payloads."""
import io,json,re
import numpy as np
from PIL import Image

def read_arrays(h):
 if 'endpose/left_endpose' in h:
  pose=np.stack([h[f'endpose/{a}_endpose'][()] for a in ['left','right']],axis=1)
  grip=np.stack([h[f'endpose/{a}_gripper'][()] for a in ['left','right']],axis=1)
  cameras={k:'observation/'+k+'/rgb' for k in h['observation'] if 'rgb' in h['observation'][k]}
  if 'third_view_rgb' in h:cameras['third_view']='third_view_rgb'
  schema='robotwin_native'
 elif 'state/left_ee_poses' in h and 'vision' in h:
  if h['data_format_version'].asstr()[()]!='v1.0':raise ValueError('unknown standardized format')
  pose=np.stack([h[f'state/{a}_ee_poses'][()] for a in ['left','right']],axis=1)
  grip=np.concatenate([h[f'state/{a}_ee_joint_states'][()].reshape(-1,1) for a in ['left','right']],axis=1)
  cameras={k:'vision/'+k+'/colors' for k in h['vision'] if 'colors' in h['vision'][k]}
  schema='robotwin_xpolicylab_v1'
 else:raise ValueError('unsupported source schema')
 pose=np.asarray(pose,dtype=np.float32)[...,[0,1,2,4,5,6,3]]
 grip=np.asarray(grip,dtype=np.float32);n=len(pose)
 if n<3 or not cameras or pose.shape!=(n,2,7) or grip.shape!=(n,2):raise ValueError('invalid source dimensions')
 if not np.isfinite(pose).all() or not np.isfinite(grip).all():raise ValueError('nonfinite source')
 norms=np.linalg.norm(pose[...,3:],axis=-1)
 if np.any(np.abs(norms-1)>.015):raise ValueError('invalid quaternion')
 pose[...,3:]/=norms[...,None]
 if any(len(h[v])!=n for v in cameras.values()):raise ValueError('unsynchronized camera lengths')
 return pose,grip,cameras,schema

def image_png(h,dataset,row,size=256):
 raw=h[dataset][row]
 if isinstance(raw,(bytes,np.bytes_)) or (isinstance(raw,np.ndarray) and raw.ndim==1):
  im=Image.open(io.BytesIO(bytes(raw)))
 else:im=Image.fromarray(np.asarray(raw,dtype=np.uint8))
 im.load();im=im.convert('RGB');im=im.resize((size,size),Image.Resampling.LANCZOS)
 b=io.BytesIO();im.save(b,format='PNG');return b.getvalue()

def episode_number(name):
 m=re.search(r'episode_?(\d+)',name)
 if not m:raise ValueError('missing episode identity')
 return int(m.group(1))

def metadata(z,member,h):
 eid=episode_number(member);prefix=member.split('/data/')[0]
 scene_path=prefix+'/scene_info.json';scene={}
 if scene_path in z.namelist():
  scene_all=json.loads(z.read(scene_path));scene=scene_all.get(f'episode_{eid}',scene_all.get(f'episode{eid}',{}))
 seeds=None;seed_path=prefix+'/seed.txt'
 if seed_path in z.namelist():
  tokens=z.read(seed_path).decode().split()
  try:seeds=int(tokens[eid])
  except (ValueError,IndexError):pass
 instruction=''
 for folder in ['instructions','instruction']:
  for stem in [f'episode{eid}',f'episode_{eid:07d}',f'episode_{eid}']:
   path=f'{prefix}/{folder}/{stem}.json'
   if path in z.namelist():
    item=json.loads(z.read(path));v=item.get('seen',item.get('instruction',[])) if isinstance(item,dict) else item
    if isinstance(v,list) and v:instruction=str(v[0])
    elif isinstance(v,str):instruction=v
    if instruction:break
  if instruction:break
 if not instruction and 'instructions' in h:
  v=json.loads(h['instructions'].asstr()[()]);instruction=next((x for x in v if isinstance(x,str) and x),'') if isinstance(v,list) else ''
 return eid,scene,seeds,instruction
