"""Record contract for native-coordinate targets with explicit evidence tier."""
import json,re
import numpy as np
from common import VERSION,stable_id
from complete_annotation import EVENT_SCHEMAS
ROLES={'pick':('pregrasp','closed','lifted'),'move':('arrived',),'place':('preplace','supported','released')}

def validate_annotation(a,n):
 if a['verb'] not in EVENT_SCHEMAS:raise ValueError('unsupported verb')
 args=a['arguments']
 if set(args)!= {'object','target','arm','completion'}:raise ValueError('argument schema')
 if args['arm'] not in ('left','right','both'):raise ValueError('arm')
 if any(not isinstance(v,str) or not re.fullmatch(r'[A-Za-z0-9_./:-]+',v) or v.replace('.','',1).isdigit() for v in args.values()):raise ValueError('symbolic args only')
 if type(a['start']) is not int or type(a['end']) is not int or not 0<=a['start']<a['end']<n:raise ValueError('bounds')
 if tuple(e['role'] for e in a['events']) not in EVENT_SCHEMAS[a['verb']]:raise ValueError('event roles')
 rows=[e['row'] for e in a['events']]
 if any(type(i) is not int for i in rows) or any(b<=a0 for a0,b in zip([a['start']]+rows,rows)) or rows[-1]!=a['end']:raise ValueError('event order')
 if a.get('quality') not in ('silver','gold'):raise ValueError('unknown evidence tier')
 if a['quality']=='gold':
  v=a.get('verification',{})
  if v.get('accepted') is not True or not all(isinstance(v.get(k),str) and v[k] for k in ('reviewer','evidence_ref','reviewed_at')):raise ValueError('gold requires verification provenance')
 return a

def make_record(uid,annotation,poses,gripper,meta,current_row):
 a=validate_annotation(annotation,len(poses));t=current_row
 if not a['start']<=t<a['end']:raise ValueError('current row outside primitive')
 arm_names=['left','right'];mask=[a['arguments']['arm'] in (x,'both') for x in arm_names]
 target=poses[a['end']].copy();target[~np.asarray(mask)]=0
 valid=target[np.asarray(mask)]
 if not np.isfinite(valid).all() or np.any(np.abs(np.linalg.norm(valid[:,3:],axis=-1)-1)>.015):raise ValueError('invalid endpoint')
 args=a['arguments'];program=a['verb']+'('+', '.join(k+'='+json.dumps(v) for k,v in args.items())+')'
 future=[dict(e) for e in a['events'] if e['row']>t]
 return {'schema':VERSION,'id':stable_id(uid,a['verb'],a['start'],a['end'],args['arm'],t),'episode_id':uid,
  'split':meta['split'],'split_group':meta['split_group'],'task':meta['task'],'robot_variant':meta['robot_variant'],
  'inputs':{'current_row':t,'instruction':meta['instruction'],'eef_pose_xyzw':poses[t].tolist(),'gripper':gripper[t].tolist()},
  'targets':{'program_text':program,'terminal_pose_xyzw':target.tolist(),'arm_mask':mask,'terminal_row':a['end'],
             'keyframes':future,'action_supervision_available':False},
  'coordinate_frame':meta['coordinate_frame'],'timebase':'source_frame_row_not_seconds',
  'quality':a['quality'],'train_eligible':a['quality']=='gold',
  'annotation':a,'provenance':meta}
