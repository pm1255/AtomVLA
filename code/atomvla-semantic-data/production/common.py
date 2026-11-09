"""Immutable artifact I/O. No label is upgraded by successful serialization."""
import hashlib,json,os,tempfile
from pathlib import Path
VERSION='atomvla_event_records_v2.1'
def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
def write_json(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 raw=json.dumps(value,ensure_ascii=False,allow_nan=False,sort_keys=True,indent=2).encode()
 fd,tmp=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
 try:
  with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
  os.replace(tmp,path)
 finally:
  if Path(tmp).exists():Path(tmp).unlink()
def stable_id(*parts):return hashlib.sha256('\0'.join(map(str,parts)).encode()).hexdigest()[:24]
def split_for(task,seed,episode_id):
 # Same task+seed stays together across robot variants and clean/randomized archives.
 group=f'{task}:seed:{seed}' if seed is not None else f'unknown_seed:{episode_id}'
 x=int(hashlib.sha256(group.encode()).hexdigest()[:8],16)%100
 return ('train' if x<80 else 'validation' if x<90 else 'test'),group
