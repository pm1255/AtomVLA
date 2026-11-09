from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional
import json, hashlib

MODES = ('benchmark','ablation','memory','coupling','features','real_robot','smoke')
BENCHMARKS = ('libero','libero_pro','robotwin','rmbench','real_robot')
VARIANTS = ('full','no_program','no_query','no_rgb','verb_only','direct_action','pi05','text_conditioned','feedback_off','feedback_on','feedback_shuffled','memory_reset','memory_shuffled','memory_fixed')

@dataclass(frozen=True)
class Episode:
    episode_id: str
    benchmark: str
    task: str
    split: str
    seed: int
    source: str
    trajectory: str = ''
    success_key: str = 'success'
    metadata: Optional[Dict[str, Any]] = None
    def validate(self):
        if self.benchmark not in BENCHMARKS: raise ValueError(f'unknown benchmark: {self.benchmark}')
        if self.split not in ('train','validation','test'): raise ValueError('invalid split')
        if not self.task or not self.episode_id: raise ValueError('missing task or episode_id')
        return self

@dataclass(frozen=True)
class TrialResult:
    episode_id: str
    task: str
    variant: str
    success: bool
    steps: int
    retries: int = 0
    pose_error_m: Optional[float] = None
    contract_ok: Optional[bool] = None
    error: Optional[str] = None
    extras: Optional[Dict[str, Any]] = None
    def as_dict(self): return asdict(self)

def load_manifest(path):
    rows=[]
    with open(path, encoding='utf-8') as stream:
      for line in stream:
        if line.strip(): rows.append(Episode(**json.loads(line)).validate())
    if not rows: raise ValueError('empty evaluation manifest')
    ids=[x.episode_id for x in rows]
    if len(ids)!=len(set(ids)): raise ValueError('duplicate episode_id')
    return rows

def save_json(path, value):
    import pathlib
    p=pathlib.Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,ensure_ascii=False,indent=2), encoding='utf-8')

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
