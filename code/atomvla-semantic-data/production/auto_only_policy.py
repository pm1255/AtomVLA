"""Automatic-only acceptance policy.
No human review is required. Unsupported or inconsistent semantics are emitted as
explicit auto_unresolved spans and never silently converted to another verb.
"""
import collections,hashlib,json,re
from dataclasses import dataclass,asdict
from complete_annotation import EVENT_SCHEMAS

VERSION='atomvla_auto_only_v1'
UNKNOWN={'unknown','unspecified','none','null',''}
@dataclass(frozen=True)
class Decision:
 status:str
 confidence:float
 reason:str
 source:str
 evidence:list


def _canon(x):
 if not isinstance(x,dict):return None
 verb=x.get('verb');args=x.get('arguments',{})
 if verb not in EVENT_SCHEMAS:return None
 if not isinstance(args,dict):return None
 vals=tuple(str(args.get(k,'' )).strip().lower() for k in ('object','target','arm','completion'))
 if any(v in UNKNOWN for v in vals):return None
 events=tuple((str(e.get('role')),int(e.get('row'))) for e in x.get('events',[]))
 return verb,vals,events

def consensus(predictions,source,required=2):
 """Accept only exact semantic agreement with distinct evidence channels.
 Predictions may carry channel, evidence and physical_evidence flags.
 """
 votes=collections.defaultdict(list)
 for p in predictions:
  key=_canon(p)
  if key is None:continue
  channel=str(p.get('channel','unknown'))
  evidence=list(p.get('evidence_refs',[]))
  physical=bool(p.get('physical_evidence',False))
  votes[key].append({'channel':channel,'evidence':evidence,'physical':physical})
 if not votes:return Decision('auto_unresolved',0.,'no_valid_semantic_prediction',source,[])
 ranked=sorted(votes.items(),key=lambda kv:(len({x['channel'] for x in kv[1]}),sum(bool(x['physical']) for x in kv[1]),len(kv[1])),reverse=True)
 key,items=ranked[0];channels={x['channel'] for x in items};physical=sum(x['physical'] for x in items)
 if len(channels)<required:return Decision('auto_unresolved',len(channels)/required,'insufficient_independent_consensus',source,[*channels])
 if key[0] in ('pick','place') and physical==0:return Decision('auto_unresolved',.5,'missing_physical_object_evidence',source,[*channels])
 confidence=min(0.99,.55+.12*len(channels)+.08*physical)
 return Decision('auto_accepted',confidence,'independent_semantic_consensus',source,[*channels])

def materialize(annotation,decision,source):
 """Attach an automatic decision without upgrading evidence to ground truth."""
 out=dict(annotation);out['auto_policy']=VERSION;out['auto_decision']=asdict(decision);out['quality']='auto_gold_candidate' if decision.status=='auto_accepted' else 'auto_unresolved';out['train_eligible']=decision.status=='auto_accepted';out['verified_by_human']=False;return out

def fingerprint(annotation):return hashlib.sha256(json.dumps(annotation,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
