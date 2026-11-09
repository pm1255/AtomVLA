"""Build an automatic-only release from converted episodes.
Every source episode must appear in the inventory; only consensus labels are train eligible.
"""
import argparse,collections,json,time
from pathlib import Path
from common import write_json,digest
from auto_only_policy import VERSION

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root);out=root/'auto_release';out.mkdir(exist_ok=True);counts=collections.defaultdict(collections.Counter)
 inv=[json.loads(x) for x in (root/'inventory.jsonl').read_text().splitlines()]
 files={s:(out/f'{s}.jsonl').open('w') for s in ('train','validation','test')};errors=[]
 try:
  for item in inv:
   s,uid=item['source'],item['episode_id'];c=counts[s];c['inventory']+=1;folder=root/'episodes'/s/uid;receipt=folder/'receipt.json';ann=folder/'annotations.json'
   if not receipt.exists() or not ann.exists():c['not_processed']+=1;continue
   try:
    r=json.loads(receipt.read_text());d=json.loads(ann.read_text());segs=d.get('segments',[]);atoms=d.get('annotations',[])
    c['episodes_processed']+=1;c['episodes_with_full_timeline']+=int(bool(d.get('full_timeline_accounted')));c['unresolved_segments']+=sum(x.get('kind') in ('unresolved','idle','hold') for x in segs);c['candidate_atoms']+=len(atoms)
    for i,x in enumerate(atoms):
     # Current converter stores automatic evidence; no semantic label is silently promoted.
     eligible=bool(x.get('auto_decision',{}).get('status')=='auto_accepted' and x.get('train_eligible'))
     split={'val':'validation'}.get(x.get('split'),x.get('split','train'));f=files.get(split)
     if f is None:raise ValueError('invalid split')
     row={'episode_id':uid,'source':s,'record_index':i,'tar':str(folder/'sample.tar'),'tar_sha256':r.get('tar_sha256'),'quality':'auto_gold_candidate' if eligible else 'auto_unresolved','train_eligible':eligible,'policy':VERSION,'verified_by_human':False}
     f.write(json.dumps(row,ensure_ascii=False)+'\n');c['train_records' if eligible else 'quarantine_records']+=1
   except Exception as exc:errors.append({'source':s,'episode_id':uid,'error':str(exc)});c['artifact_errors']+=1
 finally:
  for f in files.values():f.close()
 report={'schema':VERSION,'counts':{s:dict(c) for s,c in counts.items()},'errors':errors,'all_source_episodes_in_inventory':sum(c['inventory'] for c in counts.values())==len(inv),'all_episodes_processed':all(c['inventory']==c['episodes_processed'] for c in counts.values()),'all_semantics_auto_verified':False,'training_release_ready':False,'policy':'automatic_consensus_with_abstention','time':time.time()};write_json(out/'report.json',report);print(json.dumps(report))
if __name__=='__main__':main()
