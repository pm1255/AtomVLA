"""Independent readback; gold release never includes silver proposals."""
import argparse,json,tarfile,hashlib
from pathlib import Path
from common import digest,write_json,VERSION
from loader import load_episode,collate
p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root)
out={'episodes':0,'records':0,'gold':0,'silver':0,'errors':[],'groups':{}};index=[]
for receipt in sorted((root/'episodes').glob('*/*/receipt.json')):
 try:
  d=json.loads(receipt.read_text());tarpath=receipt.parent/'sample.tar'
  if digest(tarpath)!=d['tar_sha256']:raise ValueError('artifact hash mismatch')
  with tarfile.open(tarpath) as t:
   records=[json.loads(x) for x in t.extractfile('records.jsonl')]
   for r in records:
    if r['schema']!=VERSION:raise ValueError('schema')
    k=r['split_group'];old=out['groups'].setdefault(k,r['split'])
    if old!=r['split']:raise ValueError('split leakage')
    if not all(r['inputs']['current_row']<e['row']<=r['targets']['terminal_row'] for e in r['targets']['keyframes']):raise ValueError('future boundaries')
    for frame in [r['inputs']['images']]+[e['images'] for e in r['targets']['keyframes']]:
     for ref in frame.values():
      if hashlib.sha256(t.extractfile(ref['member']).read()).hexdigest()!=ref['sha256']:raise ValueError('image hash')
    out[r['quality']]+=1
    if r['train_eligible']:index.append({'tar':str(tarpath),'id':r['id'],'split':r['split']})
  actual=list(load_episode(tarpath,allow_silver=True))
  if actual:collate(actual[:1])
  out['episodes']+=1;out['records']+=len(records)
 except Exception as e:out['errors'].append({'path':str(receipt),'error':str(e)})
out['groups']=len(out['groups']);out['gold_ready']=bool(index) and not out['errors']
write_json(root/'audit.json',out);write_json(root/'gold_index.json',index)
# This release describes only the explicitly audited gold subset, not all source coverage.
if out['gold_ready']:write_json(root/'GOLD_SUBSET_READY.json',{'scope':'reviewed_subset_only','records':len(index),'audit':str(root/'audit.json')})
print(json.dumps(out))
