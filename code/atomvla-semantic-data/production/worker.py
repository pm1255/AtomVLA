import argparse,fcntl,json,sys
from pathlib import Path
from common import stable_id,write_json,digest
from convert_archive import run
p=argparse.ArgumentParser()
for name in ['item','revision','root','legacy-root','progress']:p.add_argument('--'+name,required=True)
p.add_argument('--mode',choices=['combined','download','convert'],default='combined')
a=p.parse_args();root=Path(a.root);old=Path(a.legacy_root);item=json.loads(a.item)
(root/'locks').mkdir(exist_ok=True);lock=(root/'locks'/(stable_id(item['path'])+'.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX)
receipt=root/'archive_receipts'/(stable_id(item['path'])+'.json')
if receipt.exists():
 print(receipt.read_text());raise SystemExit(0)
sys.path.append(str(old/'code'));from full_hf_download import download
source=old/'data/full_v3/sources/TianxingChen__RoboTwin2.0'/item['path']
if a.mode=='convert':
 if not source.exists() or source.stat().st_size!=item['size']:raise ValueError('convert-only source unavailable; download forbidden')
else:source=download(item,'TianxingChen/RoboTwin2.0',a.revision,old/'data/full_v3/sources',Path(a.progress))
if digest(source)!=item['lfs']['oid']:raise ValueError('source checksum invalid')
if a.mode=='download':
 write_json(receipt,{'relative_path':item['path'],'source_sha256':item['lfs']['oid'],'bytes':item['size'],'download_only':True,'failed':0})
 print('download verified');raise SystemExit(0)
result=run(source,item['path'],root,root/'task_sources',item['lfs']['oid'],root/'reviews',a.progress)
write_json(receipt,result);print(json.dumps(result));raise SystemExit(2 if result['failed'] else 0)
