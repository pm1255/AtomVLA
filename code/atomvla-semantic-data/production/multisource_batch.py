"""Bounded existing-shard conversion, explicit source allowlist, independent TB."""
import argparse,concurrent.futures,collections,json,os,subprocess,sys,time
from pathlib import Path
from common import write_json,stable_id
from torch.utils.tensorboard import SummaryWriter

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--legacy-root',required=True);p.add_argument('--source',choices=['droid','libero'],required=True);p.add_argument('--workers',type=int,default=2);p.add_argument('--max-shards',type=int);p.add_argument('--limit',type=int);p.add_argument('--hours',type=float,default=48);a=p.parse_args()
 root=Path(a.root);root.mkdir(parents=True,exist_ok=True);old=Path(a.legacy_root)
 policy=json.loads(Path(__file__).with_name('source_policy.json').read_text())
 if a.source not in policy['included'] or a.source in policy['excluded']:raise ValueError('source excluded')
 pattern='lerobot/droid*.json' if a.source=='droid' else 'libero/*.json'
 files=sorted((old/'data/full_v3/shards').glob(pattern))
 # Smoke and first real jobs prioritise tasks with supported single-object text.
 if a.source=='libero':
  from source_events import binding
  def supported(path):
   d=json.loads(path.read_text());return bool(d['episodes'] and binding(d['episodes'][0].get('instruction','')))
  files.sort(key=lambda p:(not supported(p),p.name))
 if a.max_shards:files=files[:a.max_shards]
 write_json(root/'plan.json',{'source':a.source,'files':[str(p) for p in files],'policy':policy,'diagnostic':a.limit is not None})
 writer=SummaryWriter(str(old/'tensorboard/semantic_multisource'/root.name));start=time.time();totals=collections.Counter();errors=[]
 (root/'logs').mkdir(exist_ok=True)
 def one(path):
  cmd=[sys.executable,str(Path(__file__).with_name('convert_existing.py')),'--receipt',str(path),'--root',str(root)]
  if a.limit:cmd+=['--limit',str(a.limit)]
  with (root/'logs'/(stable_id(path)+'.log')).open('w') as f:
   child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);began=time.time()
   while child.poll() is None:
    if time.time()-start>a.hours*3600 or time.time()-began>3600:
     import signal
     os.killpg(child.pid,signal.SIGTERM)
     try:child.wait(15)
     except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
     return {'error':'time_limit','receipt':str(path)}
    time.sleep(2)
  if child.returncode:return {'error':'worker_exit_'+str(child.returncode),'receipt':str(path)}
  return json.loads((root/'shards'/(stable_id(path)+'.json')).read_text())
 pending={};index=0
 with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
  while index<len(files) or pending:
   while index<len(files) and len(pending)<a.workers and time.time()-start<a.hours*3600:
    path=files[index];pending[pool.submit(one,path)]=path;index+=1
   if not pending:break
   done,_=concurrent.futures.wait(pending,timeout=5,return_when=concurrent.futures.FIRST_COMPLETED)
   for f in done:
    path=pending.pop(f)
    try:d=f.result()
    except Exception as e:d={'error':str(e),'receipt':str(path)}
    if 'error' in d:errors.append(d)
    else:totals.update(d['counts']);totals['shards_completed']+=1
   state={'source':a.source,'counts':dict(totals),'errors':errors,'total_shards':len(files),'active_workers':len(pending),'submitted_shards':index,'elapsed_seconds':time.time()-start,'updated_at':time.time(),'full_joint_ready':False}
   write_json(root/'progress.json',state);step=int(time.time()-start)
   for k,v in totals.items():writer.add_scalar('conversion/'+k,v,step)
   for k,v in [('total_shards',len(files)),('active_workers',len(pending))]:writer.add_scalar('conversion/'+k,v,step)
   writer.add_scalar('training/waiting_full_joint_implementation',1,step)
   for path in pending.values():
    pr=root/'workers'/(stable_id(path)+'.json')
    try:d=json.loads(pr.read_text())
    except (OSError,ValueError):continue
    for k,v in d['counts'].items():writer.add_scalar('workers/'+stable_id(path)[:8]+'/'+k,v,step)
   writer.flush()
 write_json(root/'DONE.json',state)
 if a.limit is None:
  from release_existing import publish
  release=publish(root)
  for k,v in release['counts'].items():writer.add_scalar('release/'+k,v,int(time.time()-start))
 writer.close();print(json.dumps(state))
if __name__=='__main__':main()
