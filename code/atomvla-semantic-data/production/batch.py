"""Bounded parallel acquisition + event conversion. Gold release is independent."""
import argparse,concurrent.futures,fcntl,json,os,signal,subprocess,sys,time
from pathlib import Path
from common import write_json,stable_id,digest
from torch.utils.tensorboard import SummaryWriter

def read_progress(path):
 # Downloader may unlink or replace its temporary progress file between calls.
 try:return json.loads(Path(path).read_text())
 except (OSError,ValueError):return {}

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--legacy-root',required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--hours',type=float,default=48);p.add_argument('--cached-only',action='store_true');p.add_argument('--max-files',type=int);p.add_argument('--mode',choices=['combined','download','convert'],default='combined');p.add_argument('--plan');p.add_argument('--tb-run',default='semantic_v2_r3');a=p.parse_args()
 root=Path(a.root);old=Path(a.legacy_root);root.mkdir(parents=True,exist_ok=True)
 lock=(root/'batch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 inv=json.loads((old/'configs/full_v3_hf_inventory.json').read_text())['TianxingChen/RoboTwin2.0']
 files=[x for x in inv['files'] if x['path'].startswith('dataset/') and x['path'].endswith('.zip')]
 cache=old/'data/full_v3/sources/TianxingChen__RoboTwin2.0'
 files.sort(key=lambda x:(not (cache/x['path']).exists(),'randomized' in x['path'],x['size'],x['path']))
 if a.plan:
  wanted=set(json.loads(Path(a.plan).read_text())['paths']);files=[x for x in files if x['path'] in wanted]
 if a.cached_only:files=[x for x in files if (cache/x['path']).exists()]
 if a.max_files:files=files[:a.max_files]
 write_json(root/'source_plan.json',{'revision':inv['revision'],'files':files,'bytes':sum(f['size'] for f in files)})
 (root/'logs').mkdir(exist_ok=True);(root/'worker_progress').mkdir(exist_ok=True)
 writer=SummaryWriter(str(old/'tensorboard'/a.tb_run/root.name));start=time.time();stopped=False
 signal.signal(signal.SIGTERM,lambda *_:globals().update(STOP=True));globals()['STOP']=False
 totals={'files_completed':0,'files_failed':0,'episodes':0,'silver_records':0,'gold_records':0,'review_episodes':0,'episode_errors':0}
 def one(item,worker):
  key=stable_id(item['path']);progress=root/'worker_progress'/(key+'.json');receipt=root/'archive_receipts'/(key+'.json')
  cmd=[sys.executable,str(Path(__file__).with_name('worker.py')),'--item',json.dumps(item),'--revision',inv['revision'],'--root',str(root),'--legacy-root',str(old),'--progress',str(progress),'--mode',a.mode]
  with (root/'logs'/(key+'.log')).open('a') as log:
   child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   began=time.time();last=began;stamp=None
   while child.poll() is None:
    try:current=progress.stat().st_mtime_ns
    except OSError:current=None
    if current!=stamp:stamp=current;last=time.time()
    if globals()['STOP'] or time.time()-start>a.hours*3600 or time.time()-began>6*3600 or time.time()-last>1200:
     os.killpg(child.pid,signal.SIGTERM)
     try:child.wait(20)
     except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
     return {'failed':True,'path':item['path'],'reason':'bounded_timeout_or_stop'}
    time.sleep(2)
   if child.returncode!=0 or not receipt.exists():return {'failed':True,'path':item['path'],'reason':'worker_failed','returncode':child.returncode}
   return json.loads(receipt.read_text())
 futures={};index=0
 with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
  while index<len(files) or futures:
   if time.time()-start>a.hours*3600:globals()['STOP']=True
   while not globals()['STOP'] and index<len(files) and len(futures)<a.workers:
    item=files[index];futures[pool.submit(one,item,index)]=item;index+=1
   done,_=concurrent.futures.wait(futures,timeout=5,return_when=concurrent.futures.FIRST_COMPLETED) if futures else ([],[])
   for f in done:
    item=futures.pop(f)
    try:d=f.result()
    except Exception as e:d={'failed':True,'path':item['path'],'reason':str(e)}
    if d.get('failed') is True:
     totals['files_failed']+=1;write_json(root/'failures'/(stable_id(item['path'])+'.json'),d)
    else:
     totals['files_completed']+=1
     for k in ['episodes','silver_records','gold_records','review_episodes']:totals[k]+=d.get(k,0)
     totals['episode_errors']+=d.get('failed',0)
   state={**totals,'total_files':len(files),'active_workers':len(futures),'submitted_files':index,'updated_at':time.time(),'elapsed_seconds':time.time()-start,
          'mode':a.mode,'status':'stopping' if globals()['STOP'] else 'running','training_ready':False,'stage1':'waiting_semantic_release','stage2':'waiting_stage1_and_action_validation'}
   write_json(root/'progress.json',state)
   step=int(time.time()-start)
   for k,v in totals.items():writer.add_scalar('conversion/'+k,v,step)
   writer.add_scalar('conversion/total_files',len(files),step);writer.add_scalar('conversion/active_workers',len(futures),step)
   writer.add_scalar('training/stage1_waiting',1,step);writer.add_scalar('training/stage2_waiting',1,step)
   for f,item in futures.items():
    pr=root/'worker_progress'/(stable_id(item['path'])+'.json')
    if pr.exists():
     try:
      v=read_progress(pr);tag=stable_id(item['path'])[:8]
      for k in ['bytes','episodes','silver_records','gold_records']:
       if k in v:writer.add_scalar(f'workers/{tag}/{k}',v[k],step)
     except (OSError,ValueError):pass
   writer.flush()
   if globals()['STOP'] and not futures:break
 write_json(root/'BATCH_FINISHED.json',{**state,'status':'bounded_stop' if globals()['STOP'] else 'finished','not_a_training_release':True});writer.close()
 if a.mode=='download':return
 # Always publish a bounded snapshot of successful conversions, including when
 # acquisition reaches its time limit. This does not certify physical success.
 release=root/'releases'/('stage1-weak-'+str(int(time.time())))
 with (root/'release.log').open('a') as log:
  result=subprocess.run([sys.executable,str(Path(__file__).with_name('release.py')),'--root',str(root),'--output',str(release)],stdout=log,stderr=subprocess.STDOUT)
 write_json(root/'release_status.json',{'returncode':result.returncode,'path':str(release),'policy':'stage1_weak_events_v1'})
if __name__=='__main__':main()
