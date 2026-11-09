"""Snapshot existing RoboTwin sources; download independently, then convert new sources."""
import argparse,json,os,subprocess,sys,time,fcntl
from pathlib import Path
from common import write_json,stable_id
from torch.utils.tensorboard import SummaryWriter

def main():
 p=argparse.ArgumentParser();p.add_argument('--control',required=True);p.add_argument('--data',required=True);p.add_argument('--legacy-root',required=True);a=p.parse_args()
 root=Path(a.control);data=Path(a.data);old=Path(a.legacy_root);root.mkdir(parents=True,exist_ok=True)
 lock=(root/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 inv=json.loads((old/'configs/full_v3_hf_inventory.json').read_text())['TianxingChen/RoboTwin2.0']
 cache=old/'data/full_v3/sources/TianxingChen__RoboTwin2.0';files=[x for x in inv['files'] if x['path'].startswith('dataset/') and x['path'].endswith('.zip')]
 existing=[x['path'] for x in files if (cache/x['path']).exists() and (cache/x['path']).stat().st_size==x['size']];missing=[x['path'] for x in files if x['path'] not in set(existing)]
 write_json(root/'existing.json',{'paths':existing});write_json(root/'missing.json',{'paths':missing})
 writer=SummaryWriter(str(old/'tensorboard/semantic_split/controller'));start=time.time()
 def launch(mode,scope,plan,hours,tag):
  cmd=[sys.executable,str(Path(__file__).with_name('batch.py')),'--root',str(scope),'--legacy-root',str(old),'--mode',mode,'--plan',str(plan),'--workers','4','--hours',str(hours),'--tb-run',tag]
  with (root/(tag+'.log')).open('w') as log:child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
  write_json(root/(tag+'_submission.json'),{'pid':child.pid,'command':cmd,'time':time.time()});return child
 download=launch('download',root/'download',root/'missing.json',48,'semantic_split_download')
 convert=launch('convert',data,root/'existing.json',48,'semantic_split_existing')
 state={'existing_packages':len(existing),'missing_packages':len(missing),'new_conversion':'waiting_download_completion','scope':'RoboTwin_only','started_at':start}
 try:
  while download.poll() is None or convert.poll() is None:
   state.update(download='running' if download.poll() is None else 'ended',existing_conversion='running' if convert.poll() is None else 'ended',updated_at=time.time())
   write_json(root/'progress.json',state);step=int(time.time()-start)
   writer.add_scalar('phase/download_running',int(download.poll() is None),step);writer.add_scalar('phase/existing_conversion_running',int(convert.poll() is None),step);writer.add_scalar('phase/new_conversion_waiting',1,step);writer.flush();time.sleep(5)
  state.update(download_returncode=download.returncode,existing_returncode=convert.returncode)
  # Only fully verified downloads are eligible, never .partial files.
  completed=[]
  for x in files:
   if x['path'] in missing and (root/'download/archive_receipts'/(stable_id(x['path'])+'.json')).exists():completed.append(x['path'])
  write_json(root/'new_downloaded.json',{'paths':completed});state['downloaded_packages']=len(completed)
  if completed:
   state['new_conversion']='running';write_json(root/'progress.json',state)
   new=launch('convert',data,root/'new_downloaded.json',48,'semantic_split_new')
   while new.poll() is None:
    step=int(time.time()-start);writer.add_scalar('phase/new_conversion_waiting',0,step);writer.add_scalar('phase/new_conversion_running',1,step);writer.flush();time.sleep(5)
   state['new_conversion_returncode']=new.returncode
  state['new_conversion']='ended' if completed else 'no_verified_new_downloads';state['finished_at']=time.time();write_json(root/'DONE.json',state);write_json(root/'progress.json',state)
 finally:writer.close()
if __name__=='__main__':main()
