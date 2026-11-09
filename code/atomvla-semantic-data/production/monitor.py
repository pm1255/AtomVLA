"""Read-only progress bridge for already running conversion phases."""
import argparse,json,time
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
p=argparse.ArgumentParser();p.add_argument('--progress',required=True);p.add_argument('--logdir',required=True);p.add_argument('--seconds',type=int,default=900);a=p.parse_args()
w=SummaryWriter(a.logdir);start=time.time()
while time.time()-start<a.seconds:
 f=Path(a.progress)
 if f.exists():
  try:
   d=json.loads(f.read_text())
   for k in ['episodes','silver_records','gold_records','failed','review_episodes']:w.add_scalar('conversion/'+k,d.get(k,0),int(time.time()-start))
   w.add_scalar('training/waiting',1,int(time.time()-start));w.flush()
  except ValueError:pass
 time.sleep(5)
w.close()
