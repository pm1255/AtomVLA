"""Independent monitor; never imports the simulator or interrupts computation."""
import argparse
import json
from pathlib import Path
import time
from monitor import Monitor

p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--run',required=True)
a=p.parse_args(); root=Path(a.root); m=Monitor(root,a.run); last=None;step=0
m.write(0,**{'status/waiting_for_environment':1,'evaluation/running':0})
while True:
    f=root/(a.run+'_progress.json')
    if f.exists():
        raw=f.read_text()
        if raw != last:
            try: data=json.loads(raw)
            except json.JSONDecodeError: time.sleep(2);continue
            step+=1; m.write(step,**data['metrics']);last=raw
            if data.get('finished'):break
    time.sleep(5)
m.close()
