"""Measured-state audit across actual LIBERO tasks; no semantic completion claims."""
import argparse,collections,json,time,traceback
from pathlib import Path
import h5py,numpy as np
from common import write_json
from libero_state_evidence import extract
from trajectory_events import build

def main():
 p=argparse.ArgumentParser();p.add_argument('--legacy',required=True);p.add_argument('--root',required=True);p.add_argument('--per-task',type=int,default=1);a=p.parse_args();root=Path(a.root);root.mkdir(parents=True,exist_ok=True)
 from torch.utils.tensorboard import SummaryWriter
 w=SummaryWriter(str(Path(a.legacy)/'tensorboard/semantic_event_grounding'/root.name));counts=collections.Counter();errors=[];began=time.time()
 with (root/'episodes.jsonl').open('w') as out:
  for receipt in sorted((Path(a.legacy)/'data/full_v3/shards/libero').glob('*.json')):
   for meta in json.loads(receipt.read_text())['episodes'][:a.per_task]:
    counts['attempted']+=1
    try:
     with h5py.File(meta['source_hdf5']) as f:
      g=f['data'][meta['source_group']];scene=extract(meta['model_file'],g['states'][()],g['obs/joint_states'][()],g['obs/gripper_states'][()]);n=len(g['states'])
      # EEF orientation not needed to validate source state extraction; use the
      # actual trajectory quaternion from its legacy numeric shard instead.
      import tarfile,io
      src=json.loads(receipt.read_text())
      with tarfile.open(src['tar']) as t:raw=t.extractfile(meta['episode_id']+'/trajectory.npz').read()
      with np.load(io.BytesIO(raw)) as ar:events=build(ar['poses_xyzw'],ar['gripper'],ar['timestamp'],'aperture_metres',scene)
      d={'episode_id':meta['episode_id'],'instruction':meta['instruction'],'alignment':scene['alignment'],'objects':list(scene['objects']),'articulations':list(scene['articulations']),'events':events,'source_hdf5':meta['source_hdf5'],'source_group':meta['source_group']}
      out.write(json.dumps(d,allow_nan=False)+'\n');counts['state_decoded']+=1;counts['object_tracks']+=len(scene['objects']);counts['articulation_tracks']+=len(scene['articulations']);counts['missing_aligned_rows']+=int((~scene['valid']).sum())
      for e in events['events']:counts['events_'+e['kind']]+=1
    except Exception as exc:errors.append({'episode_id':meta['episode_id'],'error':str(exc)});counts['errors']+=1
    state={'counts':dict(counts),'errors':errors,'elapsed_seconds':time.time()-began,'full_semantic_annotation_complete':False,'updated_at':time.time()};write_json(root/'progress.json',state)
    for k,v in counts.items():w.add_scalar('evidence/'+k,v,counts['attempted'])
    w.add_scalar('training/waiting',1,counts['attempted']);w.flush()
 write_json(root/'report.json',state);w.close();print(json.dumps(state))
if __name__=='__main__':main()
