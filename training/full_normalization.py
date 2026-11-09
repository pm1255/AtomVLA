"""Streaming train-only moments per native robot action space; no test leakage."""
import argparse,io,json,tarfile,time
from pathlib import Path
import numpy as np
from full_common import dump,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--catalog',type=Path,required=True);a=p.parse_args()
    rows=[json.loads(x) for x in (a.catalog/'episodes.jsonl').read_text().splitlines()]
    rows=[r for r in rows if r['split']=='train' and r['action_supervision_available']];stats={};used=[]
    bytar={}
    for row in rows:bytar.setdefault(row['tar'],[]).append(row)
    for path,group in bytar.items():
        with tarfile.open(path) as tar:
            for r in group:
                with np.load(io.BytesIO(tar.extractfile(r['episode_id']+'/trajectory.npz').read()),allow_pickle=False) as z:
                    state=z['state'].astype(np.float64);action=z['action'].astype(np.float64)
                offset=r.get('action_row_offset',0)
                if offset:state=state[:-offset];action=action[offset:]
                if not len(action):continue
                assert len(state)==len(action) and np.isfinite(state).all() and np.isfinite(action).all()
                key=r['domain'];item=stats.setdefault(key,{'episodes':0,'frames':0,'state_sum':np.zeros(state.shape[1]),'state_sumsq':np.zeros(state.shape[1]),'action_sum':np.zeros(action.shape[1]),'action_sumsq':np.zeros(action.shape[1])})
                item['episodes']+=1;item['frames']+=len(action)
                for name,value in [('state',state),('action',action)]:item[name+'_sum']+=value.sum(axis=0);item[name+'_sumsq']+=(value*value).sum(axis=0)
                used.append(r['episode_id'])
    result={}
    for domain,x in stats.items():
        value={'episodes':x['episodes'],'frames':x['frames']}
        for name in ['state','action']:
            mean=x[name+'_sum']/x['frames'];var=np.maximum(0,x[name+'_sumsq']/x['frames']-mean*mean)
            value[name+'_mean']=mean.tolist();value[name+'_std']=np.maximum(np.sqrt(var),.01).tolist()
        result[domain]=value
    dump(a.catalog/'normalization.json',{'by_domain':result,'scope':'all aligned rows from canonical TRAIN episodes only','minimum_std':.01,'used_episode_ids':used,'catalog_episodes_sha256':sha(a.catalog/'episodes.jsonl'),'time':time.time()})
    print(json.dumps({'domains':{k:{'episodes':v['episodes'],'frames':v['frames']} for k,v in result.items()}}))
if __name__=='__main__':main()
