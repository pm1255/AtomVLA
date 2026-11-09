"""Derive an immutable LIBERO benchmark index from an already released source catalog.

Keeps the original episode splits and exact per-domain TRAIN statistics. Never
reads perturbation evaluation assets or turns an unreleased catalog into READY.
RoboTwin remains blocked until its native control adapter has been verified.
"""
import argparse,json,time
from pathlib import Path
from full_common import dump,sha
from full_ft.data import REGISTRY,build_index

# full_simulation.py namespaces the official suite name with "libero_".
DOMAINS={'libero_libero_spatial','libero_libero_object','libero_libero_goal','libero_libero_10'}

def derive(source,destination,benchmark,diagnostic=False):
    source=Path(source).resolve();destination=Path(destination).resolve()
    if benchmark!='libero_pro':raise ValueError('RoboTwin index awaits a verified native control adapter')
    if destination.exists():raise FileExistsError('Choose a fresh benchmark index directory')
    source_status=json.loads((source/'status.json').read_text())
    if not diagnostic and not source_status.get('ready_for_training'):raise ValueError('Source catalog is not released')
    norm=json.loads((source/'normalization.json').read_text())
    if norm['catalog_episodes_sha256']!=sha(source/'episodes.jsonl'):raise ValueError('Source normalization mismatch')
    episodes=[]
    for line in (source/'episodes.jsonl').open():
        row=json.loads(line)
        if row['domain'] in DOMAINS:
            if not row['action_supervision_available']:raise ValueError('LIBERO controls must be independently verified')
            episodes.append(row)
    if not episodes:raise ValueError('No eligible LIBERO episodes')
    ids={row['episode_id'] for row in episodes}
    if len(ids)!=len(episodes):raise ValueError('Duplicate source episode')
    counts={split:sum(row['split']==split for row in episodes) for split in ['train','validation','test']}
    if not all(counts.values()):raise ValueError('Every retained source split must be nonempty')
    present={row['domain'] for row in episodes}
    if not diagnostic and present!=DOMAINS:raise ValueError('All four original LIBERO suites are required')
    catalog=destination/'catalog';catalog.mkdir(parents=True)
    (catalog/'episodes.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in episodes))
    for split in counts:
        with (source/(split+'.jsonl')).open() as inp,(catalog/(split+'.jsonl')).open('w') as out:
            for line in inp:
                row=json.loads(line)
                if row['episode_id'] in ids:out.write(line)
    derived={**norm,'by_domain':{key:value for key,value in norm['by_domain'].items() if key in present},
             'used_episode_ids':[uid for uid in norm['used_episode_ids'] if uid in ids],
             'catalog_episodes_sha256':sha(catalog/'episodes.jsonl'),
             'scope':'Unchanged source TRAIN statistics; original four LIBERO domains and source splits retained',
             'source_normalization_sha256':sha(source/'normalization.json'),'time':time.time()}
    dump(catalog/'normalization.json',derived)
    dump(catalog/'status.json',{'ready_for_training':not diagnostic,'diagnostic':diagnostic,
         'scope':'Derived original LIBERO demonstration subset; no benchmark evaluation data',
         'source_catalog':str(source),'source_status_sha256':sha(source/'status.json'),
         'source_episodes_sha256':sha(source/'episodes.jsonl'),'retained_split_counts':counts,
         'retained_domains':sorted(present),'source_split_preserved':True,'time':time.time()})
    result=build_index(catalog,destination/'index',diagnostic)
    result['purpose']={'kind':'benchmark_finetune','benchmark':benchmark}
    dump(destination/'index/index.json',result)
    return {'benchmark':benchmark,'diagnostic':diagnostic,'index_root':str(destination/'index'),
            'splits':{s:{k:v[k] for k in ['episodes','windows','action_episodes']} for s,v in result['splits'].items()}}

def main():
    p=argparse.ArgumentParser();p.add_argument('--source-catalog',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--benchmark',choices=['libero_pro','robotwin'],required=True)
    p.add_argument('--diagnostic',action='store_true');a=p.parse_args()
    print(json.dumps(derive(a.source_catalog,a.output,a.benchmark,a.diagnostic)))
if __name__=='__main__':main()
