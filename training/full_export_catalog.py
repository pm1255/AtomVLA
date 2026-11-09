"""Snapshot canonical catalog: source identity + exact content, no READY for previews."""
import hashlib,json,sqlite3,tarfile,time
from collections import Counter,defaultdict
from pathlib import Path
from full_common import dump,sha,split_for
from full_lerobot import ROOT,CONTROL,DATA


def identity(meta):
    if meta.get('original_episode_identity'):
        return 'geniesim_original:'+str(meta['original_episode_identity'])
    family=meta.get('source_family')
    if family in ('libero','robotwin'):
        return family+':'+meta['source_sha256']+':'+str(meta.get('source_group',meta.get('source_member')))
    return 'lerobot:'+meta['source_root']+':'+str(meta['source_episode_index'])


def group_records(records,old):
    parent=list(range(len(records)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def join(i,j):
        i,j=find(i),find(j)
        if i!=j:parent[j]=i
    seen={}
    for i,r in enumerate(records):
        for key in [('identity',r['source_identity']),('content',r['content_key'])]:
            if key in seen:join(i,seen[key])
            else:seen[key]=i
    groups=defaultdict(list)
    for i,r in enumerate(records):groups[find(i)].append(r)
    canonical=[];duplicates=[];conflicts=[]
    for group in groups.values():
        pinned=[r for r in group if r['episode_id'] in old]
        prior_splits={old[r['episode_id']] for r in pinned}
        if len(prior_splits)>1:
            conflicts.append({'reason':'conflicting_prior_splits','episodes':[r['episode_id'] for r in group]});continue
        if len({r['content_key'] for r in group})>1:
            conflicts.append({'reason':'same_original_identity_different_content_requires_review','episodes':[r['episode_id'] for r in group]});continue
        chosen=dict(min(pinned or group,key=lambda r:(not r['instruction_available'],r['episode_id'])))
        chosen['conversion_split']=chosen['split']
        chosen['split']=next(iter(prior_splits)) if prior_splits else split_for(chosen['content_key'],{})
        chosen['canonical_group_size']=len(group)
        canonical.append(chosen)
        if len(group)>1:duplicates.append({'canonical':chosen['episode_id'],'content_key':chosen['content_key'],'episodes':[r['episode_id'] for r in group]})
    return canonical,duplicates,conflicts


def export():
    audit=CONTROL/'audit';db=sqlite3.connect(audit/'catalog.sqlite')
    records=[json.loads(r[0]) for r in db.execute('SELECT json FROM episodes ORDER BY episode_id')];db.close()
    old={}
    for p in (ROOT/'data/expanded_v2/episode_metadata').glob('*.json'):
        d=json.loads(p.read_text());old[d['episode_id']]=d['split']
    bytar=defaultdict(list)
    for r in records:bytar[r['tar']].append(r)
    # Hashes refer only to shards accepted by the independent conversion audit.
    for tarpath,group in sorted(bytar.items()):
        receipt=json.loads(Path(tarpath).with_suffix('.json').read_text())
        audit_record=json.loads((audit/(receipt['id']+'.json')).read_text())
        assert audit_record['passed'] and audit_record['source_sha256']==receipt['sha256']
        with tarfile.open(tarpath) as tar:
            for r in group:
                meta=json.load(tar.extractfile(r['metadata_member']));rows=json.load(tar.extractfile(r['windows_member']))
                family=meta.get('source_family') or ('droid' if meta['domain']=='droid' else 'geniesim')
                # Do not include demo index or archive name: repackaged exact episodes should match.
                h=hashlib.sha256((family+':'+meta['trajectory_content_sha256']).encode())
                for row in rows:
                    for key in ['image_sha256','future_image_sha256']:
                        for digest in row[key]:h.update(digest.encode())
                r.update(source_family=family,source_identity=identity(meta),content_key=h.hexdigest(),source_shard_sha256=receipt['sha256'])
    canonical,duplicates,conflicts=group_records(records,old)
    checked_flows={}
    for p in CONTROL.glob('flow_batch_*/evidence/flow_audit.json'):
        d=json.loads(p.read_text())
        if d.get('passed'):
            for x in d['shard_checks']:checked_flows[x['id']]=x
    out=DATA/'catalog_preview';out.mkdir(exist_ok=True)
    split_ep=Counter();samples=Counter();domains=Counter();sources=Counter();flow_present=flow_audited=0
    streams={s:(out/(s+'.jsonl.tmp')).open('w') for s in ['train','validation','test']}
    try:
        with (out/'episodes.jsonl.tmp').open('w') as eps:
            bytar=defaultdict(list)
            for r in canonical:bytar[r['tar']].append(r)
            for tarpath,group in sorted(bytar.items()):
                sid=Path(tarpath).stem;flow_receipt=DATA/'flow_shards'/(sid+'.json');present=checked=False
                if flow_receipt.exists():
                    f=json.loads(flow_receipt.read_text());present=f['source_sha256']==group[0]['source_shard_sha256']
                    checked=present and sid in checked_flows and checked_flows[sid]['flow_sha256']==f['sha256']
                with tarfile.open(tarpath) as tar:
                    for entry in group:
                        rows=json.load(tar.extractfile(entry['windows_member']));entry['flow_labels_ready']=checked
                        eps.write(json.dumps(entry,ensure_ascii=False)+'\n');split_ep[entry['split']]+=1;domains[entry['domain_type']]+=1;sources[entry['source_family']]+=1
                        for row in rows:
                            row.update(tar=tarpath,split=entry['split'],audit_passed=True,source_shard_sha256=entry['source_shard_sha256'],source_identity=entry['source_identity'],
                                       action_row_offset=entry['action_row_offset'],action_supervision_available=entry['action_supervision_available'],
                                       flow_labels_present=present,flow_labels_ready=checked,flow_label_status='independently_audited' if checked else 'generated_pending_audit' if present else 'waiting')
                            streams[entry['split']].write(json.dumps(row,ensure_ascii=False)+'\n');samples[entry['split']]+=1
                            flow_present+=int(present);flow_audited+=int(checked)
    finally:
        for f in streams.values():f.close()
    for s in streams:(out/(s+'.jsonl.tmp')).replace(out/(s+'.jsonl'))
    (out/'episodes.jsonl.tmp').replace(out/'episodes.jsonl')
    dump(out/'dedup_details.json',{'duplicate_groups':duplicates,'conflicts':conflicts})
    report={'preview':True,'episodes_audited':len(records),'canonical_episodes':len(canonical),'exact_duplicate_groups':len(duplicates),'conflict_groups':len(conflicts),
            'duplicates_excluded':sum(len(g['episodes'])-1 for g in duplicates),'conflicting_episodes_excluded':sum(len(g['episodes']) for g in conflicts),
            'split_episodes':dict(split_ep),'split_windows':dict(samples),'domain_types':dict(domains),'by_source':dict(sources),'flow_present_windows':flow_present,'independently_audited_flow_windows':flow_audited,
            'episodes_with_rgb_windows':sum(r['windows']>0 for r in canonical),'episodes_without_rgb_windows':sum(r['windows']==0 for r in canonical),
            'action_supervision_episodes_with_rgb':sum(r['windows']>0 and r['action_supervision_available'] for r in canonical),
            'time':time.time(),'ready_for_training':False,'episodes_manifest_sha256':sha(out/'episodes.jsonl'),
            'split_policy':'source identity and exact numeric/RGB content union; preserve expanded_v2 splits; deterministic content hash for new groups; conflicting prior splits excluded for review',
            'dedup_scope':'exact observed content and recorded original identity; unrecorded identity and semantic near-duplicates not certified',
            'pending':['full source conversion and exclusions reconciliation','complete release QA','all flow labels and independent audit','train-only normalization for final catalog','native model input dimension adapter']}
    dump(out/'status.json',report);return report

if __name__=='__main__':print(json.dumps(export()))
