"""Publish explicit weak-supervision stage-1 manifests after independent readback.
Never upgrades annotation quality. Excludes overlapping transport candidates,
unknown seeds, missing text, duplicate episodes, corrupted data and action loss.
"""
import argparse,collections,hashlib,json,os,tarfile,time
from pathlib import Path
import numpy as np
from common import digest,write_json,VERSION
from records import validate_annotation
from loader import load_episode,collate
POLICY='stage1_weak_events_v1'

def select_records(records):
 selected=[];reasons=collections.Counter()
 for r in records:
  a=validate_annotation(r['annotation'],r['targets']['terminal_row']+1)
  if r['schema']!=VERSION:raise ValueError('schema mismatch')
  if not r['inputs']['instruction'] or not r['provenance']['split_group_verified']:
   reasons['missing_instruction_or_seed']+=1;continue
  # Place already includes transport toward preplace. Do not count that same
  # interval twice as move supervision with a competing program name.
  if a['verb']=='move' and any(o['annotation']['verb']=='place' and
      o['annotation']['arguments']['arm']==a['arguments']['arm'] and
      o['annotation']['start']<=a['start'] and o['annotation']['end']>=a['end'] for o in records):
   reasons['transport_subsumed_by_place']+=1;continue
  if not r['targets']['keyframes']:raise ValueError('empty future')
  if not all(r['inputs']['current_row']<e['row']<=r['targets']['terminal_row'] for e in r['targets']['keyframes']):raise ValueError('future leakage')
  if r['targets']['action_supervision_available']:raise ValueError('unexpected action contract')
  selected.append(r)
 # On the same arm, accepted primitives must not overlap. Anchors of one
 # primitive are legitimate distinct observations, not distinct primitives.
 unique={(r['annotation']['start'],r['annotation']['end'],r['annotation']['arguments']['arm'],r['annotation']['verb']) for r in selected}
 for arm in ['left','right']:
  spans=sorted((s,e,v) for s,e,a,v in unique if a in (arm,'both'))
  if any(b[0]<a[1] for a,b in zip(spans,spans[1:])):raise ValueError('overlapping primitive labels')
 return selected,reasons

def publish(root,destination):
 root=Path(root);destination=Path(destination)
 if destination.exists():raise ValueError('release destination must be new and immutable')
 temp=destination.with_name(destination.name+'.partial');temp.mkdir(parents=True,exist_ok=False)
 counts=collections.Counter();tasks=collections.Counter();variants=collections.Counter();verbs=collections.Counter();reject=collections.Counter();seen={};groups={};ids=set();details=[]
 handles={s:(temp/(s+'.jsonl')).open('w') for s in ['train','validation','test']}
 started=time.time()
 try:
  receipts=sorted((root/'episodes').glob('*/*/receipt.json'))
  for receipt in receipts:
   counts['scanned_episodes']+=1
   try:
    d=json.loads(receipt.read_text());path=receipt.parent/'sample.tar'
    if digest(path)!=d['tar_sha256']:raise ValueError('tar checksum mismatch')
    with tarfile.open(path) as tar:
     records=[json.loads(x) for x in tar.extractfile('records.jsonl')]
     chosen,reason=select_records(records);reject.update(reason)
     if not chosen:reject['no_accepted_records']+=1;continue
     refs={ref['member']:ref for r in chosen for frame in [r['inputs']['images']]+[e['images'] for e in r['targets']['keyframes']] for ref in frame.values()}
     for name,ref in refs.items():
      if hashlib.sha256(tar.extractfile(name).read()).hexdigest()!=ref['sha256']:raise ValueError('image checksum mismatch')
     # Content signature excludes archive/member identity. Identical image,
     # state, and label sequences from duplicate packs count only once.
     signature=hashlib.sha256(json.dumps([{'inputs':{k:v for k,v in r['inputs'].items() if k!='images'},'images':{c:v['sha256'] for c,v in r['inputs']['images'].items()},'annotation':r['annotation'],'pose':r['targets']['terminal_pose_xyzw'],'future':[{c:v['sha256'] for c,v in e['images'].items()} for e in r['targets']['keyframes']]} for r in chosen],sort_keys=True).encode()).hexdigest()
     if signature in seen:reject['duplicate_content_episode']+=1;continue
     if len({r['split'] for r in chosen})!=1:raise ValueError('episode split conflict')
     for r in chosen:
      if r['id'] in ids:raise ValueError('duplicate record ID')
      if groups.get(r['split_group'],r['split'])!=r['split']:raise ValueError('seed split leakage')
    # Decode through the actual training loader, not only JSON verification.
    sample=next(load_episode(path,allow_silver=True),None)
    if sample is None:raise ValueError('loader returned no sample')
    batch=collate([sample])
    if batch['terminal_pose'].shape!=(1,2,7):raise ValueError('loader pose shape')
    seen[signature]=str(path);counts['accepted_episodes']+=1
    for r in chosen:
     groups[r['split_group']]=r['split'];ids.add(r['id'])
     entry={'tar':str(path),'tar_sha256':d['tar_sha256'],'id':r['id'],'episode_id':r['episode_id'],'split':r['split'],
      'policy':POLICY,'annotation_quality':r['quality'],'supervision':'automatic_event_labels_not_physical_success_verification',
      'loss_weights':{'program':0.5 if r['quality']=='silver' else 1.0,'terminal_pose':1.0,'future_rgb':0.5 if r['quality']=='silver' else 1.0,'action':0.0},
      'coordinate_frame':r['coordinate_frame'],'robot_variant':r['robot_variant']}
     handles[r['split']].write(json.dumps(entry)+'\n');counts[r['split']+'_records']+=1;counts['records']+=1
     tasks[r['task']]+=1;variants[r['robot_variant']]+=1;verbs[r['annotation']['verb']]+=1
   except Exception as e:
    counts['error_episodes']+=1;details.append({'receipt':str(receipt),'error':str(e)})
   if counts['scanned_episodes']%100==0:write_json(temp/'progress.json',{'counts':dict(counts),'elapsed_seconds':time.time()-started})
 finally:
  for f in handles.values():f.flush();os.fsync(f.fileno());f.close()
 report={'policy':POLICY,'schema':VERSION,'root':str(root),'counts':dict(counts),'tasks':dict(tasks),'variants':dict(variants),'verbs':dict(verbs),
  'excluded':dict(reject),'errors':details,'created_at':time.time(),'elapsed_seconds':time.time()-started,
  'stage1_weak_supervision_ready':bool(counts['train_records']),'gold_verified_release':False,'stage2_action_ready':False,
  'limitations':['Automatic semantic events; attachment/support/success unverified.','Native RoboTwin world EEF coordinates; batch separately by embodiment.','Task source collector revision alignment unverified.','Only supported single-object pick/move/place schemas; complex tasks withheld.','No executable action target.'],
  'manifest_sha256':{s:digest(temp/(s+'.jsonl')) for s in handles}}
 write_json(temp/'report.json',report)
 lines=['# RoboTwin 第一阶段自动标注数据集','',f"已发布 {counts['accepted_episodes']} 条轨迹、{counts['records']} 条训练记录。适用范围：程序、终点位姿和未来图像的弱监督预训练；不包含已验证的控制动作。",'',
  f"划分：训练 {counts['train_records']}，验证 {counts['validation_records']}，测试 {counts['test_records']}。",'',
  '程序/图像默认损失权重 0.5，位姿 1.0，动作 0；这是初始配置，不是已验证的最优超参数。',
  '自动事件标签尚未核验抓稳、支撑和物理成功；不得用于报告任务成功率。',
  f"结构或内容异常隔离 {counts['error_episodes']} 条轨迹。完整分布、排除原因和错误明细见 report.json。"]
 (temp/'report_summary.md').write_text('\n'.join(lines)+'\n')
 os.replace(temp,destination)
 return report

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(publish(a.root,a.output)))
