"""All metadata episodes, grouped into resumable tar shards; no quota filtering."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import argparse, io, json, tarfile, time, traceback
from functools import lru_cache
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from PIL import Image
from prepare_data import numeric_table,exact_video_frames
from expand_dataset import metadata
from full_common import dump,sha,split_for,labels
pa.set_cpu_count(1)

ROOT=Path('/opt/atomvla')
CONTROL=ROOT/'expansions/full_v3';DATA=ROOT/'data/full_v3'

@lru_cache(maxsize=4)
def annotations(unit):
    p=Path(unit)/'meta/geniesim3_annotations.jsonl'
    if not p.exists():return {},None
    return {int(r['episode_index']):r for r in map(json.loads,p.read_text().splitlines())},sha(p)

def make_plan():
    source=Path('/opt/atomvla/dataset/lerobot_v3.0_v1')
    units=[('droid',source/'droid_1_0_1/chunk-000')]+[('geniesim_'+p.name,p) for p in sorted((source/'clean/geniesim3').iterdir()) if (p/'meta/info.json').exists()]
    old={}
    for p in (ROOT/'data/expanded_v2/episode_metadata').glob('*.json'):
        r=json.loads(p.read_text());old[r['episode_id']]=r['split']
    plans=[];inventory=[]
    for domain,unit in units:
        info=json.loads((unit/'meta/info.json').read_text());rows=metadata(unit)
        rows.sort(key=lambda r:(r['data/chunk_index'],r['data/file_index'],r['episode_index']))
        kind='real' if domain=='droid' else info['custom']['geniesim3']['provenance']
        inventory.append({'domain':domain,'root':str(unit),'episodes':len(rows),'domain_type':kind,'info_sha256':sha(unit/'meta/info.json'),
                          'no_instruction':sum(not any(str(x).strip() for x in r.get('tasks',[])) for r in rows)})
        for i in range(0,len(rows),32):
            sid=f'{domain}_{i//32:05d}'
            rows_part=[]
            for r in rows[i:i+32]:
                uid=f'{domain}__{int(r["episode_index"]):06d}'
                rows_part.append({'row':r,'uid':uid,'split':split_for(uid,old)})
            p=CONTROL/'plans'/f'{sid}.json'
            plan={'id':sid,'domain':domain,'source_root':str(unit),'info':info,'domain_type':kind,'entries':rows_part,'source_info_sha256':sha(unit/'meta/info.json')}
            dump(p,plan);plans.append(str(p))
    dump(CONTROL/'configs/lerobot_plan.json',{'shards':plans,'sources':inventory,'episodes':sum(x['episodes'] for x in inventory),'split_policy':'preserve expanded_v2; deterministic hash for all others'})
    return {'shards':len(plans),'sources':inventory}

def convert_shard(planpath):
    plan=json.loads(Path(planpath).read_text());sid=plan['id'];out=DATA/'shards/lerobot'/f'{sid}.tar';result=out.with_suffix('.json')
    if result.exists():
        prior=json.loads(result.read_text())
        if out.exists() and out.stat().st_size==prior['bytes']:return prior
        raise ValueError('Committed shard missing or truncated')
    out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix('.partial.tar');accepted=[];errors=[]
    unit=Path(plan['source_root']);info=plan['info'];fps=float(info['fps']);started=time.time()
    with tarfile.open(tmp,'w') as archive:
        for entry in plan['entries']:
            row=entry['row'];uid=entry['uid'];eid=int(row['episode_index'])
            try:
                path=unit/info['data_path'].format(chunk_index=row['data/chunk_index'],file_index=row['data/file_index'])
                table=numeric_table(str(path));d=table.filter(pc.equal(table['episode_index'],eid)).to_pydict();n=len(d['frame_index'])
                if n!=int(row['length']) or n<1:raise ValueError('source length mismatch or empty episode')
                fi=np.asarray(d['frame_index']);ts=np.asarray(d['timestamp'])
                if not np.all(np.diff(fi)==1) or not np.allclose(ts,fi/fps,atol=.005):raise ValueError('unsupported noncontiguous frame/time indexing')
                poses=np.asarray(d['observation.state.eef_pose'],np.float32).reshape(n,-1,7)
                sj=np.asarray(d['observation.state.joint_position'],np.float32);sg=np.asarray(d['observation.state.gripper'],np.float32)
                aj=np.asarray(d['action.joint_position'],np.float32);ag=np.asarray(d['action.gripper'],np.float32)
                state=np.concatenate([sj,sg],-1);action=np.concatenate([aj,ag],-1)
                copied=float(np.all(sj==aj,axis=-1).mean());action_ok=plan['domain']=='droid' and copied<1
                cameras=[k for k in info['features'] if k.startswith('observation.images.')]
                locators=[]
                for cam in cameras:
                    pre='videos/'+cam;vp=unit/info['video_path'].format(video_key=cam,chunk_index=row[pre+'/chunk_index'],file_index=row[pre+'/file_index'])
                    end=float(row[pre+'/to_timestamp']);start=float(row[pre+'/from_timestamp'])
                    if start+ts[-1]>end+.55/fps:raise ValueError('source video interval does not cover numeric timestamps')
                    locators.append({'camera':cam,'source_video':str(vp),'video_from_timestamp':start,'video_to_timestamp':end,'video_time_formula':'video_from_timestamp + original_numeric_timestamp'})
                def images(indices):
                    found={}
                    for loc in locators:
                        times=[loc['video_from_timestamp']+float(ts[t]) for t in indices]
                        frames=exact_video_frames(loc['source_video'],times,fps);values={}
                        for t,seconds in zip(indices,times):
                            b=io.BytesIO();frames[seconds][0].save(b,format='PNG')
                            values[t]=(b.getvalue(),{'source_video':loc['source_video'],'source_frame_index':int(fi[t]),'requested_timestamp':seconds,'decoded_timestamp':frames[seconds][1]})
                        found[loc['camera'].split('.')[-1]]=values
                    return found
                instruction=' '.join(str(x).strip() for x in row.get('tasks',[]) if str(x).strip())
                meta={'source_episode_index':eid,'domain':plan['domain'],'domain_type':plan['domain_type'],'split':entry['split'],
                      'instruction':instruction,'fps':fps,'source_root':str(unit),'source_numeric_file':str(path),'source_numeric_sha256':sha(path),
                      'source_info_sha256':plan['source_info_sha256'],'source_episode_metadata':row,'source_image_locators':locators,
                      'numeric_frame_start':int(fi[0]),'full_original_episode_coverage':bool(fi[0]==0),'source_available_rows_preserved':True,
                      'action_supervision_available':action_ok,'action_representation':'native_absolute_joint_target_plus_gripper' if action_ok else 'unverified_same_frame_readback',
                      'joint_action_equals_state_fraction':copied,'action_exclusion_reason':None if action_ok else 'control targets not verified; same-frame state readback',
                      'pose_convention':'canonical source eef_pose xyzw; endpoint axes standardized by source coordinate_plan; parent origins retained'}
                ann,ann_sha=annotations(str(unit))
                if eid in ann:
                    source_ann=ann[eid]
                    meta['source_annotation']=source_ann;meta['source_annotations_sha256']=ann_sha
                    meta['original_episode_identity']=source_ann.get('h5_path')
                    aligned=[]
                    for segment in source_ann.get('instruction_segments',[]):
                        first=max(0,int(segment['start_frame_index']));last=min(n-1,int(segment['end_frame_index'])-1)
                        if last<first:continue
                        displacement=poses[last,:,:3]-poses[first,:,:3];length=np.linalg.norm(displacement,axis=-1);valid=length>.001
                        aligned.append({'source_segment':segment,'start_row':first,'end_row_inclusive':last,'endpoint_pose_xyzw':poses[last].tolist(),
                                        'displacement_m':displacement.tolist(),'direction':np.where(valid[:,None],displacement/np.maximum(length[:,None],1e-8),0).tolist(),
                                        'direction_valid':valid.tolist(),'semantic_atom_verified':False,'alignment_source':'source_dataset_instruction_segment'})
                    meta['source_instruction_segments_with_motion']=aligned
                # Build one episode before committing, so a failed image cannot leave an apparent complete record.
                buffer=io.BytesIO()
                with tarfile.open(fileobj=buffer,mode='w') as episode_tar:
                    record=labels(episode_tar,uid,poses,state,action,sg,fi,ts,meta,images)
                buffer.seek(0)
                with tarfile.open(fileobj=buffer,mode='r') as ep:
                    for member in ep:archive.addfile(member,ep.extractfile(member))
                accepted.append(record)
            except Exception as e:errors.append({'episode_id':uid,'error':str(e),'traceback':traceback.format_exc()})
            dump(CONTROL/'worker_progress'/f'{sid}.json',{'time':time.time(),'completed':len(accepted),'failed':len(errors),'current':uid})
    tmp.replace(out)
    record={'id':sid,'tar':str(out),'sha256':sha(out),'bytes':out.stat().st_size,'episodes':accepted,'errors':errors,'seconds':time.time()-started,'attempted':len(plan['entries'])}
    dump(result,record);return record

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',action='store_true');p.add_argument('--shard');a=p.parse_args()
    result=make_plan() if a.plan else convert_shard(a.shard)
    print(json.dumps({k:v for k,v in result.items() if k not in ['episodes','errors']},ensure_ascii=False))
