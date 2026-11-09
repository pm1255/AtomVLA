"""Independent incremental shard audit and canonical episode catalog.

Conversion is not training readiness. Catalog masks supersede archived fields.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,ast,hashlib,io,json,sqlite3,tarfile,time,traceback
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from torch.utils.tensorboard import SummaryWriter
from full_common import dump,sha
from full_lerobot import ROOT,CONTROL,DATA

def arr(tar,member):
    with np.load(io.BytesIO(tar.extractfile(member).read()),allow_pickle=False) as z:return {k:z[k] for k in z.files}

def inspect_shard(path):
    summary=json.loads(path.read_text());source=Path(summary['tar'])
    assert source.stat().st_size==summary['bytes'] and sha(source)==summary['sha256'],'committed shard size/SHA mismatch'
    out=[];errors=[]
    with tarfile.open(source) as tar:
        names=tar.getnames();assert len(names)==len(set(names)),'duplicate tar members'
        for expected in summary['episodes']:
            uid=expected['episode_id']
            try:
                meta=json.load(tar.extractfile(uid+'/metadata.json'));assert meta==expected
                ep=arr(tar,uid+'/trajectory.npz');poses=ep['poses_xyzw'];n=meta['frames'];arms=poses.shape[1]
                assert n==len(poses)==len(ep['state'])==len(ep['action'])
                for k,v in ep.items():assert np.isfinite(v).all(),f'nonfinite {k}'
                assert np.max(np.abs(np.linalg.norm(poses[...,3:],axis=-1)-1))<1e-5
                expected_d=np.diff(poses[...,:3],axis=0,prepend=poses[:1,...,:3]);np.testing.assert_allclose(expected_d,ep['displacement_m'],atol=1e-7)
                mag=np.linalg.norm(expected_d,axis=-1);mask=mag>.001;assert np.array_equal(mask,ep['movement_direction_valid'])
                wanted=np.where(mask[...,None],expected_d/np.maximum(mag[...,None],1e-8),0)
                np.testing.assert_allclose(wanted,ep['movement_direction_parent_frame'],atol=1e-6)
                windows=json.load(tar.extractfile(uid+'/windows.json'));segments=json.load(tar.extractfile(uid+'/program_segments.json'))
                for seg in segments:
                    s,e=seg['start_row'],seg['end_row_inclusive'];assert 0<=s<=e<n
                    np.testing.assert_allclose(seg['endpoint_pose_xyzw'],poses[e],atol=1e-7)
                    np.testing.assert_allclose(seg['displacement_m'],poses[e,:,:3]-poses[s,:,:3],atol=1e-7)
                    tree=ast.parse(seg['program']);assert all(isinstance(x,ast.Expr) and isinstance(x.value,ast.Call) and isinstance(x.value.func,ast.Name) and x.value.func.id in ['move_eef','set_gripper'] for x in tree.body)
                fingerprint=hashlib.sha256(meta['trajectory_content_sha256'].encode())
                for row in windows:
                    t,future=row['frame_index'],row['future_frame_index'];assert 0<=t<future<n
                    target=arr(tar,row['pose_target_member']);indices=target['future_indices'];assert indices.min()>t and indices.max()==future
                    np.testing.assert_allclose(target['position_delta_m'][:arms],(poses[indices,:,:3]-poses[t,:,:3]).transpose(1,0,2),atol=1e-6)
                    for arm in range(arms):
                        x=target['rotation_6d'][arm];col1=x[:,:3];col2=x[:,3:];mat=np.stack([col1,col2,np.cross(col1,col2)],axis=-1)
                        actual=(Rotation.from_quat(poses[t,arm,3:]).inv()*Rotation.from_quat(poses[indices,arm,3:])).as_matrix()
                        np.testing.assert_allclose(mat,actual,atol=2e-6)
                    for key,hashkey in [('images','image_sha256'),('future_images','future_image_sha256')]:
                        assert len(row[key])==len(row[hashkey])>0
                        for loc,digest in zip(row[key],row[hashkey]):
                            raw=tar.extractfile(loc['member']).read();assert hashlib.sha256(raw).hexdigest()==digest
                            fingerprint.update(digest.encode())
                            with Image.open(io.BytesIO(raw)) as im:assert im.size==(224,224) and im.mode=='RGB';im.load()
                # Include environment/task identity to avoid merging the same motion in different scenes.
                fingerprint.update(str(meta.get('task',meta.get('source_group',meta['domain']))).encode())
                action_ok=meta['action_supervision_available'];offset=0;reason=None
                if meta.get('source_family')=='libero':
                    # Creation code records observation AFTER env.step(actions[j]).
                    # Consumers must index actions[t+1] and mask beyond the episode.
                    offset=1;reason='post_action_observation_source_contract; action[t+1] predicts from observation[t]'
                    assert meta.get('action_row_offset')==1,'LIBERO action alignment annotation missing'
                    aperture=np.sum(np.abs(ep['state'][:,-2:]),axis=-1,keepdims=True)
                    np.testing.assert_allclose(ep['gripper'],aperture,atol=1e-7)
                item={'episode_id':uid,'tar':str(source),'metadata_member':uid+'/metadata.json','windows_member':uid+'/windows.json','domain':meta['domain'],
                      'domain_type':meta['domain_type'],'split':meta['split'],'frames':n,'windows':len(windows),'fingerprint':fingerprint.hexdigest(),
                      'instruction_available':bool(meta['instruction']),'action_supervision_available':action_ok,'action_row_offset':offset,
                      'action_alignment_note':reason,'timebase_verified':bool(meta.get('fps')),'flow_labels_ready':False,'audit_passed':True}
                out.append(item)
            except Exception as e:errors.append({'episode_id':uid,'error':str(e),'traceback':traceback.format_exc()})
    return {'shard':summary['id'],'source_sha256':summary['sha256'],'episodes':out,'errors':errors,'passed':not errors,'time':time.time()}

def main():
    p=argparse.ArgumentParser();p.add_argument('--follow-hours',type=float,default=0);a=p.parse_args()
    audit=CONTROL/'audit';audit.mkdir(exist_ok=True);db=sqlite3.connect(audit/'catalog.sqlite');db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS episodes (episode_id TEXT PRIMARY KEY,fingerprint TEXT,split TEXT,json TEXT)')
    db.execute('CREATE INDEX IF NOT EXISTS fingerprint_index ON episodes(fingerprint)')
    writer=SummaryWriter(str(ROOT/'tensorboard/full_v3/quality_audit'));start=time.time();attempted=set()
    while True:
        for path in sorted((DATA/'shards').glob('*/*.json')):
            if path.name.startswith('smoke_'):continue
            record=audit/(path.stem+'.json')
            if record.exists() or str(path) in attempted:continue
            attempted.add(str(path))
            try:r=inspect_shard(path)
            except Exception as e:r={'shard':path.stem,'passed':False,'errors':[{'error':str(e),'traceback':traceback.format_exc()}],'episodes':[]}
            if r['passed']:
                for ep in r['episodes']:db.execute('INSERT OR REPLACE INTO episodes VALUES (?,?,?,?)',(ep['episode_id'],ep['fingerprint'],ep['split'],json.dumps(ep)))
                db.commit()
            dump(record,r)
            total=db.execute('SELECT COUNT(*) FROM episodes').fetchone()[0]
            unique=db.execute('SELECT COUNT(DISTINCT fingerprint) FROM episodes').fetchone()[0]
            conflict=db.execute('SELECT COUNT(*) FROM (SELECT fingerprint FROM episodes GROUP BY fingerprint HAVING COUNT(DISTINCT split)>1)').fetchone()[0]
            status={'phase':'incremental_quality_audit','updated_at':time.time(),'audited_episodes':total,'exact_fingerprints':unique,'duplicate_groups_crossing_splits':conflict,'latest_shard':path.stem,'latest_passed':r['passed'],
                    'catalog':'catalog.sqlite','not_training_ready':'full conversion, canonical dedup export, normalization and optical-flow labels still pending'}
            dump(audit/'progress.json',status)
            for tag,val in [('episodes/audited',total),('episodes/unique_fingerprints',unique),('dedup/cross_split_groups',conflict),('checks/latest_shard_passed',int(r['passed']))]:writer.add_scalar(tag,val,int(time.time()-start))
            writer.flush()
            if not r['passed']:
                dump(audit/'STOP_REVIEW_REQUIRED.json',r);writer.close();db.close();raise SystemExit(2)
        if not a.follow_hours or time.time()-start>a.follow_hours*3600:break
        time.sleep(30)
    writer.close();db.close()
if __name__=='__main__':main()
