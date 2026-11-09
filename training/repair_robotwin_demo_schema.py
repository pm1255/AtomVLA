"""One bounded repair of the verified v1.0 schema; keep old diagnostics and raw data."""
import io,json,shutil,sqlite3,tarfile,time,zipfile
from pathlib import Path
import h5py,numpy as np
from full_common import dump,sha
from full_lerobot import ROOT,CONTROL,DATA
from full_simulation import convert_episode,run
from full_audit import inspect_shard,arr

sid='robotwin__dataset__click_bell__demo_clean'
control=CONTROL/'robotwin_schema_repair_001';control.mkdir(exist_ok=True)
def progress(phase):dump(control/'evidence/progress.json',{'phase':phase,'time':time.time()})
progress('diagnostic')
receipt=DATA/'shards/robotwin'/(sid+'.json');old=json.loads(receipt.read_text())
assert not old['episodes'] and len(old['errors'])==50,'Only this wholly failed source may be replaced'
submit=json.loads((CONTROL/'evidence/robotwin_submission.json').read_text());cmd=Path('/proc')/str(submit['pid'])/'cmdline'
assert not(cmd.exists() and cmd.read_bytes()),'Source runner is still active'
backup=CONTROL/'repair_archive/robotwin_demo_schema_v1';backup.mkdir(parents=True,exist_ok=False)
for source in [receipt,Path(old['tar']),CONTROL/'audit'/(sid+'.json'),CONTROL/'robotwin/exit.json',CONTROL/'robotwin/progress.json']:
    if source.exists():shutil.copy2(source,backup/(source.parent.name+'__'+source.name))
dump(backup/'diagnostic.json',{'time':time.time(),'prior_submission':submit,'old_receipt_sha256':sha(receipt),
    'scope':sid,'source_schema':'RoboTwin native exporter v1.0; source proof pinned in converted metadata','preserves_original_source_archive':True})
plans=json.loads((CONTROL/'configs/simulation_plan.json').read_text())['robotwin'];pp=next(Path(p) for p in plans if Path(p).stem==sid);item=json.loads(pp.read_text())
src=DATA/'sources'/'TianxingChen__RoboTwin2.0'/item['item']['path'];assert sha(src)==item['item']['lfs']['oid']
records=[];sample_tar=control/'sample.tar';progress('two_episode_diagnostic')
with zipfile.ZipFile(src) as z,tarfile.open(sample_tar,'w') as tar:
    names=sorted(n for n in z.namelist() if n.endswith('.hdf5') and '/data/' in n)
    for name in [names[0],names[-1]]:
        uid=sid+'__'+Path(name).stem
        meta={'source_family':'robotwin','domain':'robotwin_demo_clean','domain_type':'simulation','split':'train','fps':None,'instruction':'',
              'source_repo':item['repo'],'source_revision':item['revision'],'source_member':name,'task':'click_bell','robot_variant':'demo_clean'}
        with h5py.File(io.BytesIO(z.read(name)),'r') as h:records.append(convert_episode(h,uid,meta,tar,src,member=name))
summary={'id':'diagnostic_schema_sample','tar':str(sample_tar),'sha256':sha(sample_tar),'bytes':sample_tar.stat().st_size,'episodes':records,'errors':[]}
sample_receipt=sample_tar.with_suffix('.json');dump(sample_receipt,summary)
audit=inspect_shard(sample_receipt);assert audit['passed'] and len(audit['episodes'])==2,audit
with zipfile.ZipFile(src) as z,tarfile.open(sample_tar) as tar:
    for meta in records:
        with h5py.File(io.BytesIO(z.read(meta['source_member'])),'r') as h:
            actual=arr(tar,meta['episode_id']+'/trajectory.npz')
            expected=np.stack([h['state/left_ee_poses'][()],h['state/right_ee_poses'][()]],axis=1)[...,[0,1,2,4,5,6,3]]
            np.testing.assert_allclose(actual['poses_xyzw'],expected,atol=2e-6)
            raw=arr(tar,meta['episode_id']+'/source_numeric.npz')
            for group in ['state','action']:
                for key in h[group]:np.testing.assert_array_equal(raw[group+'__'+key],h[group][key][()])
            assert actual['action'].shape[1]==0 and not meta['action_supervision_available'] and meta['fps'] is None
dump(control/'evidence/sample_audit.json',audit)
progress('replace_failed_empty_output')
# All failed payloads are backed up above. The original source ZIP is untouched.
receipt.unlink();Path(old['tar']).unlink()
result=run(item['item'],item['repo'],item['revision']);assert len(result['episodes'])==50 and not result['errors'],result['errors']
progress('independent_full_shard_audit')
audit=inspect_shard(receipt);assert audit['passed'] and len(audit['episodes'])==50,audit
db=sqlite3.connect(CONTROL/'audit/catalog.sqlite',timeout=30)
for ep in audit['episodes']:db.execute('INSERT OR REPLACE INTO episodes VALUES (?,?,?,?)',(ep['episode_id'],ep['fingerprint'],ep['split'],json.dumps(ep)))
db.commit();db.close();dump(CONTROL/'audit'/(sid+'.json'),audit)
out={'passed':True,'source':sid,'episodes':50,'independently_audited_episodes':50,'sample_numeric_roundtrip_episodes':2,
     'source_unchanged_sha256':sha(src),'backup':str(backup),'action_supervision':False,'timebase_verified':False,'time':time.time()}
dump(control/'evidence/DONE.json',out);progress('completed');print(json.dumps(out))
