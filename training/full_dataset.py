"""Canonical, audited tar catalog reader. Incomplete flow fails explicitly by default."""
from functools import lru_cache
import hashlib,io,json,tarfile
from pathlib import Path
import numpy as np
from PIL import Image

TEACHER_SHA256='ff5fadd56d26b40647388883af1547351ea17868b765c05b27231e72dd16a322'

@lru_cache(maxsize=8)
def verified_flow_receipt(path,size,mtime_ns):
    path=Path(path);receipt=json.loads(path.with_suffix('.json').read_text())
    if size!=receipt['bytes']:raise ValueError('Flow shard size mismatch')
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    if h.hexdigest()!=receipt['sha256']:raise ValueError('Flow shard checksum mismatch')
    if receipt['teacher_sha256']!=TEACHER_SHA256:raise ValueError('Unexpected flow teacher')
    return receipt

@lru_cache(maxsize=4)
def episode_numeric(tarpath,uid):
    with tarfile.open(tarpath) as tar:
        with np.load(io.BytesIO(tar.extractfile(uid+'/trajectory.npz').read()),allow_pickle=False) as z:
            return {k:z[k] for k in ['poses_xyzw','state','action','gripper']}

class FullDataset:
    def __init__(self,root,split='train',stage=1,require_flow=True,catalog=None,allow_preview=False):
        self.root=Path(root);self.stage=stage;self.require_flow=require_flow
        catalog=Path(catalog) if catalog is not None else self.root/'catalog'
        if not allow_preview:
            status=json.loads((catalog/'status.json').read_text())
            if not status.get('ready_for_training'):raise RuntimeError('Dataset release is not training-ready; use allow_preview only for diagnostics')
        manifest=catalog/f'{split}.jsonl'
        self.rows=[json.loads(line) for line in manifest.open()]
        if stage==2:self.rows=[r for r in self.rows if r['action_supervision_available']]
        if not self.rows:raise ValueError('No audited samples in requested split/stage')

    def __len__(self):return len(self.rows)

    def __getitem__(self,index):
        row=self.rows[index];uid=row['episode_id'];ep=episode_numeric(row['tar'],uid);t=row['frame_index']
        with tarfile.open(row['tar']) as tar:
            meta=json.load(tar.extractfile(uid+'/metadata.json'))
            images=[]
            for loc,digest in zip(row['images'],row['image_sha256']):
                raw=tar.extractfile(loc['member']).read()
                if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('Input RGB hash mismatch')
                with Image.open(io.BytesIO(raw)) as im:images.append(im.convert('RGB'))
            with np.load(io.BytesIO(tar.extractfile(row['pose_target_member']).read()),allow_pickle=False) as z:
                pose=np.concatenate([z['position_delta_m']/.1,z['rotation_6d']],axis=-1);pose_mask=z['pose_mask'].copy()
        flowpath=self.root/'flow_shards'/(Path(row['tar']).stem+'.tar')
        if flowpath.exists():
            stat=flowpath.stat();receipt=verified_flow_receipt(str(flowpath),stat.st_size,stat.st_mtime_ns)
            source_sha=row.get('source_shard_sha256') or json.loads(Path(row['tar']).with_suffix('.json').read_text())['sha256']
            if receipt['source_sha256']!=source_sha:raise ValueError('Flow and source shard disagree')
            with tarfile.open(flowpath) as tar:
                with np.load(io.BytesIO(tar.extractfile(row['id']+'.npz').read()),allow_pickle=False) as z:
                    flow=z['flow'].copy();flow_mask=z['valid'].copy()
                    expected=[row['image_sha256'][0],row['future_image_sha256'][0]]
                    if z['input_image_sha256'].tolist()!=expected:raise ValueError('Flow uses a different image pair')
                    if str(z['teacher_sha256'])!=TEACHER_SHA256:raise ValueError('Unexpected per-label flow teacher')
            if flow.shape!=(2,224,224) or flow_mask.shape!=(1,224,224) or flow_mask.dtype!=np.bool_ or not np.isfinite(flow).all():
                raise ValueError('Invalid flow label or mask')
        elif self.require_flow:raise FileNotFoundError(f'Optical-flow supervision not ready: {flowpath}')
        else:flow=None;flow_mask=None
        offset=row.get('action_row_offset',meta.get('action_row_offset',0))
        action_index=np.arange(t+offset,t+offset+8);valid=(action_index<len(ep['action']))&row['action_supervision_available']
        # Native action dimensions/units differ by embodiment. They are not silently normalized together.
        action=ep['action'][np.minimum(action_index,len(ep['action'])-1)].copy()
        state=ep['state'][t].copy();pose_now=ep['poses_xyzw'][t].copy()
        horizon=f"{row['delta_seconds']:.4f} seconds" if row.get('delta_seconds') is not None else f"{row['future_frame_index']-t} recorded rows (sampling period unknown)"
        prompt=f"Task: {row['instruction']}\nRobot source: {row['domain']}\nCurrent EEF poses xyzw in native parent frame: {pose_now.tolist()}\nCurrent state: {state.tolist()}\nPredict the next {horizon}."
        return {'id':row['id'],'prompt':prompt,'images':images,'program':row['program'] if self.stage==1 else None,
                'pose':pose,'pose_mask':pose_mask,'flow':flow,'flow_mask':flow_mask,'action':action,'action_mask':np.broadcast_to(valid[:,None],action.shape).copy(),
                'state':state,'action_row_offset':offset,'action_representation':meta['action_representation'],
                'instruction_available':bool(row['instruction']),'timebase_verified':row.get('timebase_verified',True),'domain':row['domain']}
