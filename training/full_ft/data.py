"""Indexed immutable JSONL, RGB future targets, bounded tar caches and native normalization."""
from collections import OrderedDict
import hashlib,io,json,os,re,tarfile
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset,Sampler
from full_common import sha,dump

REGISTRY={'droid':{'state_dim':8,'action_dim':8,'representation':'native_absolute_joint_target_plus_gripper'},
          'libero':{'state_dim':9,'action_dim':7,'representation':'libero_OSC_POSE_normalized_delta_xyz_axisangle_gripper'}}

def family(domain):
    return 'droid' if domain=='droid' else 'libero' if domain.startswith('libero_') else 'robotwin' if domain.startswith('robotwin_') else 'geniesim' if domain.startswith('geniesim_') else domain

def task_key(row):
    fam=family(row['domain']);uid=row['episode_id'];parts=uid.split('__')
    if fam=='robotwin' and len(parts)>3:task=parts[2]
    elif fam=='libero' and len(parts)>3:task='__'.join(parts[1:-1])
    elif fam=='geniesim':task=row['domain']
    else:task=row.get('instruction') or row['domain']
    return fam+':'+hashlib.sha256(task.encode()).hexdigest()[:16]

def build_index(catalog,destination,diagnostic=False):
    catalog=Path(catalog).resolve();destination=Path(destination).resolve()
    if destination.exists():raise FileExistsError('Choose a new immutable training index directory')
    status=json.loads((catalog/'status.json').read_text())
    if not diagnostic and not status.get('ready_for_training'):raise RuntimeError('Dataset release not ready')
    norm=json.loads((catalog/'normalization.json').read_text());ep_sha=sha(catalog/'episodes.jsonl')
    if norm['catalog_episodes_sha256']!=ep_sha:raise ValueError('Normalization and episode catalog differ')
    episodes={};identities={};contents={}
    for line in (catalog/'episodes.jsonl').open():
        r=json.loads(line);uid=r['episode_id']
        if uid in episodes:raise ValueError('Duplicate canonical episode ID')
        for table,key in [(identities,r['source_identity']),(contents,r['content_key'])]:
            if key in table:raise ValueError('Canonical identity/content duplicate')
            table[key]=r['split']
        episodes[uid]=r
    train_ids={uid for uid,r in episodes.items() if r['split']=='train' and r['action_supervision_available']}
    if not set(norm['used_episode_ids'])<=train_ids:raise ValueError('Normalization includes non-train or invalid actions')
    required={uid for uid in train_ids if episodes[uid]['frames']>episodes[uid]['action_row_offset']}
    if set(norm['used_episode_ids'])!=required:raise ValueError('Incomplete training normalization')
    destination.mkdir(parents=True);dump(destination/'normalization.json',norm)
    metadata_cache=OrderedDict()
    result={'catalog':str(catalog),'diagnostic':diagnostic,'ready_for_training':not diagnostic,'episode_catalog_sha256':ep_sha,
            'normalization_sha256':sha(destination/'normalization.json'),'registry':REGISTRY,'splits':{}}
    for split in ['train','validation','test']:
        offsets=[];episode_num=[];group_ids=[];robot_ids=[];groups={};group_names=[];ep_ids=[];ep_map={};seen_rows=set()
        path=catalog/(split+'.jsonl')
        with path.open('rb') as f:
            while True:
                offset=f.tell();line=f.readline()
                if not line:break
                r=json.loads(line);uid=r['episode_id'];ep=episodes[uid]
                if r['split']!=split or ep['split']!=split or not r['audit_passed']:raise ValueError('Split/audit contract violated')
                if r['source_shard_sha256']!=ep['source_shard_sha256']:raise ValueError('Unpinned source shard')
                if r['id'] in seen_rows:raise ValueError('Duplicate window ID')
                seen_rows.add(r['id'])
                if uid not in ep_map:ep_map[uid]=len(ep_ids);ep_ids.append(uid)
                key=task_key(r)
                if key not in groups:groups[key]=len(groups);group_names.append(key)
                dom=r['domain'];fam=family(dom);robot=-1
                if r['action_supervision_available']:
                    if fam not in REGISTRY:raise ValueError(f'Unimplemented action adapter: {fam}')
                    stat=norm['by_domain'][dom];spec=REGISTRY[fam]
                    if len(stat['action_mean'])!=spec['action_dim'] or len(stat['state_mean'])!=spec['state_dim']:raise ValueError('Native dimension mismatch')
                    representation=r.get('action_representation')
                    if representation is None:
                        tarpath=r['tar']
                        if tarpath not in metadata_cache:
                            if sha(tarpath)!=ep['source_shard_sha256']:raise ValueError('Metadata fallback source checksum mismatch')
                            with tarfile.open(tarpath) as tar:
                                metadata_cache[tarpath]={m.name.split('/')[0]:json.load(tar.extractfile(m)) for m in tar if m.name.endswith('/metadata.json')}
                            if len(metadata_cache)>8:metadata_cache.popitem(last=False)
                        metadata_cache.move_to_end(tarpath)
                        representation=metadata_cache[tarpath][uid]['action_representation']
                    if representation!=spec['representation']:raise ValueError('Native action semantics mismatch')
                    robot=list(REGISTRY).index(fam)
                offsets.append(offset);episode_num.append(ep_map[uid]);group_ids.append(groups[key]);robot_ids.append(robot)
        if not offsets:raise ValueError(f'No eligible {split} windows')
        arrays={'offsets':np.array(offsets,dtype=np.int64),'episode':np.array(episode_num,dtype=np.int32),
                'group':np.array(group_ids,dtype=np.int32),'robot':np.array(robot_ids,dtype=np.int16)}
        hashes={}
        for name,arr in arrays.items():
            target=destination/f'{split}_{name}.npy';np.save(target,arr);hashes[name]=sha(target)
        ep_path=destination/f'{split}_episodes.json';dump(ep_path,ep_ids)
        result['splits'][split]={'manifest':str(path),'manifest_sha256':sha(path),'manifest_bytes':path.stat().st_size,
             'windows':len(offsets),'episodes':len(ep_ids),'action_windows':sum(x>=0 for x in robot_ids),
             'action_episodes':len({eid for eid,rid in zip(episode_num,robot_ids) if rid>=0}),
             'group_names':group_names,'array_sha256':hashes,'episodes_sha256':sha(ep_path)}
    dump(destination/'index.json',result);return result

class Store:
    def __init__(self):self.tars=OrderedDict();self.numeric=OrderedDict();self.verified={}
    def tar(self,path):
        path=str(path)
        if path in self.tars:self.tars.move_to_end(path);return self.tars[path]
        t=tarfile.open(path);self.tars[path]=t
        if len(self.tars)>8:self.tars.popitem(last=False)[1].close()
        return t
    def bytes(self,path,member):
        f=self.tar(path).extractfile(member)
        if f is None:raise FileNotFoundError(member)
        return f.read()
    def json(self,path,member):return json.loads(self.bytes(path,member))
    def arrays(self,path,member):
        with np.load(io.BytesIO(self.bytes(path,member)),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
    def verify(self,path,expected):
        p=Path(path);stat=p.stat();stamp=(stat.st_size,stat.st_mtime_ns,expected)
        if self.verified.get(str(p))==stamp:return
        if sha(p)!=expected:raise ValueError(f'Archive checksum mismatch: {p.name}')
        self.verified[str(p)]=stamp

class IndexedDataset(Dataset):
    def __init__(self,index_root,data_root,split,stage,diagnostic=False):
        self.root=Path(index_root);self.data_root=Path(data_root);self.stage=stage;self.split=split
        self.index=json.loads((self.root/'index.json').read_text())
        if self.index['diagnostic'] and not diagnostic:raise ValueError('Diagnostic index cannot start production training')
        if not diagnostic and not self.index['ready_for_training']:raise ValueError('Training index is not ready')
        self.info=self.index['splits'][split]
        self.manifest=Path(self.info['manifest'])
        if self.manifest.stat().st_size!=self.info['manifest_bytes'] or sha(self.manifest)!=self.info['manifest_sha256']:raise ValueError('Manifest changed after indexing')
        self.arr={}
        for name,digest in self.info['array_sha256'].items():
            p=self.root/f'{split}_{name}.npy'
            if sha(p)!=digest:raise ValueError('Index array checksum mismatch')
            self.arr[name]=np.load(p,mmap_mode='r')
        normpath=self.root/'normalization.json'
        if sha(normpath)!=self.index['normalization_sha256']:raise ValueError('Normalization changed')
        self.stats=json.loads(normpath.read_text())['by_domain']
        self.selected=np.flatnonzero(self.arr['robot']>=0) if stage==2 else np.arange(len(self.arr['offsets']))
        if not len(self.selected):raise ValueError('No eligible samples')
        self.store=None;self.file=None
    def __getstate__(self):
        state=self.__dict__.copy();state['store']=None;state['file']=None;return state
    def __len__(self):return len(self.selected)
    def row(self,i):
        if self.file is None:self.file=self.manifest.open('rb')
        self.file.seek(int(self.arr['offsets'][self.selected[i]]));return json.loads(self.file.readline())

    @staticmethod
    def _future_member(member, frame_index):
        match = re.match(r"^(.*_)\d{6}(\.[^.]+)$", member)
        if match is None:
            raise ValueError(f'Cannot derive future RGB member from {member}')
        return f"{match.group(1)}{int(frame_index):06d}{match.group(2)}"

    @staticmethod
    def _rgb_tensor(raw):
        with Image.open(io.BytesIO(raw)) as im:
            im = im.convert('RGB')
            if im.size != (224, 224):
                im = im.resize((224, 224), Image.Resampling.BILINEAR)
            array = np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return torch.from_numpy(array)

    def __getitem__(self,i):
        if self.store is None:self.store=Store()
        r=self.row(i);uid=r['episode_id'];path=r['tar'];self.store.verify(path,r['source_shard_sha256'])
        key=(path,uid)
        if key not in self.store.numeric:
            self.store.numeric[key]=self.store.arrays(path,uid+'/trajectory.npz')
            if len(self.store.numeric)>8:self.store.numeric.popitem(last=False)
        ep=self.store.numeric[key];t=r['frame_index'];meta=self.store.json(path,uid+'/metadata.json')
        images=[];future_pixels=[]
        if len(r['images'])!=len(r['image_sha256']) or len(r['images'])!=len(r['future_image_sha256']):
            raise ValueError('Image hash count mismatch')
        for loc,h,future_h in zip(r['images'],r['image_sha256'],r['future_image_sha256']):
            raw=self.store.bytes(path,loc['member'])
            if hashlib.sha256(raw).hexdigest()!=h:raise ValueError('RGB checksum mismatch')
            with Image.open(io.BytesIO(raw)) as im:images.append(im.convert('RGB'))
            future_member=self._future_member(loc['member'],r['future_frame_index'])
            future_raw=self.store.bytes(path,future_member)
            if hashlib.sha256(future_raw).hexdigest()!=future_h:raise ValueError('Future RGB checksum mismatch')
            future_pixels.append(self._rgb_tensor(future_raw))
        target=self.store.arrays(path,r['pose_target_member']);pose=np.concatenate([target['position_delta_m']/.1,target['rotation_6d']],-1)
        # Stage 1 does not use the action expert; keep native state in its input prompt.
        action=np.zeros((8,8),np.float32);amask=np.zeros((8,8),bool);state=np.zeros(9,np.float32);robot='none'
        if r['action_supervision_available']:
            robot=family(r['domain']);spec=REGISTRY[robot];stats=self.stats[r['domain']];ad=spec['action_dim'];sd=spec['state_dim']
            if meta['action_representation']!=spec['representation']:raise ValueError('Changed native action representation')
            idx=np.arange(t+r['action_row_offset'],t+r['action_row_offset']+8);valid=idx<len(ep['action']);idx=np.minimum(idx,len(ep['action'])-1)
            action[:,:ad]=(ep['action'][idx]-np.array(stats['action_mean']))/np.array(stats['action_std']);amask[:,:ad]=valid[:,None]
            state[:sd]=(ep['state'][t]-np.array(stats['state_mean']))/np.array(stats['state_std'])
        if self.stage==2 and robot=='none':raise ValueError('Unverified controls entered stage 2')
        horizon=f"{r['delta_seconds']:.4f} seconds" if r.get('delta_seconds') is not None else f"{r['future_frame_index']-t} recorded rows (sampling period unknown)"
        prompt=f"Task: {r['instruction']}\nRobot source: {r['domain']}\nCurrent EEF poses xyzw in native parent frame: {ep['poses_xyzw'][t].tolist()}\nCurrent robot state: {ep['state'][t].tolist()}\nPredict the next {horizon}."
        return {'id':r['id'],'episode_id':uid,'episode_number':int(self.arr['episode'][self.selected[i]]),'domain':r['domain'],'robot':robot,'prompt':prompt,'images':images,
                'program':r['program'] if self.stage==1 else None,'pose':torch.tensor(pose,dtype=torch.float32),'pose_mask':torch.tensor(target['pose_mask'],dtype=torch.bool),
                'pixel_target':torch.stack(future_pixels).float(),'pixel_mask':torch.ones(len(future_pixels),1,224,224,dtype=torch.bool),
                'action':torch.tensor(action),'action_mask':torch.tensor(amask),'state':torch.tensor(state),'delta':r['future_frame_index']-t}

class Collator:
    def __init__(self,processor,stage,pixel_views=3):self.processor=processor;self.stage=stage;self.pixel_views=pixel_views
    def __call__(self,items):
        texts=[];contexts=[];images=[]
        for x in items:
            message=[{'role':'user','content':[{'type':'image'} for _ in x['images']]+[{'type':'text','text':x['prompt']}]}]
            context=self.processor.apply_chat_template(message,tokenize=False,add_generation_prompt=True)
            contexts.append(context);texts.append(context+(x['program']+'<|im_end|>' if self.stage==1 else ''));images.extend(x['images'])
        batch=self.processor(text=texts,images=images,padding=True,return_tensors='pt')
        if self.stage==1:
            prefix=self.processor(text=contexts,images=images,padding=True,return_tensors='pt');labels=batch.input_ids.clone()
            for i in range(len(items)):
                n=int(prefix.attention_mask[i].sum())
                if not torch.equal(prefix.input_ids[i,:n],batch.input_ids[i,:n]):raise ValueError('Prompt target boundary mismatch')
                labels[i,:n]=-100
            labels[batch.attention_mask==0]=-100;batch['labels']=labels
        if max(x['pixel_target'].shape[0] for x in items)>self.pixel_views:
            raise ValueError('sample contains more camera views than configured pixel_views')
        target=torch.zeros(len(items),self.pixel_views,3,224,224,dtype=torch.float32)
        valid=torch.zeros(len(items),self.pixel_views,1,224,224,dtype=torch.bool)
        for i,x in enumerate(items):
            views=x['pixel_target'].shape[0];target[i,:views]=x['pixel_target'];valid[i,:views]=x['pixel_mask']
        return {'vl':dict(batch),**{k:torch.stack([x[k] for x in items]) for k in ['pose','pose_mask','action','action_mask','state']},
                'pixel_target':target,'pixel_mask':valid,
                'ids':[x['id'] for x in items],'episode_ids':[x['episode_id'] for x in items],'episode_numbers':[x['episode_number'] for x in items],
                'domains':[x['domain'] for x in items],'robots':[x['robot'] for x in items],'delta':torch.tensor([x['delta'] for x in items])}

class BalancedSampler(Sampler):
    """Stateless rank-disjoint draws; family sqrt-count, task, episode, window uniform."""
    def __init__(self,dataset,seed,rank,world,start_draw,draws):
        self.seed=seed;self.rank=rank;self.world=world;self.start=start_draw;self.draws=draws
        self.families={}
        for local,original in enumerate(dataset.selected):
            group=int(dataset.arr['group'][original]);ep=int(dataset.arr['episode'][original]);name=dataset.info['group_names'][group];fam=name.split(':')[0]
            self.families.setdefault(fam,{}).setdefault(group,{}).setdefault(ep,[]).append(local)
        self.names=sorted(self.families);counts=[sum(len(eps) for eps in self.families[f].values()) for f in self.names]
        self.episode_counts=dict(zip(self.names,counts));self.probs=np.sqrt(counts);self.probs=self.probs/self.probs.sum()
        self.tree={f:[list(e.values()) for e in self.families[f].values()] for f in self.names}
    def __len__(self):return self.draws
    def __iter__(self):
        for draw in range(self.start,self.start+self.draws):
            rng=np.random.default_rng(np.random.SeedSequence([self.seed,draw*self.world+self.rank]))
            f=self.names[rng.choice(len(self.names),p=self.probs)];tasks=self.tree[f];eps=tasks[rng.integers(len(tasks))];rows=eps[rng.integers(len(eps))]
            yield rows[rng.integers(len(rows))]
