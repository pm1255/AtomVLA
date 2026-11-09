"""Atomic full checkpoints including per-rank RNG and consumed sample coverage."""
import hashlib,json,os,random,shutil,time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from full_common import dump,sha

TERMINAL='COMPLETED'
def distributed():return dist.is_available() and dist.is_initialized()
def barrier():
    if distributed():dist.barrier()
def rank():return dist.get_rank() if distributed() else 0
def world():return dist.get_world_size() if distributed() else 1

def atomic_torch(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp');torch.save(value,tmp)
    with tmp.open('rb') as f:os.fsync(f.fileno())
    tmp.replace(path);return {'bytes':path.stat().st_size,'sha256':sha(path)}

def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':[torch.cuda.get_rng_state()] if torch.cuda.is_available() else [],'cuda_scope':'local'}
def restore_rng(x):
    random.setstate(x['python']);np.random.set_state(x['numpy']);torch.set_rng_state(x['torch'])
    if x['cuda']:
        if x.get('cuda_scope')=='local':torch.cuda.set_rng_state(x['cuda'][0])
        else:torch.cuda.set_rng_state_all(x['cuda'])

def save_checkpoint(directory,model,optimizer,step,contract,coverage,counters):
    directory=Path(directory)
    if rank()==0:
        directory.mkdir(parents=True,exist_ok=False)
        model_bytes=sum(x.numel()*x.element_size() for x in model.state_dict().values())
        optimizer_bytes=sum(x.numel()*x.element_size() for state in optimizer.state.values() for x in state.values() if torch.is_tensor(x))
        required=model_bytes+optimizer_bytes+1024**3
        if shutil.disk_usage(directory).free<required:raise OSError('Insufficient free space for full model/optimizer checkpoint')
    barrier()
    rp=directory/f'rank_{rank():04d}.pt'
    rec=atomic_torch(rp,{'rng':rng_state(),'coverage':coverage,'counters':counters})
    dump(directory/f'rank_{rank():04d}.json',rec)
    barrier()
    if rank()==0:
        path=directory/'state.pt'
        # torch.save writes GPU tensors via CPU without making a second GPU model.
        rec=atomic_torch(path,{'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step,'contract':contract,'world_size':world()})
        saved=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        if set(saved['model'])!=set(model.state_dict()):raise ValueError('Full checkpoint key mismatch')
        for name,value in model.state_dict().items():
            if not torch.equal(value.detach().cpu(),saved['model'][name]):raise ValueError(f'Checkpoint reload mismatch: {name}')
        del saved
        dump(directory/'READY.json',{'step':step,'state':rec,'contract_sha256':hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest(),
                                     'world_size':world(),'verified_full_model_reload':True,'time':time.time()})
    barrier()

def load_checkpoint(directory,model,optimizer=None,contract=None,resume=False):
    directory=Path(directory);receipt=json.loads((directory/'READY.json').read_text());path=directory/'state.pt'
    if path.stat().st_size!=receipt['state']['bytes'] or sha(path)!=receipt['state']['sha256']:raise ValueError('Checkpoint integrity failure')
    saved=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    if resume and (saved['contract']!=contract or saved['world_size']!=world()):raise ValueError('Resume requires identical config/code/data and world size')
    if not resume and contract is not None:
        benchmark=contract.get('run_kind')=='benchmark_finetune'
        if benchmark:
            if contract.get('benchmark') not in ['libero_pro','robotwin']:raise ValueError('Unknown benchmark branch')
            if saved['contract']['stage']!=2 or contract['stage']!=2:raise ValueError('Benchmark finetuning initializes from main stage 2')
            if saved['contract'].get('run_kind')=='benchmark_finetune':raise ValueError('Each benchmark branch must start from common main stage 2')
            if saved['step']!=saved['contract']['stage_config']['steps']:raise ValueError('Main stage 2 must be complete before benchmark finetuning')
        elif saved['contract']['stage']!=1 or contract['stage']!=2:raise ValueError('Stage 2 initializes from a stage 1 full checkpoint')
        if not contract.get('diagnostic') and saved['contract'].get('diagnostic'):raise ValueError('Production stage 2 cannot initialize from a diagnostic checkpoint')
        for key in ['architecture','registry']:
            if saved['contract'][key]!=contract[key]:raise ValueError('Stage initialization architecture mismatch')
        if not benchmark and not contract.get('diagnostic') and saved['contract']['index_sha256']!=contract['index_sha256']:raise ValueError('Main two-stage run requires the same audited data index')
    model.load_state_dict(saved['model'],strict=True)
    if not resume:return 0,None,None
    optimizer.load_state_dict(saved['optimizer']);step=saved['step'];del saved
    rp=directory/f'rank_{rank():04d}.pt';r=json.loads(rp.with_suffix('.json').read_text())
    if rp.stat().st_size!=r['bytes'] or sha(rp)!=r['sha256']:raise ValueError('Rank state checksum mismatch')
    local=torch.load(rp,map_location='cpu',weights_only=False);restore_rng(local['rng'])
    return step,local['coverage'],local['counters']
