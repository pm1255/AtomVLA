"""Full-parameter two-stage training. Launch with torchrun; never imports pilot model."""
import argparse,contextlib,hashlib,json,math,os,random,signal,time
import shutil
from collections import Counter,defaultdict
from datetime import timedelta
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from full_common import dump,sha
from full_ft.data import IndexedDataset,Collator,BalancedSampler,REGISTRY
from full_ft.model import FullAtomVLA
from full_ft.schedule import gates,lr_multiplier
from full_ft.runtime import save_checkpoint,load_checkpoint,rank,world,barrier,distributed

def move(x,device):
    if isinstance(x,torch.Tensor):return x.to(device,non_blocking=True)
    if isinstance(x,dict):return {k:move(v,device) for k,v in x.items()}
    return x

def autocast(device):
    return torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' else contextlib.nullcontext()

def module_gradient_norms(parameters):
    groups=defaultdict(list)
    for name,p in parameters:
        if name.startswith('backbone.model.model.visual.'):
            group='visual_connection' if 'merger' in name else 'vision'
        elif name.startswith('backbone.model.'):
            group='text_embedding' if 'embed_tokens' in name else 'text_output' if 'lm_head' in name else 'language'
        else:group='query' if 'difference_query' in name else name.split('.')[0]
        groups.setdefault(group,[])
        if p.grad is not None:groups[group].append(p.grad.detach())
    return {key:float(torch.stack(torch._foreach_norm(values)).norm()) if values else 0. for key,values in groups.items()}

def code_hashes():
    root=Path(__file__).parent
    paths=[root/'train_full.py']+list((root/'full_ft').glob('*.py'))+list((root/'vendor_delta').glob('*.py'))+[root/'full_common.py']
    return {str(p.relative_to(root)):sha(p) for p in paths}

def validate(model,ds,collate,device,stage,weights,limit,writer,step,progress):
    # Per-source fixed validation subset, full auxiliary losses, RNG isolated from training.
    rows=defaultdict(list)
    for i,orig in enumerate(ds.selected):
        fam=ds.info['group_names'][int(ds.arr['group'][orig])].split(':')[0];rows[fam].append(i)
    chosen=[]
    for family,indices in sorted(rows.items()):
        n=min(len(indices),max(1,limit//len(rows)));chosen.extend(indices[int(j)] for j in np.unique(np.linspace(0,len(indices)-1,n).astype(int)))
    chosen=sorted(chosen);local=chosen[rank()::world()];sums=defaultdict(float);counts=Counter();by_source=defaultdict(lambda:defaultdict(float));by_n=defaultdict(Counter)
    model.eval()
    with torch.random.fork_rng(devices=[device.index] if device.type=='cuda' else []),torch.no_grad():
        torch.manual_seed(20261031+rank())
        for done,i in enumerate(local,1):
            b=move(collate([ds[i]]),device)
            with autocast(device):_,metrics=model(b,stage,dict(query=True,pixel=True,pose=True),weights,evaluate=True)
            source=b['domains'][0]
            for k,v in metrics.items():
                value=float(v.detach());
                if not math.isfinite(value):raise FloatingPointError(f'Nonfinite validation {k}')
                sums[k]+=value;counts[k]+=1;by_source[source][k]+=value;by_n[source][k]+=1
            if done%16==0 or done==len(local):
                dump(progress,{'phase':'validation','step':step,'done':done,'target':len(local),'updated_at':time.time()})
                if rank()==0:dump(progress.parent/'progress.json',{'phase':'validation','step':step,'done':done,'target':len(local),'updated_at':time.time()})
                writer.add_scalar('evaluation/completed_fraction',done/max(1,len(local)),step*max(1,len(local))+done);writer.flush()
    payload=(dict(sums),dict(counts),dict(by_source),dict(by_n));all_payload=[None]*world()
    if distributed():dist.all_gather_object(all_payload,payload)
    else:all_payload=[payload]
    total=defaultdict(float);n=Counter();per=defaultdict(lambda:defaultdict(float));pn=defaultdict(Counter)
    for ss,nn,bb,bn in all_payload:
        for k,v in ss.items():total[k]+=v;n[k]+=nn[k]
        for source,values in bb.items():
            for k,v in values.items():per[source][k]+=v;pn[source][k]+=bn[source][k]
    result={k:v/n[k] for k,v in total.items()}
    if rank()==0:
        for k,v in result.items():writer.add_scalar('validation/'+k,v,step)
        for source,values in per.items():
            for k,v in values.items():writer.add_scalar('validation_by_domain/'+source+'/'+k,v/pn[source][k],step)
    writer.flush();model.train();return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--stage',type=int,choices=[1,2],required=True)
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--init',type=Path);p.add_argument('--resume',type=Path)
    p.add_argument('--diagnostic',action='store_true');p.add_argument('--smoke-steps',type=int)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda');a=p.parse_args()
    cfg=json.loads(a.config.read_text());sc=cfg['stages'][str(a.stage)]
    run_kind=cfg.get('run_kind','main');benchmark=cfg.get('benchmark')
    if run_kind not in ['main','benchmark_finetune']:raise ValueError('Unknown run kind')
    if run_kind=='benchmark_finetune':
        if a.stage!=2 or benchmark not in ['libero_pro','robotwin']:raise ValueError('Benchmark finetuning uses stage 2 losses and an explicit benchmark')
        robot='libero' if benchmark=='libero_pro' else 'robotwin'
        if robot not in REGISTRY:raise ValueError('Benchmark control adapter has not been verified and implemented')
    if cfg['finetuning']!='full' or cfg.get('use_lora',False):raise ValueError('Only full-parameter finetuning is accepted')
    if a.stage==2 and not (a.init or a.resume):raise ValueError('Stage 2 needs a full stage 1 checkpoint')
    if a.init and a.resume:raise ValueError('Choose init or resume')
    if a.smoke_steps is not None:
        if not 1<=a.smoke_steps<=100:raise ValueError('Smoke is capped at 100 steps')
        a.diagnostic=True;sc={**sc,'steps':a.smoke_steps,'eval_every':min(sc['eval_every'],a.smoke_steps),'save_every':min(sc['save_every'],a.smoke_steps),'validation_samples':min(sc['validation_samples'],16),'warmup':min(4,a.smoke_steps)}
    if a.diagnostic and a.smoke_steps is None:raise ValueError('Diagnostic preview allowed only for bounded smoke')
    if a.diagnostic:cfg={**cfg,'accumulation':1,'microbatch':1,'loader_workers':min(cfg['loader_workers'],2)}
    if a.device=='cpu' and not a.diagnostic:raise ValueError('CPU execution is only for explicitly bounded diagnostics')
    local=int(os.environ.get('LOCAL_RANK',0));device=torch.device('cuda',local) if a.device=='cuda' else torch.device('cpu')
    if device.type=='cuda':torch.cuda.set_device(local)
    if int(os.environ.get('WORLD_SIZE',1))>1:dist.init_process_group('nccl' if device.type=='cuda' else 'gloo',timeout=timedelta(seconds=cfg['distributed_timeout_seconds']))
    torch.set_num_threads(cfg['cpu_threads_per_rank']);seed=cfg['seed'];torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    if not a.diagnostic and world()!=cfg['world_size']:raise ValueError('Configured world size does not match launch')
    run=a.run_dir.resolve();writer=None;log=None
    if rank()==0:
        if run.exists() and not a.resume:raise FileExistsError('Use a fresh run directory or explicit resume')
        run.mkdir(parents=True,exist_ok=True)
        resource_target=Path(os.environ.get('ATOMVLA_RESOURCE_LOGDIR',str(Path(str(run)+'_watchdog')/'tensorboard/resource_monitor')))
        resource_link=Path(cfg['tensorboard_root'])/'resources'/run.name
        resource_link.parent.mkdir(parents=True,exist_ok=True)
        if not resource_link.exists() and not resource_link.is_symlink():resource_link.symlink_to(resource_target,target_is_directory=True)
    barrier();(run/'logs').mkdir(exist_ok=True);(run/'evidence').mkdir(exist_ok=True)
    import fcntl
    lock=(run/f'rank_{rank()}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    progress=run/'evidence'/f'progress_rank_{rank()}.json';start=time.monotonic();deadline=start+cfg['wall_seconds'];stop=False
    def stopped(sig,frame):
        nonlocal stop;stop=True
    signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
    log=(run/'logs'/f'rank_{rank()}.jsonl').open('a',buffering=1)
    tb=Path(cfg['tensorboard_root'])/('smoke' if a.diagnostic else 'formal')/run.name/f'rank_{rank()}'
    writer=SummaryWriter(str(tb))
    def emit(x):log.write(json.dumps({'time':time.time(),'rank':rank(),**x},allow_nan=False)+'\n')
    def heartbeat(phase,**extra):
        dump(progress,{'phase':phase,'updated_at':time.time(),**extra})
        if rank()==0:dump(run/'evidence/progress.json',{'phase':phase,'updated_at':time.time(),**extra})
    try:
        heartbeat('data_loading');train=IndexedDataset(cfg['index_root'],cfg['data_root'],'train',a.stage,a.diagnostic);valid=IndexedDataset(cfg['index_root'],cfg['data_root'],'validation',a.stage,a.diagnostic)
        if run_kind=='benchmark_finetune' and not a.diagnostic:
            if train.index.get('purpose')!={'kind':'benchmark_finetune','benchmark':benchmark}:raise ValueError('Benchmark needs its own released, split-preserving training index')
        if not a.diagnostic and train.info['episodes']<cfg['minimum_canonical_train_episodes']:raise ValueError('Insufficient canonical training episodes for main experiment')
        if a.stage==2 and not a.diagnostic and train.info['action_episodes']<cfg['minimum_action_train_episodes']:raise ValueError('Insufficient verified-control episodes')
        heartbeat('model_loading');model=FullAtomVLA(cfg['base_model'],REGISTRY,cfg['architecture']);model.configure_stage(a.stage);model.to(device)
        # No adapters or explicit VLM freezing. Inactive heads have no artificial gradients.
        params=[(n,p) for n,p in model.named_parameters() if p.requires_grad];back=[p for n,p in params if n.startswith('backbone.model.')];new=[p for n,p in params if not n.startswith('backbone.model.')]
        optimizer=torch.optim.AdamW([{'params':back,'lr':sc['vlm_lr'],'peak_lr':sc['vlm_lr'],'group_name':'vlm'},
                                    {'params':new,'lr':sc['new_lr'],'peak_lr':sc['new_lr'],'group_name':'query_and_heads'}],weight_decay=cfg['weight_decay'])
        contract={'stage':a.stage,'architecture':cfg['architecture'],'registry':REGISTRY,'config':cfg,'stage_config':sc,'code_hashes':code_hashes(),
                  'index_sha256':sha(Path(cfg['index_root'])/'index.json'),'finetuning':'full','diagnostic':a.diagnostic,'device_type':device.type}
        if run_kind=='benchmark_finetune':contract.update(run_kind=run_kind,benchmark=benchmark)
        coverage=np.zeros(train.info['episodes'],bool);counters={'microbatches':0,'query_dropped':0,'pixel_dropped':0,'pose_dropped':0,'both_aux_dropped':0,'domain_samples':{}};begin=0
        if a.init or a.resume:
            begin,oldcover,oldcount=load_checkpoint(a.resume or a.init,model,optimizer,contract,bool(a.resume))
            if a.resume:coverage=oldcover;counters=oldcount
        if begin>sc['steps']:raise ValueError('Checkpoint step exceeds configured budget')
        if rank()==0:dump(run/'contract.json',contract)
        if rank()==0:model.processor.save_pretrained(run/'processor')
        emit({'event':'model_ready','total_parameters':sum(p.numel() for p in model.parameters()),'trainable_parameters':sum(p.numel() for n,p in params),
              'frozen_vlm_parameters':sum(p.numel() for p in model.backbone.parameters() if not p.requires_grad),'train_windows':len(train),'validation_windows':len(valid)})
        micro=cfg['microbatch'];accum=cfg['accumulation'];global_batch=micro*accum*world();collate=Collator(model.processor,a.stage,cfg['architecture'].get('pixel_views',3))
        # Dedicated loader RNG avoids changing model RNG during exact checkpoint resume.
        sampler=BalancedSampler(train,seed+a.stage,rank(),world(),begin*accum*micro,(sc['steps']-begin)*accum*micro)
        if rank()==0:dump(run/'sampling_policy.json',{'family_probabilities':dict(zip(sampler.names,sampler.probs.tolist())),
             'family_episodes':sampler.episode_counts,'within_family':'uniform task -> uniform episode -> uniform cached window'})
        loader=DataLoader(train,batch_size=micro,sampler=sampler,collate_fn=collate,num_workers=cfg['loader_workers'],
                          pin_memory=device.type=='cuda',persistent_workers=cfg['loader_workers']>0,timeout=cfg['loader_timeout_seconds'] if cfg['loader_workers'] else 0,
                          generator=torch.Generator().manual_seed(seed+rank()),**({'prefetch_factor':2,'multiprocessing_context':'spawn'} if cfg['loader_workers'] else {}))
        wrapped=DistributedDataParallel(model,device_ids=[local] if device.type=='cuda' else None,find_unused_parameters=True,gradient_as_bucket_view=True) if distributed() else model
        iterator=iter(loader);heartbeat('ready');writer.add_text('configuration',json.dumps(contract),0)
        writer.add_scalar('pipeline/target_steps',sc['steps'],0);writer.add_scalar('pipeline/global_batch',global_batch,0);writer.add_scalar('pipeline/completed',0,begin)
        weights=sc['weights'];last=begin;path=a.resume
        if begin==0:emit({'event':'baseline_validation','step':0,'metrics':validate(model,valid,collate,device,a.stage,weights,sc['validation_samples'],writer,0,progress)})
        for step in range(begin,sc['steps']):
            if stop or time.monotonic()>deadline:raise TimeoutError('Training wall limit or termination request')
            optimizer.zero_grad(set_to_none=True);sums=defaultdict(float);counts=Counter();sample_ids=[];before=time.monotonic()
            for mi in range(accum):
                b=move(next(iterator),device);g=gates(seed,a.stage,step,mi,cfg['stage1_query_dropout'],cfg['stage2_pixel_loss_dropout'],cfg['stage2_pose_loss_dropout'])
                sync=contextlib.nullcontext() if not distributed() or mi==accum-1 else wrapped.no_sync()
                with sync:
                    with autocast(device):loss,metrics=wrapped(b,a.stage,g,weights)
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite objective')
                    (loss/accum).backward()
                counters['microbatches']+=1;counters['query_dropped']+=int(not g['query'])
                if a.stage==2:
                    counters['pixel_dropped']+=int(not g['pixel']);counters['pose_dropped']+=int(not g['pose']);counters['both_aux_dropped']+=int(not g['pixel'] and not g['pose'])
                for dom in b['domains']:counters['domain_samples'][dom]=counters['domain_samples'].get(dom,0)+1
                coverage[b['episode_numbers']]=True;sample_ids.extend(b['ids'])
                for k,v in metrics.items():sums[k]+=float(v.detach());counts[k]+=1
                heartbeat('training_microbatch',step=step,microbatch=mi)
            grad=torch.nn.utils.clip_grad_norm_([p for n,p in params],cfg['clip_norm'],error_if_nonfinite=True)
            module_grads=module_gradient_norms(params)
            qg=model.backbone.difference_query.weight.grad;qnorm=float(qg.norm()) if qg is not None else 0.
            if a.stage==2 and not qnorm>0:raise ValueError('Action path lost Query gradients')
            for group in optimizer.param_groups:group['lr']=group['peak_lr']*lr_multiplier(step,sc['steps'],sc['warmup'])
            optimizer.step()
            if device.type=='cuda':torch.cuda.synchronize()
            last=step+1
            event={'event':'step','step':last,'objective':sums['objective']/counts['objective'],'active_loss_means':{k:sums[k]/counts[k] for k in sums},'active_loss_counts':dict(counts),
                   'gradient_norm':float(grad),'query_gradient_norm':qnorm,'gradient_norm_by_module_after_clipping':module_grads,'seconds':time.monotonic()-before,'sample_ids':sample_ids,
                   'sample_draws_per_rank':last*accum*micro,'unique_episodes_rank':int(coverage.sum()),'counters':counters,
                   'gpu_memory_peak_gb':torch.cuda.max_memory_allocated()/1e9 if device.type=='cuda' else 0.,'gpu_memory_gb':torch.cuda.memory_allocated()/1e9 if device.type=='cuda' else 0.,
                   'lr':{x['group_name']:x['lr'] for x in optimizer.param_groups}}
            emit(event)
            for k,v in event['active_loss_means'].items():writer.add_scalar('train/active_loss/'+k,v,last)
            for k,v in event['active_loss_counts'].items():writer.add_scalar('train/active_loss_count/'+k,v,last)
            for k,v in module_grads.items():writer.add_scalar('gradient_after_clipping/'+k,v,last)
            for k in ['gradient_norm','query_gradient_norm','seconds','gpu_memory_peak_gb','gpu_memory_gb','unique_episodes_rank']:writer.add_scalar('train/'+k,event[k],last)
            for k in ['query_dropped','pixel_dropped','pose_dropped','both_aux_dropped']:writer.add_scalar('dropout/'+k+'_fraction',counters[k]/counters['microbatches'],last)
            for k,v in event['lr'].items():writer.add_scalar('learning_rate/'+k,v,last)
            for dom,n in counters['domain_samples'].items():writer.add_scalar('coverage/domain_draws/'+dom,n,last)
            writer.add_scalar('progress/completed_fraction',last/sc['steps'],last);writer.add_scalar('progress/equivalent_window_passes',last*global_batch/len(train),last)
            if last%sc['coverage_every']==0 or last==sc['steps']:
                bit=torch.tensor(coverage.astype(np.uint8),device=device)
                if distributed():dist.all_reduce(bit,op=dist.ReduceOp.MAX)
                writer.add_scalar('coverage/unique_training_episodes_global',int(bit.sum()),last)
            writer.flush();heartbeat('training',step=last,target_steps=sc['steps'])
            if last%sc['eval_every']==0 or last==sc['steps']:
                result=validate(model,valid,collate,device,a.stage,weights,sc['validation_samples'],writer,last,progress);emit({'event':'validation','step':last,'metrics':result})
            if last%sc['save_every']==0 or last==sc['steps']:
                heartbeat('checkpoint',step=last);path=run/'checkpoints'/f'step_{last:08d}'
                writer.add_scalar('checkpoint/disk_free_gb',shutil.disk_usage(run).free/1e9,last)
                save_checkpoint(path,model,optimizer,last,contract,coverage,counters);writer.add_scalar('checkpoint/last_saved_step',last,last);writer.flush();emit({'event':'checkpoint_verified','step':last,'path':str(path)})
                if rank()==0:
                    ready=sorted(p for p in (run/'checkpoints').glob('step_*') if p.is_dir() and (p/'READY.json').exists())
                    for old in ready[:-cfg.get('keep_checkpoints',3)]:
                        shutil.rmtree(old);emit({'event':'own_old_checkpoint_pruned','path':str(old)})
                barrier()
        writer.add_scalar('pipeline/completed',1,last);writer.flush();heartbeat('completed',step=last)
        if rank()==0:dump(run/'DONE.json',{'stage':a.stage,'steps':last,'diagnostic':a.diagnostic,'checkpoint':str(path),'time':time.time()})
    except Exception as e:
        import traceback
        dump(run/'evidence'/f'FAILED_rank_{rank()}.json',{'error':str(e),'traceback':traceback.format_exc(),'time':time.time()})
        raise
    finally:
        if writer:writer.close()
        if log:log.close()
        if distributed():dist.destroy_process_group()
if __name__=='__main__':main()
