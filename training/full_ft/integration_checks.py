"""Full trainer CLI, actual cached data/processor, tiny random Qwen; CPU diagnostic only."""
import argparse,json,os,subprocess,sys,tempfile,time
from pathlib import Path

ROOT=Path('/opt/atomvla')

def child(args):
    from unittest.mock import patch
    import torch
    from transformers import Qwen3VLConfig,Qwen3VLForConditionalGeneration
    import train_full
    config=json.loads(Path(args.config).read_text())
    torch.manual_seed(config['seed'])
    small=Qwen3VLConfig.from_pretrained(config['base_model'],local_files_only=True)
    text=small.text_config
    text.hidden_size=32;text.intermediate_size=64;text.num_hidden_layers=1
    text.num_attention_heads=4;text.num_key_value_heads=2;text.head_dim=8
    text.rope_scaling={'rope_type':'default','mrope_section':[1,1,2],'mrope_interleaved':True}
    vision=small.vision_config
    vision.depth=1;vision.hidden_size=32;vision.intermediate_size=64;vision.num_heads=4
    vision.out_hidden_size=32;vision.deepstack_visual_indexes=[]
    small._attn_implementation='sdpa'
    qwen=Qwen3VLForConditionalGeneration(small)
    sys.argv=['train_full.py','--config',args.config,'--stage',str(args.stage),
              '--run-dir',args.run_dir,'--device','cpu','--diagnostic','--smoke-steps','2']
    if args.init:sys.argv+=['--init',args.init]
    if args.resume:sys.argv+=['--resume',args.resume]
    with patch('vendor_delta.qwen_vl_backbone.Qwen3VLForConditionalGeneration.from_pretrained',return_value=qwen):
        train_full.main()

def main():
    p=argparse.ArgumentParser();p.add_argument('--child',action='store_true');p.add_argument('--config')
    p.add_argument('--stage',type=int);p.add_argument('--run-dir');p.add_argument('--init');p.add_argument('--resume');a=p.parse_args()
    if a.child:return child(a)
    started=time.time()
    root=ROOT/'diagnostics'/('full_ft_cli_'+str(time.time_ns()));root.mkdir(parents=True)
    cfg=json.loads((ROOT/'configs/full_ft/main.json').read_text())
    cfg.update(index_root=str(ROOT/'data/full_v3/ft_diagnostic_index_003'),world_size=1,loader_workers=1,
               cpu_threads_per_rank=1,tensorboard_root=str(ROOT/'tensorboard/full_ft/cpu_integration'/root.name))
    cfg['architecture'].update(queries=4,expert_hidden=32,expert_layers=1,expert_heads=4,sampling_steps=2)
    for sc in cfg['stages'].values():sc.update(save_every=1,eval_every=2,validation_samples=3,coverage_every=1)
    config=root/'config.json';config.write_text(json.dumps(cfg))
    benchmark_config=root/'benchmark_config.json'
    benchmark_config.write_text(json.dumps({**cfg,'run_kind':'benchmark_finetune','benchmark':'libero_pro'}))
    runs=[('stage1',1,None,None,config),('stage2',2,root/'stage1/checkpoints/step_00000002',None,config),
          ('stage2_resume',2,None,root/'stage2/checkpoints/step_00000001',config),
          ('libero_transfer',2,root/'stage2/checkpoints/step_00000002',None,benchmark_config)]
    for name,stage,init,resume,run_config in runs:
        cmd=[sys.executable,'-m','full_ft.integration_checks','--child','--config',str(run_config),'--stage',str(stage),'--run-dir',str(root/name)]
        if init:cmd+=['--init',str(init)]
        if resume:cmd+=['--resume',str(resume)]
        with (root/(name+'.log')).open('w') as log:
            result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=240,env={**os.environ,'TOKENIZERS_PARALLELISM':'false'})
        if result.returncode:raise RuntimeError(f'{name} failed: '+(root/(name+'.log')).read_text()[-10000:])
        done=json.loads((root/name/'DONE.json').read_text());assert done['diagnostic'] and done['steps']==2
    import torch
    expected=torch.load(root/'stage2/checkpoints/step_00000002/state.pt',map_location='cpu',weights_only=False,mmap=True)
    actual=torch.load(root/'stage2_resume/checkpoints/step_00000002/state.pt',map_location='cpu',weights_only=False,mmap=True)
    assert all(torch.equal(value,actual['model'][name]) for name,value in expected['model'].items())
    def events(name):return [json.loads(line) for line in (root/name/'logs/rank_0.jsonl').read_text().splitlines()]
    x=next(e for e in events('stage2') if e.get('event')=='step' and e['step']==2)
    y=next(e for e in events('stage2_resume') if e.get('event')=='step' and e['step']==2)
    assert x['sample_ids']==y['sample_ids'] and x['objective']==y['objective'] and x['counters']==y['counters']
    transfer=json.loads((root/'libero_transfer/contract.json').read_text())
    assert transfer['run_kind']=='benchmark_finetune' and transfer['benchmark']=='libero_pro'
    first=next(e for e in events('libero_transfer') if e.get('event')=='step')
    assert first['step']==1 and first['counters']['microbatches']==1
    from full_ft.runtime import load_checkpoint
    rejected=[]
    for bad,reason in [(root/'stage1/checkpoints/step_00000002','main stage 2'),
                       (root/'stage2/checkpoints/step_00000001','must be complete'),
                       (root/'libero_transfer/checkpoints/step_00000002','common main stage 2')]:
        try:load_checkpoint(bad,torch.nn.Linear(1,1),contract=transfer)
        except ValueError as exc:
            assert reason in str(exc);rejected.append(str(exc))
        else:raise AssertionError('Invalid benchmark source accepted')
    from build_benchmark_index import derive
    from full_ft.data import IndexedDataset
    source=ROOT/'data/full_v3/catalog_preview'
    try:derive(source,root/'forbidden_production','libero_pro')
    except ValueError as exc:assert 'not released' in str(exc)
    else:raise AssertionError('Preview was promoted to production')
    derived=derive(source,root/'libero_index','libero_pro',True)
    original={row['episode_id']:row['split'] for row in (json.loads(line) for line in (source/'episodes.jsonl').open())}
    subset=[json.loads(line) for line in (root/'libero_index/catalog/episodes.jsonl').open()]
    assert all(row['split']==original[row['episode_id']] for row in subset)
    samples=[]
    for split in ['train','validation','test']:
        ds=IndexedDataset(derived['index_root'],cfg['data_root'],split,2,True);sample=ds[0]
        assert sample['robot']=='libero' and sample['action_mask'][:,:7].all() and not sample['action_mask'][:,7:].any()
        samples.append({'split':split,'sample':sample['id'],'episodes':ds.info['episodes']})
    report={'passed':True,'device':'CPU','model':'tiny randomly initialized real Qwen3-VL, actual pretrained processor and cached source records',
            'trainer_entrypoint':'train_full.main','stage1_steps':2,'stage2_steps':2,'resume_from_step':1,
            'exact_resume_parameters_and_loss':True,'spawned_loader_workers':1,'full_pretrained_2b_gpu_tested':False,
            'stage2_to_benchmark_warmstart':{'passed':True,'optimizer_and_counters_reset':True,'rejected_sources':rejected},
            'benchmark_index':{'scope':'actual preview records, diagnostic only','source_splits_preserved':True,'unreleased_production_blocked':True,'actual_samples':samples},
            'evidence_root':str(root),'seconds':time.time()-started,'time':time.time()}
    (root/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
