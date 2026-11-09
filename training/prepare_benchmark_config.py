"""Prepare a full-parameter benchmark configuration after its derived index is ready.

No allocation, model training, or benchmark evaluation is launched by this tool.
The budget is derived from eligible windows, bounded to 2k--20k optimizer steps.
"""
import argparse,json,math
from pathlib import Path
from full_common import dump,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--main-config',type=Path,required=True)
    p.add_argument('--index',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Use a new configuration path')
    index=json.loads((a.index/'index.json').read_text());purpose=index.get('purpose',{})
    if purpose!={'kind':'benchmark_finetune','benchmark':'libero_pro'}:raise ValueError('Only verified LIBERO benchmark indexes are currently supported')
    if index['diagnostic'] or not index['ready_for_training']:raise ValueError('Benchmark index is not released')
    cfg=json.loads(a.main_config.read_text());tr=index['splits']['train']
    if tr['action_windows']!=tr['windows'] or tr['action_episodes']<1000:raise ValueError('Incomplete four-suite LIBERO control data')
    batch=64;steps=min(20000,max(2000,math.ceil(10*tr['action_windows']/batch)))
    cfg.update(run_kind='benchmark_finetune',benchmark='libero_pro',index_root=str(a.index.resolve()),
               world_size=4,microbatch=1,accumulation=16,minimum_canonical_train_episodes=1000,minimum_action_train_episodes=1000)
    cfg['stages']['2'].update(steps=steps,warmup=max(100,steps//20),vlm_lr=2e-6,new_lr=2e-5,
                             save_every=min(1000,steps),eval_every=min(500,steps),validation_samples=512)
    cfg['budget_basis']={'index_sha256':sha(a.index/'index.json'),'training_action_episodes':tr['action_episodes'],
        'training_action_windows':tr['action_windows'],'target_window_draws':10,'minimum_steps':2000,'maximum_steps':20000,
        'global_batch':batch,'planned_steps':steps,'equivalent_window_passes':steps*batch/tr['action_windows'],
        'note':'Initial budget, not a measured optimum. No test-success feedback is used to choose it.'}
    dump(a.output,cfg);print(json.dumps({'config':str(a.output),'budget':cfg['budget_basis'],'submitted':False}))
if __name__=='__main__':main()
