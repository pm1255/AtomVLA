"""Calculate data exposures from verified index counts, never nominal ZIP totals."""
import argparse,json,math
from pathlib import Path

def plan(config,index=None):
    batch=config['world_size']*config['microbatch']*config['accumulation'];result={'global_batch':batch,'stages':{}}
    for s,c in config['stages'].items():
        windows=(index['splits']['train']['windows' if s=='1' else 'action_windows'] if index else None)
        result['stages'][s]={'optimizer_steps':c['steps'],'global_sample_draws':c['steps']*batch,'verified_training_windows':windows,
            'equivalent_window_passes':c['steps']*batch/windows if windows else None,
            'note':'Weighted sampling means equivalent passes are draw count divided by unique windows, not uniform epochs; use logged distinct episode coverage.'}
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--index',type=Path);a=p.parse_args()
    print(json.dumps(plan(json.loads(a.config.read_text()),json.loads(a.index.read_text()) if a.index else None),indent=2))
