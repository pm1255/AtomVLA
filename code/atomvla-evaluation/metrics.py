from collections import Counter, defaultdict
import math, statistics

def summarize(results):
    by=defaultdict(list)
    for r in results: by[(r['variant'],r['task'])].append(r)
    rows=[]
    for (variant,task),items in sorted(by.items()):
        errors=Counter((x.get('error') or 'none').split(':',1)[0] for x in items)
        rows.append({'variant':variant,'task':task,'n':len(items),'success_rate':sum(x['success'] for x in items)/len(items),
                     'successes':sum(x['success'] for x in items),'mean_steps':statistics.mean(x['steps'] for x in items),
                     'mean_retries':statistics.mean(x['retries'] for x in items),
                     'mean_pose_error_m':_mean(x.get('pose_error_m') for x in items),
                     'contract_rate':_mean(x.get('contract_ok') for x in items),
                     'error_counts':dict(errors)})
    variants=defaultdict(list)
    for r in rows: variants[r['variant']].append(r)
    overall=[]
    for v,items in sorted(variants.items()):
        n=sum(x['n'] for x in items)
        flat=[r for r in results if r['variant']==v]
        overall.append({'variant':v,'n':n,'success_rate':sum(x['successes'] for x in items)/n,
                        'mean_steps':sum(x['mean_steps']*x['n'] for x in items)/n,
                        'mean_retries':sum(x['mean_retries']*x['n'] for x in items)/n,
                        'tasks':len(items),'errors':sum(x.get('error') is not None for x in flat),
                        'bootstrap_success_rate':bootstrap_ci(flat)})
    return {'by_task':rows,'overall':overall,'n_trials':len(results),
            'errors':sum(x.get('error') is not None for x in results)}

def _mean(xs):
    v=[float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return sum(v)/len(v) if v else None

def bootstrap_ci(results, key='success', rounds=1000, seed=0):
    import random
    rng=random.Random(seed);x=[float(r[key]) for r in results]
    if not x:return {'mean':None,'lo':None,'hi':None,'n':0}
    means=[sum(rng.choice(x) for _ in x)/len(x) for _ in range(rounds)]
    means.sort();return {'mean':sum(x)/len(x),'lo':means[int(.025*rounds)],'hi':means[int(.975*rounds)],'n':len(x)}
