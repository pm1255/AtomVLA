import csv, json, time, pathlib
try:
    from .schema import TrialResult, save_json
    from .metrics import summarize
except ImportError:  # direct execution from this directory
    from schema import TrialResult, save_json
    from metrics import summarize

class Runner:
    """Runs an evaluator backend against fixed episodes and variants.

    Backend signature: backend.run(episode, variant) -> dict with success and
    optional steps/retries/pose_error_m/contract_ok. No projected values are
    generated; missing execution evidence is recorded as an error.
    """
    def __init__(self, backend, out, writer=None): self.backend=backend; self.out=pathlib.Path(out); self.writer=writer
    def run(self, episodes, variants):
        if not episodes: raise ValueError('no episodes')
        if not variants: raise ValueError('no variants')
        ids = [e.episode_id for e in episodes]
        if len(ids) != len(set(ids)): raise ValueError('duplicate episode_id')
        results=[]; total=len(episodes)*len(variants); done=0
        for variant in variants:
            for ep in episodes:
                t=time.time()
                try:
                    raw=self.backend.run(ep,variant) or {}
                    if 'success' not in raw: raise ValueError('backend omitted success')
                    r=TrialResult(ep.episode_id,ep.task,variant,bool(raw['success']),int(raw.get('steps',0)),int(raw.get('retries',0)),raw.get('pose_error_m'),raw.get('contract_ok'),raw.get('error'),{k:v for k,v in raw.items() if k not in {'success','steps','retries','pose_error_m','contract_ok','error'}})
                except Exception as exc:
                    r=TrialResult(ep.episode_id,ep.task,variant,False,0,error=f'{type(exc).__name__}: {exc}')
                results.append(r.as_dict());done+=1
                if self.writer:
                    self.writer('progress/fraction',done/max(1,total),done)
                    self.writer('progress/errors',sum(x['error'] is not None for x in results),done)
                    self.writer('evaluation/success',sum(x['success'] for x in results)/done,done)
                    self.writer('evaluation/failed_trials',sum(not x['success'] for x in results),done)
        save_json(self.out/'results.json',results)
        summary=summarize(results)
        save_json(self.out/'summary.json',summary)
        # CSV is deliberately flat so it can be imported directly into a paper table.
        fields=['episode_id','task','variant','success','steps','retries','pose_error_m','contract_ok','error','extras']
        with (self.out/'results.csv').open('w', newline='', encoding='utf-8') as stream:
            writer=csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
            for row in results:
                row=dict(row); row['extras']=json.dumps(row.get('extras') or {}, ensure_ascii=False, sort_keys=True)
                writer.writerow({k:row.get(k) for k in fields})
        save_json(self.out/'manifest_check.json', {'episode_ids':ids,'variants':list(variants),
                                                    'expected_trials':total,'actual_trials':len(results),
                                                    'consistent':len(results)==total})
        return results

class DryRunBackend:
    def run(self, episode, variant):
        return {'success':False,'steps':0,'retries':0,'contract_ok':False,'reason':'dry_run_no_environment'}
