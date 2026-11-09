import argparse
import pathlib
try:
    from .schema import load_manifest, save_json, MODES, VARIANTS, sha256
    from .runner import Runner
    from .backends import SmokeBackend, CommandBackend
except ImportError:  # direct execution from this directory
    from schema import load_manifest, save_json, MODES, VARIANTS, sha256
    from runner import Runner
    from backends import SmokeBackend, CommandBackend

DEFAULTS={
 'benchmark':['full'], 'ablation':['full','no_program','no_query','no_rgb','verb_only'],
 'memory':['full','memory_reset','memory_shuffled','memory_fixed'],
 'coupling':['direct_action','pi05','text_conditioned','full','feedback_off','feedback_on','feedback_shuffled'],
 'features':[], 'real_robot':['full'], 'smoke':['full']}

def main(argv=None):
 p=argparse.ArgumentParser(description='AtomVLA reproducible evaluation runner');p.add_argument('--mode',choices=MODES,required=True);p.add_argument('--manifest',required=True);p.add_argument('--out',required=True);p.add_argument('--backend',choices=['smoke','command'],default='smoke');p.add_argument('--command',nargs='+');p.add_argument('--variants',nargs='+');p.add_argument('--tensorboard');p.add_argument('--allow-smoke-success',action='store_true');a=p.parse_args(argv)
 eps=load_manifest(a.manifest);variants=a.variants or DEFAULTS[a.mode]
 if not variants: raise SystemExit('feature mode uses extract_features.py')
 if any(v not in VARIANTS for v in variants): raise SystemExit('unknown variant')
 if a.backend=='command' and not a.command: raise SystemExit('--command is required with --backend command')
 if a.mode=='benchmark' and len({e.benchmark for e in eps}) != 1: raise SystemExit('benchmark mode requires one benchmark')
 if a.mode=='memory' and any(e.benchmark!='rmbench' for e in eps): raise SystemExit('memory mode requires rmbench manifest')
 if a.mode=='real_robot' and any(e.benchmark!='real_robot' for e in eps): raise SystemExit('real_robot mode requires real_robot manifest')
 out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True)
 save_json(out/'protocol.json',{'mode':a.mode,'manifest':str(pathlib.Path(a.manifest).resolve()),'manifest_sha256':sha256(a.manifest),'episodes':len(eps),'variants':variants,'backend':a.backend,'projected_values':False,'seed_set':sorted({e.seed for e in eps}),'task_set':sorted({e.task for e in eps})})
 writer=None
 if a.tensorboard:
  try:
   try:
    from torch.utils.tensorboard import SummaryWriter
   except ImportError:
    try: from .tb import ScalarWriter as SummaryWriter
    except ImportError: from tb import ScalarWriter as SummaryWriter
   tbw=SummaryWriter(a.tensorboard)
   def writer(tag,value,step): tbw.add_scalar(tag,value,step);tbw.flush()
  except Exception as exc: raise SystemExit(f'cannot create TensorBoard writer: {exc}')
 backend=SmokeBackend(a.allow_smoke_success) if a.backend=='smoke' else CommandBackend(a.command)
 Runner(backend,out,writer).run(eps,variants)
 if a.tensorboard: tbw.close()

if __name__=='__main__':main()
