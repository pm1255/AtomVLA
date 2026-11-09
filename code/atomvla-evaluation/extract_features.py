import argparse,json,pathlib,os,importlib
import numpy as np

def _load(path):
    p=pathlib.Path(path)
    if p.is_dir():
        for name in ('hidden_states.npz','features.npz','paired_hidden_states.npz'):
            candidate=p/name
            if candidate.exists(): return candidate, np.load(candidate)
        module_name=os.environ.get('ATOMVLA_FEATURE_EXTRACTOR')
        if module_name:
            module=importlib.import_module(module_name)
            fn=getattr(module,'extract',None)
            if fn is None: raise SystemExit(f'{module_name} has no extract(checkpoint)')
            return p, fn(str(p))
        raise SystemExit('checkpoint has no hidden_states.npz; provide a real feature extractor via ATOMVLA_FEATURE_EXTRACTOR')
    return p,np.load(p)

def main():
 p=argparse.ArgumentParser();src=p.add_mutually_exclusive_group(required=True);src.add_argument('--features',help='npz with text and program_query arrays');src.add_argument('--checkpoint',help='checkpoint containing real hidden states or an extractor adapter');p.add_argument('--out',required=True);p.add_argument('--tensorboard');a=p.parse_args()
 source=a.features or a.checkpoint;resolved,z=_load(source);required={'text','program_query'}
 keys=set(z.files) if hasattr(z,'files') else set(z)
 if not required<=keys:raise SystemExit(f'missing arrays: {required-keys}')
 text=z['text'];prog=z['program_query']
 if text.ndim!=2 or prog.ndim!=2 or text.shape[1]!=prog.shape[1]:raise SystemExit('feature shape mismatch')
 if len(text)!=len(prog):raise SystemExit('paired features required')
 x=np.concatenate([text,prog]).astype('float64');x_centered=x-x.mean(axis=0,keepdims=True)
 # SVD is equivalent to the required two-component PCA and keeps this tool
 # usable in the minimal training image where scikit-learn is absent.
 _,_,vt=np.linalg.svd(x_centered,full_matrices=False);components=vt[:2]
 xy=x_centered.dot(components.T)
 variance=(x_centered**2).sum(axis=0).sum();explained=[float((xy[:,i]**2).sum()/variance) if variance else 0.0 for i in range(2)]
 labels=['text']*len(text)+['program_query']*len(prog)
 projection=[{'index':i,'stream':labels[i],'x':float(v[0]),'y':float(v[1])} for i,v in enumerate(xy)]
 out={'source':str(resolved),'n':len(text),'dim':int(text.shape[1]),'pca_explained_variance':explained,'paired_l2_mean':float(np.linalg.norm(text-prog,axis=1).mean()),'projection':projection}
 out_path=pathlib.Path(a.out);out_path.parent.mkdir(parents=True,exist_ok=True);out_path.write_text(json.dumps(out,indent=2),encoding='utf-8')
 import csv
 with (out_path.with_suffix('.csv')).open('w',newline='',encoding='utf-8') as stream:
  writer=csv.DictWriter(stream,fieldnames=['index','stream','x','y']);writer.writeheader();writer.writerows(projection)
 if a.tensorboard:
  try:
   from torch.utils.tensorboard import SummaryWriter
  except ImportError:
   try: from .tb import ScalarWriter as SummaryWriter
   except ImportError: from tb import ScalarWriter as SummaryWriter
  writer=SummaryWriter(a.tensorboard);writer.add_scalar('features/paired_l2',out['paired_l2_mean'],0);writer.add_scalar('features/samples',len(text),0);writer.flush();writer.close()

if __name__=='__main__':main()
