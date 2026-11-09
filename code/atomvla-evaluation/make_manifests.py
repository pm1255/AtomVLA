import argparse,json,pathlib
try:
 from .schema import Episode,save_json
except ImportError:
 from schema import Episode,save_json

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--benchmark',choices=['libero','libero_pro','robotwin','rmbench','real_robot'],required=True);p.add_argument('--tasks',nargs='+',required=True);p.add_argument('--trials',type=int,default=3);p.add_argument('--split',choices=['train','validation','test'],default='test');a=p.parse_args();rows=[]
 for task in a.tasks:
  for seed in range(a.trials):
   ep=Episode(f'{a.benchmark}__{task}__seed{seed}',a.benchmark,task,a.split,seed,'manifest').__dict__;rows.append(ep)
 pathlib.Path(a.out).parent.mkdir(parents=True,exist_ok=True)
 with open(a.out,'w') as f:
  for x in rows:f.write(json.dumps(x)+'\n')

if __name__=='__main__':main()
