import argparse,json
from pathlib import Path
from full_ft.data import build_index
p=argparse.ArgumentParser();p.add_argument('--catalog',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--diagnostic',action='store_true');a=p.parse_args()
print(json.dumps(build_index(a.catalog,a.output,a.diagnostic)))
