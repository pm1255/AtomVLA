"""Export bounded, source-linked local review sheets; no automatic acceptance."""
import argparse,html,json
from pathlib import Path
from PIL import Image,ImageDraw
from common import write_json
from full_visual_convert import inventory,read_episode,sheets
from grounded_adapter import evidence_for
from local_semantics import windows,inspection_rows

def main():
 p=argparse.ArgumentParser();p.add_argument('--legacy',required=True);p.add_argument('--robot',required=True);p.add_argument('--root',required=True);p.add_argument('--source',default='libero');p.add_argument('--limit',type=int,default=3);a=p.parse_args();out=Path(a.root);out.mkdir(parents=True,exist_ok=True);done=0;cards=[];reviews=[]
 for item in inventory(a.legacy,a.robot):
  if item['source']!=a.source:continue
  if done>=a.limit:break
  # distinct tasks by choosing only demo_0 for the LIBERO diagnostic
  if a.source=='libero' and item['meta'].get('source_group')!='demo_0':continue
  with read_episode(item) as (pose,grip,state,actions,available,meta,decode):
   ev,scene=evidence_for(item,pose,grip);folder=out/item['episode_id'];folder.mkdir(exist_ok=True);write_json(folder/'evidence.json',ev)
   for i,w in enumerate(windows(ev,ev['fps'])):
    rows=inspection_rows(w,len(pose),ev);frames=sheets(decode(rows),rows);width=frames[0].width;height=frames[0].height
    canvas=Image.new('RGB',(width*3,height*((len(frames)+2)//3)),'white')
    for j,im in enumerate(frames):canvas.paste(im,((j%3)*width,(j//3)*height))
    name=f'{item["episode_id"]}/window_{i:03d}.jpg';canvas.save(out/name,quality=90)
    key=f'{item["episode_id"]}:{w["arm"]}:{w["start"]}:{w["end"]}'
    record={'id':key,'episode_id':item['episode_id'],'source':item['source'],'window':w,'source_receipt':item['receipt'],'coordinate_frame':meta['coordinate_frame'],'rows':rows,'image':name,'review':{'status':'pending','reviewer':None,'annotations':[],'notes':None}};reviews.append(record)
    cards.append(f'<article><h2>{html.escape(key)}</h2><p>{html.escape(meta["instruction"])}</p><p>Candidate interval, not a verified primitive. Record object, operation, boundaries, events, and unresolved parts in review.jsonl.</p><img loading="lazy" src="{html.escape(name)}" style="max-width:100%"></article>')
   done+=1
 (out/'review.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in reviews));(out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>AtomVLA annotation review</title><h1>Pending annotation review</h1>'+''.join(cards));write_json(out/'summary.json',{'episodes':done,'windows':len(reviews),'accepted':0,'status':'pending_independent_review'});print(json.dumps({'episodes':done,'windows':len(reviews),'root':str(out)}))
if __name__=='__main__':main()
