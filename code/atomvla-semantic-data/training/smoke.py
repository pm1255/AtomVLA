"""Bounded real-record integration check; tiny Qwen is random, never pretrained evidence."""
import argparse,copy,hashlib,json,os,random,sys,time
from pathlib import Path
import numpy as np
import torch
from transformers import Qwen3VLConfig,Qwen3VLForConditionalGeneration
from unittest.mock import patch
from torch.utils.tensorboard import SummaryWriter
from loader import load_release
from common import write_json,digest
from stage1 import TerminalProgramModel,EventCollator,pose_objective
from full_ft.runtime import save_checkpoint,load_checkpoint

def tiny_model(base):
 c=Qwen3VLConfig.from_pretrained(base,local_files_only=True);t=c.text_config
 t.hidden_size=32;t.intermediate_size=64;t.num_hidden_layers=1;t.num_attention_heads=4;t.num_key_value_heads=2;t.head_dim=8
 t.rope_scaling={'rope_type':'default','mrope_section':[1,1,2],'mrope_interleaved':True}
 v=c.vision_config;v.depth=1;v.hidden_size=32;v.intermediate_size=64;v.num_heads=4;v.out_hidden_size=32;v.deepstack_visual_indexes=[]
 c._attn_implementation='sdpa';q=Qwen3VLForConditionalGeneration(c)
 with patch('vendor_delta.qwen_vl_backbone.Qwen3VLForConditionalGeneration.from_pretrained',return_value=q):return TerminalProgramModel(base)

def subset(manifest,out,limit,variant):
 rows=[json.loads(x) for x in Path(manifest).read_text().splitlines()];chosen=[];seen=set()
 # Stable randomized episode selection, one anchor per episode.
 rng=random.Random(20261104);rng.shuffle(rows)
 for e in rows:
  if e['robot_variant']!=variant or e['episode_id'] in seen:continue
  chosen.append(e);seen.add(e['episode_id'])
  if len(chosen)==limit:break
 if len(chosen)!=limit:raise ValueError('not enough independent episodes')
 Path(out).write_text(''.join(json.dumps(x)+'\n' for x in chosen));return list(load_release(out,'stage1_weak_events_v1'))

def main():
 p=argparse.ArgumentParser();p.add_argument('--release',required=True);p.add_argument('--run',required=True);p.add_argument('--tensorboard',required=True);p.add_argument('--base',required=True);p.add_argument('--steps',type=int,default=32);a=p.parse_args()
 if not 2<=a.steps<=100:raise ValueError('diagnostic bounded to 2..100 steps')
 run=Path(a.run);run.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.manual_seed(37);random.seed(37);np.random.seed(37)
 writer=SummaryWriter(a.tensorboard);started=time.time();phase='loading'
 def progress(stage,step=0):
  write_json(run/'progress.json',{'phase':stage,'step':step,'steps':a.steps,'time':time.time(),'diagnostic':True,'device':'CPU','model':'tiny random Qwen3-VL'})
 try:
  progress(phase);writer.add_scalar('progress/completed_fraction',0,0);writer.add_scalar('formal_training/waiting',1,0);writer.flush()
  train=subset(Path(a.release)/'train.jsonl',run/'train.jsonl',16,'franka_clean_50');valid=subset(Path(a.release)/'validation.jsonl',run/'validation.jsonl',8,'franka_clean_50')
  assert not ({r['split_group'] for r in train}&{r['split_group'] for r in valid})
  model=tiny_model(a.base);collate=EventCollator(model.processor)
  # Use real pretrained tokenizer/image processor and real all-camera inputs.
  bt=[collate([r]) for r in train];bv=[collate([r]) for r in valid]
  optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
  contract={'stage':1,'diagnostic':True,'architecture':'terminal_pose_string_program_v1_tiny32','registry':{},'release_sha':digest(Path(a.release)/'report.json'),
    'train_manifest_sha':digest(run/'train.jsonl'),'validation_manifest_sha':digest(run/'validation.jsonl'),'steps':a.steps,
    'supervision':['program_ce','terminal_position','terminal_orientation'],'not_tested':['future_rgb_generation','action_expert','full_pretrained_model','GPU','closed_loop']}
  write_json(run/'contract.json',contract)
  # Metamorphic checks with actual records: future data cannot affect inputs.
  changed=copy.deepcopy(train[0]);changed['targets']['terminal_pose_xyzw'][0][0]+=3;changed['targets']['keyframes'][0]['row']+=7
  changed['future_rgb']=[{c:np.zeros_like(x) for c,x in f.items()} for f in changed['future_rgb']]
  bc=collate([changed])
  assert all(torch.equal(v,bc['query'][k]) for k,v in bt[0]['query'].items())
  assert all(torch.equal(v,bc['text'][k]) for k,v in bt[0]['text'].items())
  model.eval()
  with torch.no_grad():
   pp,qq=model.predict_pose(bt[0]['query'])
   altered=copy.deepcopy(train[0]);altered['targets']['program_text']='move(object="test_object", target="test_target", arm="left", completion="arrived")'
   _,q2=model.predict_pose(collate([altered])['query']);assert not torch.equal(qq,q2)
   target=bt[0]['pose'].clone();mask=bt[0]['arm_mask'];l1,_=pose_objective(pp,target,mask);target[~mask]=float('nan');l2,_=pose_objective(pp,target,mask);assert torch.equal(l1,l2)
   signed=bt[0]['pose'].clone();signed[...,3:]*=-1;l3,_=pose_objective(pp,signed,mask);assert torch.allclose(l1,l3)
  checks={'future_target_isolation':True,'program_condition_changes_query':True,'inactive_arm_nan_mask':True,'quaternion_sign_invariance':True,'disjoint_train_validation_seed_groups':True}
  write_json(run/'checks.json',checks)
  def evaluate(batches,step,split):
   model.eval();total={}
   with torch.no_grad():
    for i,b in enumerate(batches):
     loss,m=model(b)
     if not torch.isfinite(loss):raise ValueError('nonfinite evaluation')
     for k,v in m.items():total[k]=total.get(k,0)+float(v)/len(batches)
     writer.add_scalar('evaluation/'+split+'_completed_fraction',(i+1)/len(batches),step)
   for k,v in total.items():writer.add_scalar(split+'/'+k,v,step)
   writer.flush();return total
  baseline={'train':evaluate(bt,0,'train_fixed'),'validation':evaluate(bv,0,'validation')};write_json(run/'baseline.json',baseline)
  gradients={};events=[]
  def update(b,check=False):
   model.train();optimizer.zero_grad(set_to_none=True);loss,m=model(b)
   if not torch.isfinite(loss):raise ValueError('nonfinite training loss')
   loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
   if check:
    groups={'vision':'.visual.','language':'.language_model.','query':'difference_query','pose_head':'pose.'}
    for key,pattern in groups.items():
     gs=[p.grad for n,p in model.named_parameters() if pattern in n and p.grad is not None];value=sum(float(g.square().sum()) for g in gs)**.5
     if value<=0 or not np.isfinite(value):raise ValueError('missing gradient '+key)
     gradients[key]=value
    # Qwen may tie lm_head to input embeddings; named_parameters deduplicates
    # aliases, so check the actual output tensor rather than its name.
    output_grad=model.backbone.model.lm_head.weight.grad
    if output_grad is None or not torch.isfinite(output_grad).all() or output_grad.norm()<=0:raise ValueError('missing output-head gradient')
    gradients['text_head']=float(output_grad.norm())
   optimizer.step();return {k:float(v.detach()) for k,v in m.items()},float(norm)
  for step in range(1,a.steps+1):
   if time.time()-started>1500:raise TimeoutError('diagnostic wall limit')
   progress('training',step-1);m,gn=update(bt[(step-1)%len(bt)],step==1);events.append({'step':step,**m,'gradient_norm':gn})
   for k,v in m.items():writer.add_scalar('train/'+k,v,step)
   writer.add_scalar('train/gradient_norm',gn,step);writer.add_scalar('progress/completed_fraction',step/a.steps,step)
   if step==1:
    for k,v in gradients.items():writer.add_scalar('gradient/'+k,v,step)
   if step%8==0 or step==a.steps:
    evaluate(bv,step,'validation');writer.flush()
  final={'train':evaluate(bt,a.steps,'train_fixed'),'validation':evaluate(bv,a.steps,'validation')};write_json(run/'final_metrics.json',final)
  progress('checkpoint',a.steps);path=run/'checkpoint';coverage=np.ones(len(train),bool);save_checkpoint(path,model,optimizer,a.steps,contract,coverage,{'samples':a.steps})
  writer.add_scalar('checkpoint/last_saved_step',a.steps,a.steps);writer.flush()
  # Exact optimizer+RNG recovery, one repeated next update. No extra curve points.
  expected,_=update(bt[0]);state={k:v.detach().clone() for k,v in model.state_dict().items()}
  restored,_,_=load_checkpoint(path,model,optimizer,contract,True);actual,_=update(bt[0]);assert restored==a.steps and expected==actual
  assert all(torch.equal(state[k],v) for k,v in model.state_dict().items())
  checks['exact_checkpoint_next_step']=True
  report={'passed':True,'scope':'CPU tiny random real Qwen3-VL; real converted records and actual pretrained processor','full_pretrained_gpu_verified':False,
   'train_episodes':len(train),'validation_episodes':len(valid),'steps':a.steps,'batch_size':1,'robot':'franka_clean_50','baseline':baseline,'final':final,'gradients':gradients,'checks':checks,
   'seconds':time.time()-started,'checkpoint':str(path),'losses_active':['program_ce(weight=0.5)','terminal_position(weight=1)','terminal_orientation(weight=1)'],
   'losses_unavailable':['future_rgb_MoT_not_implemented','action_no_verified_targets'],'parameters':sum(p.numel() for p in model.parameters())}
  write_json(run/'steps.json',events);write_json(run/'result.json',report);progress('completed',a.steps);writer.add_scalar('checks/all_passed',1,a.steps);writer.flush();print(json.dumps(report))
 except Exception as e:
  import traceback
  write_json(run/'FAILED.json',{'error':str(e),'traceback':traceback.format_exc()});progress('failed');raise
 finally:writer.close()
if __name__=='__main__':main()
