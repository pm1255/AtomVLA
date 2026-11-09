"""Single-terminal-pose and string-program warm-up. No image/action generation claim."""
import json
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from vendor_delta.qwen_vl_backbone import QwenVLBackbone

class EventCollator:
 def __init__(self,processor):self.processor=processor
 def __call__(self,records):
  contexts=[];texts=[];query_texts=[];images=[]
  for r in records:
   cameras=sorted(r['current_rgb']);images.extend(Image.fromarray(r['current_rgb'][c]) for c in cameras)
   prompt=('Describe the next atomic call using only string arguments.\nInstruction: '+r['inputs']['instruction']+
     '\nRobot: '+r['robot_variant']+'\nCoordinates: '+r['coordinate_frame']+'\nCameras: '+','.join(cameras)+
     '\nCurrent EEF xyzw and gripper: '+json.dumps([r['inputs']['eef_pose_xyzw'],r['inputs']['gripper']]))
   msg=[{'role':'user','content':[{'type':'image'} for _ in cameras]+[{'type':'text','text':prompt}]}]
   ctx=self.processor.apply_chat_template(msg,tokenize=False,add_generation_prompt=True)
   contexts.append(ctx);texts.append(ctx+r['targets']['program_text']+'<|im_end|>')
   # Teacher-forced symbolic program is intentional stage-1 conditioning. Never
   # add terminal pose, future RGB, event times or completion measurements.
   query_texts.append(ctx+r['targets']['program_text']+'<|im_end|>')
  def process(text):return dict(self.processor(text=text,images=images,padding=True,return_tensors='pt'))
  text=process(texts);prefix=process(contexts);query=process(query_texts);labels=text['input_ids'].clone()
  for i in range(len(records)):
   n=int(prefix['attention_mask'][i].sum())
   if not torch.equal(prefix['input_ids'][i,:n],text['input_ids'][i,:n]):raise ValueError('text supervision boundary mismatch')
   labels[i,:n]=-100
  labels[text['attention_mask']==0]=-100
  if not (labels!=-100).any(1).all():raise ValueError('missing program target')
  text['labels']=labels
  return {'text':text,'query':query,
   'pose':torch.tensor([r['targets']['terminal_pose_xyzw'] for r in records],dtype=torch.float32),
   'arm_mask':torch.tensor([r['targets']['arm_mask'] for r in records],dtype=torch.bool),
   'program_weight':torch.tensor([r['loss_weights']['program'] for r in records]),
   'pose_weight':torch.tensor([r['loss_weights']['terminal_pose'] for r in records]),
   'ids':[r['id'] for r in records]}

def pose_objective(pred,target,mask):
 if not mask.any(1).all():raise ValueError('sample without active arm')
 # Select valid arms before all arithmetic: NaN padding must never leak via 0*NaN.
 p=pred[mask];t=target[mask]
 if not torch.isfinite(t).all():raise ValueError('nonfinite active pose target')
 if not torch.allclose(t[:,3:].norm(dim=-1),torch.ones(len(t),device=t.device),atol=.015):raise ValueError('bad target quaternion')
 pos=F.smooth_l1_loss(p[:,:3],t[:,:3],reduction='none',beta=.05).mean(-1)
 q=F.normalize(p[:,3:],dim=-1);qt=F.normalize(t[:,3:],dim=-1);dot=(q*qt).sum(-1)
 # Sign-invariant smooth SO(3) surrogate, stable at both zero and pi rotations.
 rot=(1-dot.square()).clamp_min(0)
 per=torch.zeros_like(mask,dtype=pred.dtype);per[mask]=pos+rot
 per=per.sum(1)/mask.sum(1)
 metrics={'position_l2_m':(p[:,:3]-t[:,:3]).norm(dim=-1).mean().detach(),
          'rotation_deg':(2*torch.acos(dot.detach().abs().clamp(0,1))*180/torch.pi).mean(),
          'position_loss':pos.mean().detach(),'rotation_loss':rot.mean().detach()}
 return per,metrics

class TerminalProgramModel(nn.Module):
 def __init__(self,base,backbone=None):
  super().__init__();self.backbone=backbone if backbone is not None else QwenVLBackbone(base,True,{'use_lora':False},use_difference_query=True,num_difference_queries=4,vlm_attention_backend='sdpa')
  self.backbone.float();self.backbone.model.config.use_cache=False
  h=self.backbone.difference_query.hidden_size
  self.pose=nn.Sequential(nn.LayerNorm(h),nn.Linear(h,h),nn.GELU(),nn.Linear(h,14))
  with torch.no_grad():
   self.pose[-1].bias.zero_();self.pose[-1].bias.reshape(2,7)[:,6]=1
  self.backbone.processor.tokenizer.padding_side='right'
 @property
 def processor(self):return self.backbone.processor
 def predict_pose(self,query):
  q=self.backbone(query,compute_vlm_loss=False)['backbone_embeddings']
  raw=self.pose(q.float().mean(1)).reshape(-1,2,7)
  return torch.cat([raw[...,:3],F.normalize(raw[...,3:],dim=-1)],-1),q
 def forward(self,b):
  text={k:v for k,v in b['text'].items() if k!='labels'}
  out=self.backbone.model(**text,return_dict=True,use_cache=False)
  labels=b['text']['labels'][:,1:];valid=labels!=-100
  token_loss=F.cross_entropy(out.logits[:,:-1].float().reshape(-1,out.logits.shape[-1]),labels.reshape(-1),ignore_index=-100,reduction='none').reshape_as(labels)
  ce=(token_loss*valid).sum(1)/valid.sum(1)
  pred,q=self.predict_pose(b['query']);pl,metrics=pose_objective(pred,b['pose'],b['arm_mask'])
  loss=(ce*b['program_weight']+pl*b['pose_weight']).mean()
  metrics.update(program_ce=ce.mean().detach(),terminal_pose_loss=pl.mean().detach(),objective=loss.detach(),
    token_accuracy=((out.logits[:,:-1].argmax(-1)==labels)&valid).sum().float()/valid.sum())
  return loss,metrics
