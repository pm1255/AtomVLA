"""Two CPU ranks test branch switching and gradient accumulation under DDP."""
import json,os
from datetime import timedelta
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from full_ft.checks import tiny_model,batch

def main():
    torch.set_num_threads(1);dist.init_process_group('gloo',timeout=timedelta(seconds=90));torch.manual_seed(77)
    m=tiny_model();m.configure_stage(2);m.train();wrapped=DistributedDataParallel(m,find_unused_parameters=True)
    opt=torch.optim.AdamW(m.parameters(),lr=1e-4)
    combinations=[[(False,False),(True,True)],[(True,False),(False,True)],[(False,False),(False,False)]]
    for step,seq in enumerate(combinations):
        opt.zero_grad(set_to_none=True)
        for micro,(pixel,pose) in enumerate(seq):
            import contextlib
            ctx=wrapped.no_sync() if micro==0 else contextlib.nullcontext()
            with ctx:
                loss,_=wrapped(batch(2,('droid',) if dist.get_rank()==0 else ('libero',)),2,dict(query=True,pixel=pixel,pose=pose),{'pixel':.1,'pose':.1});(loss/2).backward()
        opt.step()
    for name,value in m.state_dict().items():
        other=value.clone();dist.broadcast(other,0);assert torch.equal(value,other),name
    del wrapped,opt,m
    torch.manual_seed(88);m=tiny_model();m.configure_stage(1);m.train();wrapped=DistributedDataParallel(m,find_unused_parameters=True)
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-4)
    for step,seq in enumerate([(True,False),(False,True)]):
        opt.zero_grad(set_to_none=True)
        for micro,q in enumerate(seq):
            import contextlib
            with (wrapped.no_sync() if micro==0 else contextlib.nullcontext()):
                loss,_=wrapped(batch(1,('droid',) if dist.get_rank()==0 else ('libero',)),1,dict(query=q,pixel=q,pose=q),{'pixel':1.,'pose':.25});(loss/2).backward()
        opt.step()
    for name,value in m.state_dict().items():
        other=value.clone();dist.broadcast(other,0);assert torch.equal(value,other),name
    if dist.get_rank()==0:print(json.dumps({'passed':True,'ranks':2,'backend':'gloo','stage2_steps':3,'stage1_steps':2,'gradient_accumulation':2,'scope':'all model tensors synchronized after mixed robot heads, independent auxiliary branch switching, action-only steps and stage1 Query dropout'}))
    dist.destroy_process_group()
if __name__=='__main__':main()
