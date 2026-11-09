"""Real pretrained processor + actual data, without loading full model/GPU."""
import json
from pathlib import Path
import torch
from transformers import AutoProcessor
from full_ft.data import IndexedDataset,Collator,family
R=Path('/opt/atomvla')
def main():
    processor=AutoProcessor.from_pretrained('/opt/models/Qwen3-VL-2B-Instruct',local_files_only=True);processor.tokenizer.padding_side='right'
    out=[]
    for stage in [1,2]:
        ds=IndexedDataset(R/'data/full_v3/ft_diagnostic_index_003',R/'data/full_v3','train',stage,True);chosen={}
        for i,orig in enumerate(ds.selected):
            f=ds.info['group_names'][int(ds.arr['group'][orig])].split(':')[0]
            if f not in chosen:chosen[f]=i
        rows=[ds[i] for i in chosen.values()];collate=Collator(processor,stage);batch=collate(rows)
        assert ('labels' in batch['vl'])==(stage==1)
        if stage==1:
            assert (batch['vl']['labels']!=-100).any(dim=1).all()
            assert batch['vl']['labels'][batch['vl']['attention_mask']==0].eq(-100).all()
        assert batch['action'].shape[-1]==8 and batch['state'].shape[-1]==9
        assert batch['pixel_target'].shape[1:]==(3,3,224,224)
        assert batch['pixel_mask'].shape[1:]==(3,1,224,224)
        out.append({'stage':stage,'sources':list(chosen),'input_shape':list(batch['vl']['input_ids'].shape),'image_grid':batch['vl']['image_grid_thw'].tolist(),'pixel_target_shape':list(batch['pixel_target'].shape),'has_target_labels':'labels' in batch['vl']})
    print(json.dumps({'passed':True,'processor':'actual Qwen3-VL-2B processor','checks':out}))
if __name__=='__main__':main()
