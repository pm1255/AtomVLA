import unittest,tempfile,json,tarfile,hashlib,io
from pathlib import Path
import numpy as np
from PIL import Image
from full_visual_release import validate_tar
from visual_annotation import validate_prediction,VERSION,prompt
from records import make_record
from convert_archive import tar_bytes

class Release(unittest.TestCase):
 def build(self,path,mutate=None):
  a={'kind':'primitive','verb':'move','start':0,'end':4,'arguments':{'object':'gripper','target':'above_cup','arm':'left','completion':'arrived'},'events':[{'role':'arrived','row':4}],'observation':'Empty gripper moves above visible cup.'}
  doc=validate_prediction({'segments':[a]},5,['left'],[0,4],{})
  p=np.zeros((5,2,7));p[:,:,6]=1;g=np.zeros((5,2));m=dict(split='train',split_group='a',instruction='move',task='move',robot_variant='fixture',coordinate_frame='fixture',action_row_offset=1)
  r=make_record('a',doc['annotations'][0],p,g,m,0);r['schema']=VERSION;r['loss_weights']={'action':1};r['targets'].update(action_supervision_available=True,actions=np.zeros((8,7)).tolist(),action_mask=[True]*4+[False]*4,action_source_rows=[1,2,3,4]);b=io.BytesIO();Image.new('RGB',(4,4)).save(b,format='PNG');pixels=b.getvalue()
  def ref(row):return {'member':f'images/{row}.png','source_row':row,'sha256':hashlib.sha256(pixels).hexdigest()}
  r['inputs']['images']={'camera':ref(0)};r['targets']['keyframes'][0]['images']={'camera':ref(4)}
  if mutate:mutate(r)
  with tarfile.open(path,'w') as t:
   for row in [0,4]:tar_bytes(t,f'images/{row}.png',pixels)
   for name,obj in [('records.jsonl',r),('annotations.json',doc),('episode.json',m)]:tar_bytes(t,name,json.dumps(obj).encode())
 def test_contract(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'s.tar';self.build(p);r,doc=validate_tar(p);self.assertEqual(len(r),1)
 def test_action_shift_detected(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'s.tar';self.build(p,lambda r:r['targets'].update(action_source_rows=[0,1,2,3]));
   with self.assertRaises(ValueError):validate_tar(p)
 def test_future_image_row_detected(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'s.tar';self.build(p,lambda r:r['targets']['keyframes'][0]['images']['camera'].update(source_row=3));
   with self.assertRaises(ValueError):validate_tar(p)
 def test_prompt(self):
  p=np.zeros((5,1,7));p[:,:,6]=1;g=np.zeros((5,1));s=prompt(5,['left'],'open drawer',[0,4],p,g,'fixture');self.assertIn('"kind":"primitive"',s)
if __name__=='__main__':unittest.main()
