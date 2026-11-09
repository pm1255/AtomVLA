import unittest,tempfile,json,sys
from pathlib import Path
import numpy as np
from annotate import proposals
from records import make_record,validate_annotation
from common import split_for
from task_schema import infer_schema

class Contracts(unittest.TestCase):
 def fixture(self):
  p=np.zeros((42,2,7));p[:,:,6]=1
  p[4:7,0,0]=[.03,.06,.09];p[7:10,0,0]=.09
  p[10:13,0,0]=[.10,.11,.12];p[13:,0,0]=.12
  p[20:23,0,2]=[.03,.06,.09];p[23:,0,2]=.09
  p[26:29,0,0]=[.16,.20,.24];p[29:,0,0]=.24
  p[32:35,0,2]=[.06,.03,0];p[35:,0,2]=0
  g=np.ones((42,2));g[16:38,0]=0
  s={'grasp_actors':['self.bowl'],'bindings':{'self.bowl':'{A}'}}
  return p,g,s,{'info':{'{A}':'bowl/base0'}}
 def test_pick_move(self):
  p,g,s,i=self.fixture();a,issues=proposals(p,g,'place_bowl',s,i)
  self.assertEqual([x['verb'] for x in a],['pick','move','place']);self.assertEqual([len(x['events']) for x in a],[3,1,3]);self.assertTrue(all(x['quality']=='silver' for x in a))
 def test_click_is_not_pick(self):
  p,g,s,i=self.fixture();a,reason=proposals(p,g,'click_bell',s,i);self.assertFalse(a)
 def test_no_lift_no_pick(self):
  p,g,s,i=self.fixture();p[:,:,2]=0;a,reason=proposals(p,g,'place_bowl',s,i);self.assertFalse(a)
 def test_multi_object_ambiguous(self):
  p,g,s,i=self.fixture();s['grasp_actors'].append('self.other');self.assertFalse(proposals(p,g,'place_bowl',s,i)[0])
 def test_gold_cannot_be_self_declared(self):
  p,g,s,i=self.fixture();a=proposals(p,g,'place_bowl',s,i)[0][0];a['quality']='gold'
  with self.assertRaises(ValueError):validate_annotation(a,len(p))
 def test_single_endpoint_and_string_params(self):
  p,g,s,i=self.fixture();a=proposals(p,g,'place_bowl',s,i)[0][0];m={'split':'train','split_group':'x','task':'place_bowl','robot_variant':'test','instruction':'put bowl','coordinate_frame':'world'}
  r=make_record('id',a,p,g,m,a['start']);self.assertEqual(np.shape(r['targets']['terminal_pose_xyzw']),(2,7));self.assertFalse(r['train_eligible']);self.assertNotIn('keyframes',r['inputs'])
  a['arguments']['target']='0.123'
  with self.assertRaises(ValueError):validate_annotation(a,len(p))
 def test_release_removes_duplicate_transport(self):
  from release import select_records
  p,g,s,i=self.fixture();annotations=proposals(p,g,'place_bowl',s,i)[0]
  m={'split':'train','split_group':'x','split_group_verified':True,'task':'place_bowl','robot_variant':'test','instruction':'put bowl','coordinate_frame':'world'}
  records=[make_record('id',a,p,g,m,a['start']) for a in annotations]
  chosen,reasons=select_records(records)
  self.assertEqual([r['annotation']['verb'] for r in chosen],['pick','place'])
  self.assertEqual(reasons['transport_subsumed_by_place'],1)
 def test_release_rejects_future_leakage(self):
  from release import select_records
  p,g,s,i=self.fixture();a=proposals(p,g,'place_bowl',s,i)[0][0]
  m={'split':'train','split_group':'x','split_group_verified':True,'task':'place_bowl','robot_variant':'test','instruction':'put bowl','coordinate_frame':'world'}
  r=make_record('id',a,p,g,m,a['start']);r['targets']['keyframes'][0]['row']=0
  with self.assertRaises(ValueError):select_records([r])
 def test_seed_group_shared(self):
  self.assertEqual(split_for('task',42,'robot_a'),split_for('task',42,'robot_b'))
 def test_task_parser_no_execution(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'test.py';p.write_text('raise RuntimeError("never execute")\nclass Test:\n def play_once(self):\n  self.grasp_actor(self.bowl)\n')
   self.assertEqual(infer_schema(p)['grasp_actors'],['self.bowl'])
if __name__=='__main__':unittest.main()
