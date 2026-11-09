import unittest,copy
import numpy as np
from visual_annotation import validate_prediction,sample_rows,parse_json
from records import make_record

def atom(verb='move',s=0,e=9,arm='left',events=None):
 return {'kind':'primitive','verb':verb,'start':s,'end':e,'arguments':{'object':'cup','target':'table','arm':arm,'completion':'arrived'},'events':events or [{'role':'arrived','row':e}],'observation':'The cup visibly moves to the table.'}
class Visual(unittest.TestCase):
 def run_doc(self,segments,arms=['left']):return validate_prediction({'segments':segments},10,arms,list(range(10)),{'model':'fixture'})
 def test_move_one_endpoint(self):
  d=self.run_doc([atom()]);self.assertTrue(d['automatically_labeled_complete']);self.assertFalse(d['verified_complete']);self.assertEqual(len(d['annotations'][0]['events']),1)
 def test_gap_explicit(self):
  d=self.run_doc([atom(s=2,e=7)]);self.assertEqual(d['counts']['unresolved_transitions'],4);self.assertEqual([(s['start'],s['end']) for s in d['segments'] if s['kind']=='unresolved'],[(0,2),(7,9)])
 def test_overlap_rejects_both(self):
  d=self.run_doc([atom(e=6),atom(s=5)]);self.assertFalse(d['annotations']);self.assertEqual(d['counts']['unresolved_transitions'],9)
 def test_both_arm_overlap(self):
  d=self.run_doc([atom(arm='both'),atom(arm='right')],['left','right']);self.assertEqual(d['counts']['unresolved_transitions'],18)
 def test_inactive_arm_accounting(self):
  d=self.run_doc([atom(),{'kind':'hold','start':0,'end':9,'arm':'right','observation':'Right arm maintains a grasp on the box.'}],['left','right']);self.assertTrue(d['automatically_labeled_complete'])
 def test_unsampled_event_rejected(self):
  a=atom('pick',events=[{'role':'closed','row':4},{'role':'lifted','row':9}]);d=validate_prediction({'segments':[a]},10,['left'],[0,9],{});self.assertFalse(d['annotations'])
 def test_string_parameters_only(self):
  a=atom();a['arguments']['target']=[.1,.2,.3];self.assertFalse(self.run_doc([a])['annotations'])
 def test_open_variable_frames(self):
  for ev in [[{'role':'opened','row':9}],[{'role':'engaged','row':3},{'role':'opened','row':9}]]:
   d=self.run_doc([atom('open',events=ev)]);self.assertTrue(d['annotations'])
 def test_single_state_explicit_error(self):
  with self.assertRaises(ValueError):validate_prediction({'segments':[]},1,['left'],[0],{})
 def test_sampling_endpoints(self):
  p=np.zeros((200,2,7));p[:,:,6]=1;g=np.ones((200,2));g[80:,0]=0;r=sample_rows(p,g);self.assertEqual(r[0],0);self.assertEqual(r[-1],199);self.assertIn(80,r);self.assertLessEqual(len(r),40)
 def test_pose_extracted_not_generated(self):
  a=self.run_doc([atom()])['annotations'][0];p=np.zeros((10,2,7));p[:,:,6]=1;p[-1,0,:3]=[1,2,3];g=np.zeros((10,2));meta=dict(split='train',split_group='x',task='move',robot_variant='fixture',instruction='move cup',coordinate_frame='fixture')
  r=make_record('x',a,p,g,meta,0);self.assertEqual(r['targets']['terminal_pose_xyzw'][0][:3],[1,2,3]);self.assertEqual(r['targets']['terminal_pose_xyzw'][1],[0]*7);self.assertFalse(r['train_eligible'])
 def test_code_not_executed(self):
  with self.assertRaises(ValueError):parse_json('__import__("os").system("false")')
if __name__=='__main__':unittest.main()

class Compact(unittest.TestCase):
 def test_compact_roundtrip(self):
  from visual_annotation import parse_compact
  text='[["move","left",0,9,"gripper","above_cup",[["arrived",9]],"Gripper approaches the visible cup."]]'
  d=validate_prediction(parse_compact(text),10,['left'],[0,9],{});self.assertTrue(d['automatically_labeled_complete'])
 def test_no_repair_unknown_verb(self):
  from visual_annotation import parse_compact
  with self.assertRaises(ValueError):parse_compact('[["invented","left",0,9,"cup","table",[],"Unsupported action"]]')
