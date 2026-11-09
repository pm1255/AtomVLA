import tempfile,unittest
from pathlib import Path
import numpy as np
from replay_semantic_audit import contract,entity,check_atom,repair
class ReplayAudit(unittest.TestCase):
 def data(self):
  n=20;xyz=np.zeros((n,3));xyz[6:,2]=.04;held=np.zeros(n,bool);held[3:14]=True;sup=[set() if 6<=i<12 else {'stove_1'} for i in range(n)]
  return {'n':n,'xyz':{'pot_1_joint0':xyz},'held':{'pot_1_joint0':held},'supports':{'pot_1_joint0':sup},'valid':np.ones(n,bool),'object_entities':{'pot_1_joint0':'pot_1'},'grip':np.r_[np.zeros(14),np.ones(6)*.05],'contract':{'objects':['pot_1','stove_1'],'regions':{},'goals':[['On','pot_1','stove_1']]}}
 def atom(self):return {'verb':'pick','start':0,'end':12,'arguments':{'object':'pot_1_joint0','target':'above_support'},'events':[{'role':'closed','row':3},{'role':'lifted','row':12}]}
 def test_late_lift_contradiction(self):
  d=check_atom(self.atom(),self.data());self.assertEqual(d['status'],'contradicted');self.assertIn('no_support_at_lift_endpoint',d['failures'])
 def test_repair_is_candidate_not_training(self):
  a,log=repair(self.atom(),self.data());self.assertEqual(a['end'],6);self.assertFalse(a['train_eligible']);self.assertEqual(check_atom(a,self.data())['status'],'supported')
 def test_unavailable_endpoint_is_not_pass(self):
  r=self.data();a=self.atom();a['end']=6;a['events'][-1]['row']=6;r['valid'][6]=False
  self.assertNotEqual(check_atom(a,r)['status'],'supported')
 def test_no_support_is_not_place(self):
  a={'verb':'place','start':12,'end':16,'arguments':{'object':'pot_1_joint0','target':'pot_1_joint0'},'events':[{'role':'supported','row':13},{'role':'released','row':16}]}
  self.assertIn('nonself_target',check_atom(a,self.data())['failures'])
 def test_bddl_relation(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'a.bddl';p.write_text('(define (problem X) (:objects pot_1 - pot) (:fixtures stove_1 - stove) (:regions (top (:target stove_1))) (:goal (And (On pot_1 stove_1_top))))')
   c=contract(p);self.assertEqual(entity('stove_1_top',c),'stove_1');self.assertEqual(c['goals'],[['On','pot_1','stove_1_top']])
if __name__=='__main__':unittest.main()
