import unittest
from complete_annotation import audit,EVENT_SCHEMAS
V={'accepted':True,'reviewer':'test_fixture','evidence_ref':'fixture://groundtruth','reviewed_at':'test'}
def move(s,e):
 return {'verb':'move','start':s,'end':e,'arguments':{'object':'cup','target':'goal','arm':'left','completion':'arrived'},'events':[{'role':'arrived','row':e,'verification':V}], 'verification':V,'binding_verification':V,'boundary_verification':V}
class Coverage(unittest.TestCase):
 def test_endpoint_only_move_complete(self):self.assertTrue(audit(10,['left'],[move(0,9)])['complete_annotation'])
 def test_gap_not_silently_discarded(self):
  d=audit(10,['left'],[move(2,7)]);self.assertEqual(d['arms']['left']['unannotated_intervals'],[{'start':0,'end':2},{'start':7,'end':9}]);self.assertFalse(d['complete_annotation'])
 def test_overlap_rejected(self):self.assertFalse(audit(10,['left'],[move(0,5),move(4,9)])['complete_annotation'])
 def test_one_arm_does_not_cover_other(self):self.assertFalse(audit(10,['left','right'],[move(0,9)])['complete_annotation'])
 def test_all_time_covered_not_enough_without_evidence(self):
  a=move(0,9);a.pop('verification');d=audit(10,['left'],[a]);self.assertFalse(d['complete_annotation']);self.assertFalse(d['arms']['left']['unannotated_intervals'])
 def test_idle_needs_review(self):
  d=audit(10,['left'],[],[{'kind':'idle','start':0,'end':9,'arm':'left'}]);self.assertFalse(d['complete_annotation']);self.assertTrue(d['errors'])
 def test_vocab(self):self.assertEqual(len(EVENT_SCHEMAS),16)
if __name__=='__main__':unittest.main()
