import unittest
from auto_only_policy import consensus,materialize
class Auto(unittest.TestCase):
 def p(self,verb='move',channel='vision',physical=False):return {'verb':verb,'arguments':{'object':'cup','target':'table','arm':'left','completion':'arrived'},'events':[{'role':'arrived','row':9}],'channel':channel,'physical_evidence':physical,'evidence_refs':[channel]}
 def test_two_channels_accept(self):self.assertEqual(consensus([self.p('move','vision'),self.p('move','state')],'libero').status,'auto_accepted')
 def test_one_channel_abstains(self):self.assertEqual(consensus([self.p()],'droid').status,'auto_unresolved')
 def test_conflict_abstains(self):self.assertEqual(consensus([self.p('move','vision'),self.p('pick','state')],'libero').status,'auto_unresolved')
 def test_pick_requires_physical(self):self.assertEqual(consensus([self.p('pick','vision'),self.p('pick','state')],'libero').status,'auto_unresolved')
 def test_unknown_never_promoted(self):self.assertEqual(consensus([self.p('move','vision'),{**self.p('move','state'),'arguments':{'object':'unknown','target':'table','arm':'left','completion':'arrived'}}],'droid').status,'auto_unresolved')
 def test_materialize(self):d=materialize({'verb':'move'},consensus([self.p('move','vision'),self.p('move','state')],'libero'),'libero');self.assertTrue(d['train_eligible']);self.assertFalse(d['verified_by_human'])
if __name__=='__main__':unittest.main()
