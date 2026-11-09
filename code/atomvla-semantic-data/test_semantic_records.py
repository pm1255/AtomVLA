"""Synthetic contract tests only. These are not robot/dataset accuracy evaluations."""
import copy
import unittest
import numpy as np
from semantic_records import make_record, pack_keyframes, ReviewRequired

def fixture(verb='pick'):
    n = 10
    poses = np.zeros((n,1,7)); poses[:,:,6] = 1
    poses[:,0,0] = np.arange(n)*.01
    proof = {'verified':True,'source':'unit_test_fixture_only'}
    ep = dict(id='synthetic_test',split='train',instruction='test',timestamp=np.arange(n)*.1,
              poses=poses,state=np.zeros((n,2)),actions=np.ones((n-1,7)),
              action_valid=np.ones(n-1,dtype=bool),arm_names=['right'],
              cameras={'front':[{'ref':f'test_{i}.png','timestamp':float(i*.1)} for i in range(n)]},
              calibration=proof,pose_convention='base_m_quat_xyzw',
              action_convention='state_j_to_j_plus_1',camera_tolerance_seconds=.01)
    roles = [('pregrasp',2),('closed',4),('lifted',7)] if verb=='pick' else [('arrived',7)]
    ann = dict(start=0,end=7,verb=verb,arguments=dict(object='bowl',target='above_table',arm='right',completion='held' if verb=='pick' else 'arrived'),
               semantics=proof,binding=proof,completion=proof,
               events=[dict(role=r,row=t,**proof) for r,t in roles])
    return ep,ann

class RecordsTest(unittest.TestCase):
    def test_pick_and_move_variable_lengths(self):
        a=make_record(*fixture(),0); b=make_record(*fixture('move'),0)
        self.assertEqual([f['role'] for f in a['targets']['keyframes']],['pregrasp','closed','lifted'])
        self.assertEqual(len(b['targets']['keyframes']),1)
        self.assertEqual(pack_keyframes([a,b])['temporal_mask'],[[True]*3,[True,False,False]])
        self.assertEqual(np.shape(a['targets']['terminal_pose_xyzw']),(1,7))
    def test_remaining_events_and_action_boundary(self):
        a=make_record(*fixture(),4,horizon=8)
        self.assertEqual([f['role'] for f in a['targets']['keyframes']],['lifted'])
        self.assertEqual(a['targets']['action_mask'],[True]*3+[False]*5)
        self.assertTrue(np.all(np.array(a['targets']['actions'])[3:]==0))
        self.assertNotIn('keyframes',a['inputs'])
    def test_missing_evidence_and_failure_rejected(self):
        for key in ('semantics','binding','completion'):
            ep,ann=fixture(); ann[key]={'verified':False,'source':'uncertain'}
            with self.assertRaises(ReviewRequired): make_record(ep,ann,0)
    def test_numeric_text_and_unknown_verb_rejected(self):
        for value in ([.1,.2,.3],'.123'):
            ep,ann=fixture();ann['arguments']['target']=value
            with self.assertRaises(ReviewRequired): make_record(ep,ann,0)
        ep,ann=fixture();ann['verb']='unknown'
        with self.assertRaises(ReviewRequired): make_record(ep,ann,0)
    def test_event_order_and_completion(self):
        for idx,row in ((1,2),(2,6),(0,11)):
            ep,ann=fixture();ann['events'][idx]['row']=row
            with self.assertRaises(ReviewRequired): make_record(ep,ann,0)
    def test_current_image_causality_and_sync(self):
        for row,time in ((0,.001),(2,9.)):
            ep,ann=fixture();ep['cameras']['front'][row]['timestamp']=time
            with self.assertRaises(ReviewRequired): make_record(ep,ann,0)
    def test_missing_camera_masks(self):
        ep,ann=fixture();ep['cameras']['wrist']=[None]*10
        record=make_record(ep,ann,0)
        self.assertFalse(record['targets']['keyframes'][0]['images']['wrist']['valid'])
        ep['cameras']['front'][2]=None
        with self.assertRaises(ReviewRequired): make_record(ep,ann,0)
    def test_bad_geometry_alignment_and_legacy_rejected(self):
        for field,val in [('pose_convention','unknown'),('action_convention','offset'),('split','unknown')]:
            ep,ann=fixture();ep[field]=val
            with self.assertRaises(ReviewRequired): make_record(ep,ann,0)
        ep,ann=fixture();ep['poses'][7,0,6]=0
        with self.assertRaises(ReviewRequired): make_record(ep,ann,0)
        with self.assertRaises(ReviewRequired): pack_keyframes([{'version':'full_v3'}])

if __name__ == '__main__': unittest.main()
