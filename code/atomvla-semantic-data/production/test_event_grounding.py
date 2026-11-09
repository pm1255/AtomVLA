import unittest,json
import numpy as np
from trajectory_events import build
from libero_state_evidence import extract
from local_semantics import assemble,windows,inspection_rows

class Events(unittest.TestCase):
 def data(self,n=30,arms=1):
  p=np.zeros((n,arms,7));p[:,:,6]=1;return p,np.ones((n,arms)),np.arange(n)*.1
 def test_full_both_arm_partition(self):
  p,g,t=self.data(arms=2);p[10:,0,0]=np.arange(20)*.01;d=build(p,g,t,'normalized_open_positive')
  for arm in ['left','right']:
   count=np.zeros(29)
   for a in d['partitions']:
    if a['arm']==arm:count[a['start']:a['end']]+=1
   self.assertTrue(np.all(count==1))
  self.assertFalse(d['semantic_annotation_complete'])
 def test_no_pick_from_gripper(self):
  p,g,t=self.data();g[10:]=0;d=build(p,g,t,'normalized_open_positive');self.assertTrue(any(e['kind']=='gripper_closing' for e in d['events']));self.assertFalse(any(e.get('verb')=='pick' for e in d['events']))
 def test_constant_aperture_not_normalized(self):
  p,g,t=self.data();g[:]=.02;d=build(p,g,t,'aperture_metres');self.assertEqual(d['events'],[])
 def test_quaternion_sign(self):
  p,g,t=self.data();p[10:,:,6]=-1;d=build(p,g,t,'aperture_metres');self.assertFalse(any(e['kind']=='rotation' for e in d['events']))
 def test_bad_timestamps(self):
  p,g,t=self.data();t[10]=t[9]
  with self.assertRaises(ValueError):build(p,g,t,'aperture_metres')
 def test_unknown_gripper_no_semantics(self):
  p,g,t=self.data();g[10:]=0;d=build(p,g,t,'unknown');self.assertFalse(d['events'])
 def test_robotwin_unit_inference_is_range_bounded(self):
  from grounded_adapter import evidence_for
  p,g,t=self.data();g[:,0]=np.linspace(.9,.1,len(g))
  d,_=evidence_for({'source':'robotwin','meta':{}},p,g)
  self.assertEqual(d['gripper_units'],'normalized_open_positive')
  self.assertIn('robotwin_timebase_assumed_20Hz_requires_collector_check',d['issues'])
 def test_windows_preserve_coverage(self):
  p,g,t=self.data();d=build(p,g,t,'aperture_metres');w=windows(d,10,max_seconds=.6);self.assertEqual(w[0]['start'],0);self.assertEqual(w[-1]['end'],29);self.assertTrue(all(a['end']==b['start'] for a,b in zip(w,w[1:])))

class State(unittest.TestCase):
 xml='<mujoco><worldbody><body name="robot"><joint name="robot0_joint1"/><body><joint name="gripper0_finger_joint1" type="slide"/></body></body><body name="cup"><freejoint name="cup_joint"/></body><body name="drawer"><joint name="drawer_joint" type="slide"/></body></worldbody></mujoco>'
 def data(self):
  # nq=10 nv=9; columns time,qrobot,qfinger,xyz,wxyz,qdrawer,qvel
  s=np.zeros((6,20));s[:,0]=np.arange(6)*.05;s[:,1]=np.arange(6)*.1;s[:,2]=np.arange(6)*.01;s[:,3:6]=[.1,.2,.3];s[:,6]=1
  obs=np.r_[s[1:,1],.8][:,None];g=np.r_[s[1:,2],.08][:,None];return s,obs,g
 def test_offset_and_missing_last(self):
  s,q,g=self.data();d=extract(self.xml,s,q,g);self.assertEqual(d['alignment']['state_row_offset'],1);self.assertFalse(d['valid'][-1]);self.assertTrue(np.isnan(d['objects']['cup_joint']['xyz'][-1]).all())
 def test_no_contacts_claimed(self):
  s,q,g=self.data();d=extract(self.xml,s,q,g);self.assertFalse(d['alignment']['contact_evidence_available'])
 def test_ambiguous_alignment(self):
  s,q,g=self.data();s[:,1:3]=0;q[:]=0;g[:]=0
  with self.assertRaises(ValueError):extract(self.xml,s,q,g)
 def test_extra_columns_rejected(self):
  s,q,g=self.data()
  with self.assertRaises(ValueError):extract(self.xml,np.c_[s,np.zeros(6)],q,g)
 def test_joint_mismatch(self):
  s,q,g=self.data();q+=10
  with self.assertRaises(ValueError):extract(self.xml,s,q,g)

class Semantics(unittest.TestCase):
 def test_no_generated_indices(self):
  d={'verb':'move','object':'gripper','target':'above_cup','events':{'arrived':'F99'},'evidence':'Gripper moves above the cup.'}
  with self.assertRaises(ValueError):assemble(json.dumps(d),{'start':0,'end':9,'arm':'left'},{'F0':0,'F1':9},{'events':[]},10)
 def test_pick_requires_observed_closure(self):
  d={'verb':'pick','object':'cup','target':'above_table','events':{'closed':'F1','lifted':'F2'},'evidence':'Gripper lifts a cup.'}
  with self.assertRaises(ValueError):assemble(json.dumps(d),{'start':0,'end':9,'arm':'left'},{'F0':0,'F1':3,'F2':9},{'events':[]},10)
 def test_endpoint_only_move(self):
  d={'verb':'move','object':'gripper','target':'above_cup','events':{'arrived':'F1'},'evidence':'Gripper moves above the cup.'};a=assemble(json.dumps(d),{'start':0,'end':9,'arm':'left'},{'F0':0,'F1':9},{'events':[]},10);self.assertEqual(a['events'],[{'role':'arrived','row':9}])
 def test_unresolved_not_filled(self):
  d={'verb':'unresolved','object':'unknown','target':'unknown','events':{},'evidence':'Object is obscured in this segment.','split':'F1'};a=assemble(json.dumps(d),{'start':0,'end':9,'arm':'left'},{'F0':0,'F1':3,'F2':9},{'events':[]},10);self.assertEqual(a['split_row'],3);self.assertEqual(a['kind'],'unresolved')
if __name__=='__main__':unittest.main()

class Refinement(unittest.TestCase):
 def run_pipeline(self,model,n=11,fps=10):
  from local_semantics import annotate
  from pathlib import Path
  from PIL import Image
  import tempfile,io
  p=np.zeros((n,1,7));p[:,:,6]=1;g=np.ones((n,1));e=build(p,g,np.arange(n)/fps,'normalized_open_positive');buf=io.BytesIO();Image.new('RGB',(8,8)).save(buf,format='PNG')
  def decode(rows):return {'camera':{r:(buf.getvalue(),{'source_row':r}) for r in rows}}
  with tempfile.TemporaryDirectory() as folder:return annotate(p,g,e,None,'test',decode,model,Path(folder),fps)
 def test_invalid_retries_leave_full_gap(self):
  class Model:
   path='fixture';calls=0
   def generate(self,q,images):self.calls+=1;return 'invalid'
  m=Model();d=self.run_pipeline(m);self.assertEqual(m.calls,2);self.assertEqual(d['counts']['unresolved_transitions'],10);self.assertEqual(len(d['review_queue']),1)
 def test_split_children_cover_parent_without_overlap(self):
  import re
  class Model:
   path='fixture';calls=0
   def generate(self,q,images):
    self.calls+=1;s,e=map(int,re.search(r'start row=(\d+), end row=(\d+)',q).groups());tokens=json.loads(q.split('Observed frame tokens: ')[1].split('. Measured changes:')[0])
    if self.calls==1:
     token=next(k for k,v in tokens.items() if s<v<e)
     return json.dumps({'verb':'unresolved','evidence':'Two visible operations need separation.','split':token})
    end=next(k for k,v in tokens.items() if v==e)
    return json.dumps({'verb':'move','object':'gripper','target':'above_cup','events':{'arrived':end},'evidence':'Empty gripper visibly moves above cup.'})
  m=Model();d=self.run_pipeline(m);self.assertTrue(d['automatically_labeled_complete']);self.assertEqual(len(d['annotations']),2);self.assertFalse(d['verified_complete']);self.assertFalse(d['rejected'])
 def test_adjacent_failed_windows_merge_once(self):
  import re
  class Model:
   path='fixture';calls=0
   def generate(self,q,images):
    self.calls+=1;s,e=map(int,re.search(r'start row=(\d+), end row=(\d+)',q).groups());tokens=json.loads(q.split('Observed frame tokens: ')[1].split('. Measured changes:')[0])
    if e-s<50:return json.dumps({'verb':'unresolved','evidence':'Operation extends beyond this short interval.'})
    end=next(k for k,v in tokens.items() if v==e)
    return json.dumps({'verb':'move','object':'gripper','target':'above_cup','events':{'arrived':end},'evidence':'One continuous free space move is visible.'})
  m=Model();d=self.run_pipeline(m,n=61);self.assertTrue(d['automatically_labeled_complete']);self.assertEqual(m.calls,6);self.assertEqual(len(d['annotations']),1)

class Contradictions(unittest.TestCase):
 def test_idle_contradicts_motion(self):
  d={'verb':'idle','evidence':'The robot appears stationary.'};e={'events':[{'arm':'left','start':0,'end':9,'kind':'translation'}]}
  with self.assertRaises(ValueError):assemble(json.dumps(d),{'start':0,'end':9,'arm':'left'},{'F0':0,'F1':9},e,10)
 def test_hinge_change_not_press_ground_truth(self):
  d={'verb':'press','object':'stove','target':'stove','events':{'pressed':'F1'},'evidence':'The stove visibly changes state.'};e={'events':[]};scene={'objects':{},'articulations':{}}
  with self.assertRaises(ValueError):assemble(json.dumps(d),{'start':0,'end':9,'arm':'left'},{'F0':0,'F1':9},e,10,scene)
