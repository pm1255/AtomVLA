"""Isolated synthetic geometry checks; not benchmark results."""
import unittest
import numpy as np
from geometry import annotate, endpoint_program


class GeometryTest(unittest.TestCase):
    def test_boundary_does_not_cross(self):
        p = np.array([[0,0,0], [0.1,0,0], [0.2,0,0], [0.2,0.1,0]])
        s = [{'start':0,'end':2,'primitive':'move'}, {'start':2,'end':3,'primitive':'move'}]
        result = annotate(p,s,frame='world',arm_names=['panda'])
        self.assertEqual(result['local_directions'][1]['future_frame'], 2)
        self.assertEqual(result['segments'][1]['net_direction'], [0.,1.,0.])
        self.assertEqual(set(endpoint_program(p,s)[0]), {'primitive','target_xyz_m'})

    def test_returning_and_stationary_are_distinct(self):
        p = np.array([[0,0,0], [0.1,0,0], [0,0,0], [0,0,0]])
        s = [{'start':0,'end':2}, {'start':2,'end':3}]
        result = annotate(p,s,frame='world',arm_names=['panda'])['segments']
        self.assertEqual(result[0]['motion_type'], 'returning_motion')
        self.assertFalse(result[0]['net_direction_valid'])
        self.assertEqual(result[1]['motion_type'], 'stationary_translation')

    def test_bimanual_and_rotation(self):
        p = np.zeros((2,2,3)); p[1,1,2]=0.1
        q = np.zeros((2,2,4)); q[:,:,3]=1; q[1,0]=[0,0,1,0]
        result = annotate(p,[{'start':0,'end':1}],frame='world',arm_names=['left','right'],
                          quaternions_xyzw=q)['segments']
        self.assertEqual(result[0]['motion_type'],'rotation_without_translation')
        self.assertEqual(result[1]['net_direction'],[0.,0.,1.])

    def test_gap_and_nan_rejected(self):
        with self.assertRaises(ValueError):
            annotate(np.zeros((3,3)),[{'start':1,'end':2}],frame='world',arm_names=['panda'])
        with self.assertRaises(ValueError):
            annotate(np.full((2,3),np.nan),[{'start':0,'end':1}],frame='world',arm_names=['panda'])


if __name__ == '__main__':
    unittest.main()
