"""Concrete adapter for the LIBERO single-Panda OSC_POSE API.

Feedback control, NOT a collision-aware path planner or learned inverse model.
Uses controller scaling from the installed controller rather than assuming it.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from atoms import ExecutionError, checked_pose


class LiberoBackend:
    def __init__(self, env):
        self.env = env
        if len(env.robots) != 1:
            raise ValueError('This adapter supports single-Panda LIBERO only')
        self.robot = env.robots[0]
        self.ctrl = self.robot.controller
        if self.ctrl.__class__.__name__ != 'OperationalSpaceController':
            raise ValueError('Expected robosuite OperationalSpaceController')
        if not self.ctrl.use_delta or not self.ctrl.control_ori:
            raise ValueError('Expected delta OSC_POSE with orientation control')
        self.in_lo = np.asarray(self.ctrl.input_min, dtype=float)
        self.in_hi = np.asarray(self.ctrl.input_max, dtype=float)
        self.out_lo = np.asarray(self.ctrl.output_min, dtype=float)
        self.out_hi = np.asarray(self.ctrl.output_max, dtype=float)
        for limits in (self.in_lo, self.in_hi, self.out_lo, self.out_hi):
            if limits.shape != (6,) or not np.all(np.isfinite(limits)):
                raise ValueError('Expected six finite OSC controller limits')
        if not (np.all(self.out_hi > self.out_lo) and np.all(self.in_hi > self.in_lo)):
            raise ValueError('Invalid controller ranges')
        if not (np.allclose(self.in_lo, -1) and np.allclose(self.in_hi, 1)):
            raise ValueError('This adapter requires normalized [-1, 1] actions')
        self.gripper = -1.0
        self.steps = 0
        self.records = []

    def validate(self, object, target, arm, frame):
        if arm != 'panda' or frame != 'world':
            raise ValueError('LIBERO binding requires arm=panda, frame=world')
        names = self.env.env.obj_body_id
        for name in (object, target):
            if name != 'none' and name not in names:
                raise ValueError(f'Unknown LIBERO object: {name}')

    def pose(self):
        i = self.robot.eef_site_id
        return (self.env.sim.data.site_xpos[i].copy(),
                self.env.sim.data.site_xmat[i].reshape(3, 3).copy())

    def _error(self, p, r):
        current_p, current_r = self.pose()
        return p-current_p, Rotation.from_matrix(r @ current_r.T).as_rotvec()

    def _step_towards(self, p, r):
        if self.steps >= 4000:
            raise ExecutionError('Episode exceeded 4000 controller steps')
        dp, dr = self._error(p, r)
        delta = np.r_[dp, dr]
        action = self.in_lo + (delta-self.out_lo) / (self.out_hi-self.out_lo) * (self.in_hi-self.in_lo)
        action = np.clip(action, [-.3]*3 + [-.5]*3, [.3]*3 + [.5]*3)
        if not np.all(np.isfinite(action)):
            raise ExecutionError('Non-finite controller action')
        _, _, done, _ = self.env.step(np.r_[action, self.gripper])
        self.steps += 1
        if done:
            raise ExecutionError('Environment terminated during primitive')

    def move(self, position, orientation, arm, frame):
        self.validate('none', 'none', arm, frame)
        p, q = checked_pose(position, orientation)
        p, r = np.asarray(p), Rotation.from_quat(q).as_matrix()
        stable = 0
        for _ in range(160):
            dp, dr = self._error(p, r)
            reached = np.linalg.norm(dp) < .006 and np.linalg.norm(dr) < .10
            stable = stable+1 if reached else 0
            if stable >= 4:
                self.records.append({'position_error_m': float(np.linalg.norm(dp)),
                                     'rotation_error_rad': float(np.linalg.norm(dr)),
                                     'step': self.steps})
                return
            self._step_towards(p, r)
        raise ExecutionError('Pose target not reached within 160 steps')

    def grip(self, command, arm):
        self.validate('none', 'none', arm, 'world')
        self.gripper = {'open': -1.0, 'close': 1.0}[command]

    def hold(self, duration, arm):
        self.validate('none', 'none', arm, 'world')
        count = {'gripper': 18, 'settle': 30}[duration]
        p, r = self.pose()
        for _ in range(count):
            self._step_towards(p, r)

    def require_grasp(self, object, arm):
        self.validate(object, 'none', arm, 'world')
        obj = self.env.env.objects_dict[object]
        if not self.env.env._check_grasp(self.robot.gripper, obj.contact_geoms):
            raise ExecutionError(f'No opposing finger contact with {object}')
