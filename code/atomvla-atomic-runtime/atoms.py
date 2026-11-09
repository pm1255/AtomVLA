"""Portable atomic program. Generated arguments: poses and strings only.

The backend is bound once by the deployment host; it is not a VLM argument.
Position is the END-EFFECTOR origin in metres; orientation is quaternion xyzw.
"""
from typing import Protocol
import math


class ExecutionError(RuntimeError):
    pass


class Backend(Protocol):
    """Blocking calls must raise on timeout/failure, never silently continue.

    Deployments implement frame/TCP conversion, reachability, motion planning,
    controller limits and gripper feedback here. No universal hardware SDK is
    assumed. Contact checks are backend-specific evidence, not object identity
    inferred from a gripper-close command.
    """
    def validate(self, object: str, target: str, arm: str, frame: str): ...
    def move(self, position, orientation, arm: str, frame: str): ...
    def grip(self, command: str, arm: str): ...
    def hold(self, duration: str, arm: str): ...
    def require_grasp(self, object: str, arm: str): ...


def checked_pose(position, orientation):
    p, q = tuple(position), tuple(orientation)
    if len(p) != 3 or len(q) != 4:
        raise ValueError('Expected XYZ metres and quaternion XYZW')
    if any(isinstance(v, (bool, str)) or not isinstance(v, (int, float))
           or not math.isfinite(v) for v in p + q):
        raise ValueError('Pose must contain finite numbers')
    norm = math.sqrt(sum(v*v for v in q))
    if abs(norm - 1.0) > 0.01:
        raise ValueError('Orientation must be a unit quaternion, XYZW')
    return p, tuple(v/norm for v in q)


class Atoms:
    def __init__(self, backend: Backend):
        self.robot = backend
        # Logical state only; physical grasp is checked separately by backend.
        self.held_object = None
        self.at_placement = False
        self.failed = False

    def _check(self, object, target, arm, frame):
        if self.failed:
            raise ExecutionError('Episode faulted; reinitialize before resuming')
        if any(not isinstance(v, str) or not v for v in (object, target, arm, frame)):
            raise TypeError('Non-pose arguments must be nonempty strings')
        self.robot.validate(object, target, arm, frame)

    def _move(self, position, orientation, arm, frame):
        p, q = checked_pose(position, orientation)
        try:
            self.robot.move(p, q, arm, frame)
        except Exception:
            self.failed = True
            raise

    def _loaded(self, object, arm):
        if self.held_object != (object, arm):
            raise ExecutionError('This arm has not grasped the named object')
        try:
            self.robot.require_grasp(object, arm)
        except Exception:
            self.failed = True
            raise

    def wait(self, position, orientation, *, object='none', target='none', arm='panda', frame='world'):
        self._check(object, target, arm, frame)
        self._move(position, orientation, arm, frame)
        self.robot.hold('settle', arm)

    def align(self, position, orientation, *, object='none', target='none', arm='panda', frame='world'):
        self._check(object, target, arm, frame)
        if self.held_object is not None:
            raise ExecutionError('Align is an empty-gripper operation')
        self._move(position, orientation, arm, frame)

    def approach(self, position, orientation, *, object, target='none', arm='panda', frame='world'):
        self._check(object, target, arm, frame)
        if self.held_object is not None:
            raise ExecutionError('Cannot approach a new grasp while holding an object')
        # Validate all pose values before opening the gripper.
        checked_pose(position, orientation)
        self.robot.grip('open', arm)
        self._move(position, orientation, arm, frame)

    def grasp(self, position, orientation, *, object, target='none', arm='panda', frame='world'):
        self._check(object, target, arm, frame)
        if self.held_object is not None:
            raise ExecutionError('Already holding an object')
        self._move(position, orientation, arm, frame)
        self.robot.grip('close', arm)
        self.robot.hold('gripper', arm)
        try:
            self.robot.require_grasp(object, arm)
        except Exception:
            self.failed = True
            raise
        self.held_object = (object, arm)
        self.at_placement = False

    def lift(self, position, orientation, *, object, target='none', arm='panda', frame='world'):
        self._check(object, target, arm, frame)
        self._loaded(object, arm)
        self.at_placement = False
        self._move(position, orientation, arm, frame)
        self._loaded(object, arm)

    def transport(self, position, orientation, *, object, target, arm='panda', frame='world'):
        self._check(object, target, arm, frame)
        self._loaded(object, arm)
        self.at_placement = False
        self._move(position, orientation, arm, frame)
        self._loaded(object, arm)

    def place(self, position, orientation, *, object, target, arm='panda', frame='world'):
        self._check(object, target, arm, frame)
        self._loaded(object, arm)
        self.at_placement = False
        self._move(position, orientation, arm, frame)
        self._loaded(object, arm)
        self.at_placement = (object, target, arm)

    def release(self, position, orientation, *, object, target, arm='panda', frame='world'):
        self._check(object, target, arm, frame)
        self._loaded(object, arm)
        if self.at_placement != (object, target, arm):
            raise ExecutionError('Release requires a completed place for this target')
        self._move(position, orientation, arm, frame)
        self.robot.grip('open', arm)
        self.robot.hold('gripper', arm)
        self.held_object = None
        self.at_placement = False

    def retreat(self, position, orientation, *, object='none', target='none', arm='panda', frame='world'):
        self._check(object, target, arm, frame)
        if self.held_object is not None:
            raise ExecutionError('Retreat requires release first')
        self._move(position, orientation, arm, frame)
