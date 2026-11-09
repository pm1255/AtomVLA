import inspect
import unittest
from unittest.mock import Mock
from types import SimpleNamespace
import numpy as np
from atoms import Atoms, ExecutionError, checked_pose
from episode_bowl_on_plate import run_episode
from libero_backend import LiberoBackend

P = [0., 0., 1.]
Q = [0., 0., 0., 1.]
KW = dict(object='bowl', target='plate')


class AtomicTests(unittest.TestCase):
    def test_only_pose_and_string_public_parameters(self):
        for name in ('wait', 'align', 'approach', 'grasp', 'lift', 'transport', 'place', 'release', 'retreat'):
            params = inspect.signature(getattr(Atoms, name)).parameters
            self.assertEqual(set(params), {'self', 'position', 'orientation', 'object', 'target', 'arm', 'frame'})
            for key in ('object', 'target', 'arm', 'frame'):
                value = params[key].default
                self.assertTrue(value is inspect.Parameter.empty or isinstance(value, str))

    def test_episode_has_11_real_primitive_calls(self):
        api = Mock()
        run_episode(api)
        self.assertEqual([c[0] for c in api.mock_calls], [
            'wait', 'align', 'approach', 'approach', 'grasp', 'lift',
            'transport', 'place', 'release', 'retreat', 'wait'])
        backend = Mock()
        run_episode(Atoms(backend))
        self.assertEqual(backend.move.call_count, 11)
        self.assertEqual([c.args[0] for c in backend.grip.call_args_list],
                         ['open', 'open', 'close', 'open'])

    def test_failed_place_never_releases_and_faults_episode(self):
        backend = Mock()
        atoms = Atoms(backend)
        atoms.grasp(P, Q, **KW)
        backend.move.side_effect = ExecutionError('unreachable')
        with self.assertRaises(ExecutionError):
            atoms.place(P, Q, **KW)
        backend.move.side_effect = None
        with self.assertRaises(ExecutionError):
            atoms.release(P, Q, **KW)
        self.assertEqual([c.args[0] for c in backend.grip.call_args_list], ['close'])

    def test_failed_contact_prevents_lift(self):
        backend = Mock()
        backend.require_grasp.side_effect = ExecutionError('no contact')
        atoms = Atoms(backend)
        with self.assertRaises(ExecutionError):
            atoms.grasp(P, Q, **KW)
        with self.assertRaises(ExecutionError):
            atoms.lift(P, Q, **KW)
        self.assertEqual(backend.move.call_count, 1)

    def test_cannot_release_without_place(self):
        backend = Mock()
        atoms = Atoms(backend)
        atoms.grasp(P, Q, **KW)
        with self.assertRaises(ExecutionError):
            atoms.release(P, Q, **KW)
        self.assertEqual(backend.grip.call_count, 1)

    def test_invalid_input_has_no_motion(self):
        backend = Mock()
        atoms = Atoms(backend)
        with self.assertRaises(TypeError):
            atoms.approach(P, Q, object=1)
        with self.assertRaises(ValueError):
            atoms.approach([float('nan'), 0, 0], Q, object='bowl')
        self.assertFalse(backend.move.called)
        self.assertFalse(backend.grip.called)
        for q in ([0, 0, 0, 0], [0, 0, 0], [0, 0, 0, float('inf')]):
            with self.assertRaises(ValueError):
                checked_pose(P, q)


class ControllerTests(unittest.TestCase):
    def make_backend(self):
        controller_type = type('OperationalSpaceController', (), {})
        controller = controller_type()
        controller.use_delta = True
        controller.control_ori = True
        controller.input_min = np.full(6, -1.)
        controller.input_max = np.full(6, 1.)
        controller.output_max = np.array([.02]*3 + [.2]*3)
        controller.output_min = -controller.output_max
        robot = SimpleNamespace(controller=controller, eef_site_id=0)
        sim = SimpleNamespace(data=SimpleNamespace(
            site_xpos=np.array([[0., 0., 1.]]), site_xmat=np.eye(3).reshape(1, 9)))
        env = SimpleNamespace(robots=[robot], sim=sim,
                              env=SimpleNamespace(obj_body_id={}),
                              step=Mock(return_value=(None, 0., False, {})))
        return LiberoBackend(env), env

    def test_scaling_reads_actual_controller(self):
        backend, env = self.make_backend()
        backend._step_towards(np.array([.004, 0., 1.]), np.eye(3))
        np.testing.assert_allclose(env.step.call_args.args[0], [.2, 0, 0, 0, 0, 0, -1])

    def test_unreachable_target_raises(self):
        backend, env = self.make_backend()
        with self.assertRaises(ExecutionError):
            backend.move([.1, 0, 1], Q, 'panda', 'world')
        self.assertEqual(env.step.call_count, 160)

    def test_reached_pose_returns_after_stable_samples(self):
        backend, env = self.make_backend()
        backend.move(P, Q, 'panda', 'world')
        self.assertEqual(env.step.call_count, 3)

    def test_unsupported_frame_fails_before_action(self):
        backend, env = self.make_backend()
        with self.assertRaises(ValueError):
            backend.move(P, Q, 'panda', 'camera')
        self.assertFalse(env.step.called)


if __name__ == '__main__':
    unittest.main(verbosity=2)
