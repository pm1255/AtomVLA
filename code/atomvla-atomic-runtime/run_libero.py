"""Run the full executable episode from its saved initial LIBERO state."""
import argparse
import json
from pathlib import Path
import sys
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--libero', type=Path, required=True, help='LIBERO repository')
    parser.add_argument('--recording', type=Path, required=True, help='seed11 recording directory')
    parser.add_argument('--out', type=Path, required=True, help='New output directory')
    args = parser.parse_args()
    if args.recording.name != 'put_the_bowl_on_the_plate__seed11':
        parser.error('This example is bound to put_the_bowl_on_the_plate__seed11')
    sys.path.insert(0, str(args.libero.resolve()))
    import numpy as np
    from tensorboardX import SummaryWriter
    from libero.libero.envs import OffScreenRenderEnv
    from atoms import Atoms
    from libero_backend import LiberoBackend
    from episode_bowl_on_plate import run_episode

    args.out.mkdir(parents=True, exist_ok=False)
    writer = SummaryWriter(str(args.out / 'tensorboard'))
    writer.add_scalar('status/waiting_for_environment', 1, 0)
    writer.add_scalar('evaluation/completion_fraction', 0, 0)
    writer.flush()
    env = None
    backend = None
    result = {'status': 'initializing', 'physics_verified': False,
              'kind': 'new atomic-runtime replay, not the old endpoint pilot'}
    try:
        bddl = args.libero / 'libero/libero/bddl_files/libero_goal/put_the_bowl_on_the_plate.bddl'
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), use_camera_obs=False,
                                 control_freq=20, horizon=4000, ignore_done=True)
        env.seed(11)
        env.reset()
        initial_gripper = env.robots[0].gripper.current_action.copy()
        # The saved XML may contain absolute asset paths from the original host.
        # Use the original runtime/assets or relocate paths explicitly before run.
        env.reset_from_xml_string((args.recording / 'initial_model.xml').read_text())
        initial_state = np.load(args.recording / 'initial_state.npy', allow_pickle=False)
        env.sim.reset()
        env.set_init_state(initial_state)
        robot = env.robots[0]
        robot.gripper.current_action = initial_gripper
        robot.controller.update(force=True)
        robot.controller.reset_goal()
        robot.controller.use_delta = True
        if not np.allclose(env.get_sim_state(), initial_state, atol=1e-8, rtol=0):
            raise RuntimeError('Saved simulator state could not be restored exactly')
        backend = LiberoBackend(env)
        writer.add_scalar('status/waiting_for_environment', 0, 1)
        writer.add_scalar('evaluation/running', 1, 1)
        writer.flush()
        run_episode(Atoms(backend))
        success = bool(env.check_success())
        result.update(status='complete', physics_verified=True, official_success=success)
        writer.add_scalar('evaluation/official_success', int(success), 2)
        writer.add_scalar('evaluation/completion_fraction', 1, 2)
    except Exception as exc:
        result.update(status='error', error=str(exc), traceback=traceback.format_exc())
        writer.add_scalar('evaluation/error', 1, 2)
    finally:
        if backend is not None:
            result.update(controller_steps=backend.steps, motion_records=backend.records)
        (args.out / 'result.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
        writer.add_scalar('evaluation/running', 0, 3)
        writer.flush()
        writer.close()
        if env is not None:
            env.close()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get('official_success') else 1


if __name__ == '__main__':
    raise SystemExit(main())
