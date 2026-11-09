"""Snapshot relevant source hashes and package versions, without credentials."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import platform
import shutil
import subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--environment-root', required=True)
p.add_argument('--out', required=True)
p.add_argument('--backend', choices=['libero', 'robotwin'], required=True)
a = p.parse_args()
src, out = Path(a.environment_root), Path(a.out) / 'provenance' / a.backend
out.mkdir(parents=True, exist_ok=True)
if a.backend == 'robotwin':
    files = ['envs/adjust_bottle.py', 'envs/_base_task.py', 'envs/robot/robot.py',
             'task_config/demo_clean.yml', 'task_config/_embodiment_config.yml', 'task_config/_camera_config.yml']
    files += [str(f.relative_to(src)) for f in (src / 'envs').rglob('*curobo*planner*.py')]
else:
    files = ['libero/libero/bddl_files/libero_goal/put_the_bowl_on_the_plate.bddl',
             'libero/libero/bddl_files/libero_goal/put_the_cream_cheese_in_the_bowl.bddl']
hashes = {}
for name in files:
    f = src / name
    if f.is_file():
        target = out / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f, target)
        hashes[name] = hashlib.sha256(f.read_bytes()).hexdigest()
packages = {}
for name in ['numpy', 'scipy', 'torch', 'mujoco', 'mujoco-py', 'robosuite', 'sapien', 'imageio', 'tensorboard']:
    try:
        packages[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        pass
rev = subprocess.run(['git', '-C', str(src), 'rev-parse', 'HEAD'], capture_output=True, text=True)
value = {'environment_root': str(src), 'python': platform.python_version(), 'packages': packages,
         'git_commit': rev.stdout.strip() if rev.returncode == 0 else None,
         'git_commit_unavailable_reason': rev.stderr.strip() if rev.returncode else None,
         'source_sha256': hashes}
(out / 'environment.json').write_text(json.dumps(value, indent=2))
print(json.dumps(value, indent=2))
