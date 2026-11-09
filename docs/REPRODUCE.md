# Reproduction guide

## Lightweight validation

From the repository root, with Python 3.12:

```bash
python -m pip install -r requirements-test.txt
python scripts/check_release.py
python -m unittest discover -s code/atomvla-evaluation/tests -v
python -m unittest discover -s code/atomvla-semantic-data/production -v
python -m unittest discover -s code/atomvla-endpoints -p 'test_*.py' -v
python -m unittest discover -s code/atomvla-atomic-runtime -v
```

## Data conversion

Acquire DROID, LIBERO and RoboTwin from their original distributors and obey their licenses. No dataset is redistributed. The conversion policy is `code/atomvla-semantic-data/production/source_policy.json`; GenieSim is excluded. Start with the schema and source receipt expectations in that directory's README. Keep raw sources immutable and use new output directories for releases.

```bash
python code/atomvla-semantic-data/production/convert_existing.py --help
python code/atomvla-semantic-data/production/preflight_large_scale.py --help
python code/atomvla-semantic-data/production/audit.py --help
```

The automatic annotations are weak supervision unless independently verified. Coverage and kinematic checks do not prove object-reference or operation semantics. Missing action contracts must remain masked; they must not become fabricated control targets.

## Training

Install a compatible CUDA build of PyTorch and `requirements-train.txt`. Download a compatible Qwen3-VL base model separately. Edit `base_model`, `data_root`, `index_root`, `tensorboard_root` and `world_size` in a copy of `configs/development.json` before running.

An accepted catalog must contain status, normalization, episode identities and split manifests. Build its index without bypassing release checks:

```bash
python training/build_training_index.py --catalog /path/to/accepted/catalog --output /path/to/new/index
```

Launch using the GPU count configured in your edited configuration:

```bash
torchrun --standalone --nproc_per_node=1 training/train_full.py --config configs/local.json --stage 1 --run-dir runs/stage1 --device cuda
torchrun --standalone --nproc_per_node=1 training/train_full.py --config configs/local.json --stage 2 --init runs/stage1/checkpoints/step_00002000 --run-dir runs/stage2 --device cuda
```

The checkpoint step above is an example; use the actual completed Stage 1 checkpoint. An eight-step diagnostic smoke can use `--smoke-steps 8`, but requires compatible diagnostic data and does not establish model quality. The native action registry currently implements DROID and LIBERO. RoboTwin's production action contract remains a blocker.

## Evaluation

Test only the runner and output format with:

```bash
python code/atomvla-evaluation/evaluate.py --mode smoke --manifest code/atomvla-evaluation/manifests/robotwin_three_tasks.jsonl --out runs/interface --backend smoke --tensorboard runs/interface/events
python -m tensorboard.main --logdir runs/interface/events --host 127.0.0.1 --port 6006
```

Open http://127.0.0.1:6006/ to inspect events. The smoke backend is a mock, so its success values are not measurements. For real evaluation, implement `evaluate(episode, variant)` in an importable module, set `ATOMVLA_ADAPTER_MODULE`, and use the corresponding command adapter. Supply genuine task IDs, seed/initial-state mappings, checkpoint IDs and environment versions. The default adapter reports unavailable; it never runs a robot.

A supplied NPZ containing real `text` and `program_query` arrays can be analyzed with `extract_features.py --features ... --out ...`. The repository does not contain a validated checkpoint extractor.

## Paper

```bash
cd paper
latexmk -pdf main.tex
```

Requires a local LaTeX distribution with the packages in `preamble.tex`. No completed benchmark results are implied by compiling the proposal. `paper/figures/plot_expected_features.py` reproduces only the labeled synthetic illustration; `paper/editable/` contains the editable pipeline slides.
