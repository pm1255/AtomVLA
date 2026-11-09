# AtomVLA

**Typed atomic programs, terminal-pose queries, and future-state imagination for robotic manipulation.**

[中文说明](README.zh-CN.md) · [Paper source](paper/main.tex) · [Implementation status](docs/STATUS.md) · [Reproduction](docs/REPRODUCE.md)

This repository releases the currently available research code and proposal manuscript. It is a work in progress, **not an empirically validated benchmark release**. Manuscript performance tables contain explicitly labeled planning estimates; the feature illustration is synthetic. No measured leaderboard scores, trained weights, or datasets are distributed here.

## Method

A text head represents an atomic operation and its string-valued arguments. A continuous action query grounds the terminal end-effector pose. Primitive-specific future keyframes are intended to replace an explicit procedural body, while an action expert generates native continuous controls. The annotation pipeline retains program arguments, terminal-pose supervision, event-aligned frames, evidence masks, source identity, and episode splits.

The current training implementation includes a VLM Query backbone, RGB-pixel/pose auxiliary heads, and a Pi-style prefix/suffix action expert. **It does not establish full implementation of every mechanism proposed in the manuscript**; see the status matrix.

## Repository

| Path | Contents |
| --- | --- |
| `code/atomvla-semantic-data/` | Conversion, atomic segmentation, semantic binding, event keyframes, release gates and tests |
| `training/` | Current two-stage full-parameter trainer, model, collator, checkpoint runtime and index builders |
| `configs/` | Training configurations with portable example paths |
| `code/atomvla-evaluation/` | Fixed-manifest runner, metrics, CSV/JSON export, TensorBoard and adapter contracts |
| `code/atomvla-endpoints/` | Earlier LIBERO/RoboTwin endpoint execution pilots and geometry audits |
| `code/atomvla-atomic-runtime/` | Earlier executable primitive demonstration |
| `paper/` | LaTeX, bibliography, figures, editable pipeline slides and synthetic plotting code |
| `docs/` | Reproduction, implementation limits, attribution and release manifest |

## Quick check

Python 3.12 is recommended. The lightweight suite does not require robot hardware or model weights.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-test.txt
python scripts/check_release.py
python -m unittest discover -s code/atomvla-evaluation/tests -v
python -m unittest discover -s code/atomvla-semantic-data/production -v
```

For training dependencies, install `requirements-train.txt` in a compatible PyTorch/CUDA environment, download the base model under its upstream license, and follow [REPRODUCE.md](docs/REPRODUCE.md). Existing conversion and audit tools expect dataset-specific receipts and assets; they are not a one-command download of a released training set.

## Evaluation boundaries

The supplied environment adapters are **extension points**, not completed simulator integrations. Variant names are dispatched to backends; the generic runner does not implement model ablations itself. Current fixtures are protocol examples, not official exhaustive benchmark manifests. The paper proposes RoboMME while the older memory fixture still uses RMBench. Feature extraction currently consumes supplied paired hidden states or a user-provided extraction hook.

An earlier allocation on one H100 completed interface-only smoke bookkeeping. That job read checkpoint `READY.json` metadata but did not load weights, exercise GPU inference, or execute learned-policy robot rollouts. It must not be cited as proof that the full evaluation implementation passed GPU validation.

## License and attribution

Original project code is released under [Apache-2.0](LICENSE). The VLA-Delta-derived files retain their upstream license and provenance. Dataset assets, simulator assets, pretrained models and third-party figure/template content retain their respective licenses; this release does not grant rights to those assets. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
