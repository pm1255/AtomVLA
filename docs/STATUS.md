# Implementation and evidence status

This is a source release, not a claim that all paper experiments are implemented.

| Component | Available | Still required |
| --- | --- | --- |
| Annotation | Event segmentation, discrete arguments, endpoint and variable keyframe labels, evidence masks, audits | Independent semantic accuracy verification and accepted complete training release |
| Training | Current server snapshot: full VLM training, terminal-pose/RGB heads, prefix action expert, two stages, checkpoint reload checks | Full training on the accepted dataset and verification of all paper mechanisms |
| Future imagination | RGB auxiliary training exists | Full primitive-dependent multi-keyframe imagined-rollout implementation and validation |
| Benchmark evaluation | Manifest dispatch, metrics, CI, errors, CSV/JSON | Real policy/environment adapters, official task IDs, complete test episodes |
| Ablations and coupling | Named variants and dispatch | Independently trained variants, feedback mechanisms, matched closed-loop validation |
| Memory | Legacy three-task RMBench fixture | Paper-aligned RoboMME manifest, normal VLA baselines and actual memory handling |
| Feature analysis | Paired NPZ/PCA processing and optional extractor interface | A checkpoint-backed extractor and measured representations |
| Real robot | Five task names and adapter interface | Robot-specific control, calibrated coordinates, task definitions and real execution |
| Endpoint pilots | LIBERO and RoboTwin demonstration/endpoint scripts | These are not learned AtomVLA benchmark rollouts |

## Prior verification

A separate training smoke was recorded as eight steps per stage with checkpoints. That is not a completed training run. The later evaluation job's 183 trials used an unconditional mock response; its checkpoint check only parsed metadata. The public release deliberately does not distribute those mock successes as experimental results.

Current evaluation manifests include names such as aggregate LIBERO suites, which are not executable episode IDs. LIBERO-Pro fixture names and RMBench memory fixture also differ from the current paper protocol. Do not use them to claim official benchmark coverage.

Paper numbers and synthetic features are proposal scenarios, explicitly labeled in the LaTeX. No empirical ranking, improvement claim, or real-robot success rate is established by this repository.

Runtime configuration paths under `/opt/atomvla` are examples, not bundled data. Some historical conversion helpers retain environment assumptions. Adapt them to the local datasets and keep audit gates enabled.
