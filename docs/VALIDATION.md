# Release validation

Local release checks were run with Python 3.12.14 in a clean virtual environment.

- Evaluation runner: 3 unit tests passed.
- Conversion/annotation production suite: 79 unit tests passed.
- Endpoint geometry: 4 unit tests passed.
- Atomic runtime/controller contract: 10 unit tests passed.
- Python syntax, JSON/JSONL parsing and referenced LaTeX input/figure paths checked.
- Publication text scanned for common credential formats, internal account paths and hostnames; no matches found. Editable slide XML checked separately for private paths and external relationships.

These are source/contract checks, not real simulator rollouts, benchmark accuracy validation, full model training, or a LaTeX compilation check. Dependencies in the training requirements are compatibility ranges, not a fully verified lockfile. CI runs the lightweight checks on Python 3.12.
