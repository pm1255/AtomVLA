"""Check published source integrity without loading a model or simulator."""
import ast
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
errors = []
files = [p for p in ROOT.rglob('*') if p.is_file() and not any(
    part in ('.git', '__pycache__', '.venv', 'runs', '.pytest_cache') for part in p.relative_to(ROOT).parts)]
for path in files:
    if path.suffix == '.py':
        try:
            ast.parse(path.read_text())
        except SyntaxError as exc:
            errors.append(f'{path.relative_to(ROOT)}: {exc}')
    if path.suffix in ('.json', '.jsonl'):
        try:
            text = path.read_text()
            if path.suffix == '.jsonl':
                for line in text.splitlines():
                    if line.strip(): json.loads(line)
            else: json.loads(text)
        except ValueError as exc:
            errors.append(f'{path.relative_to(ROOT)}: invalid JSON: {exc}')
for path in (ROOT / 'paper').rglob('*.tex'):
    text = '\n'.join(line.split('%')[0] for line in path.read_text().splitlines())
    for command, target in re.findall(r'\\(input|includegraphics|bibliography)(?:\[[^\]]*\])?\{([^}]+)\}', text):
        extension = {'input': '.tex', 'bibliography': '.bib', 'includegraphics': '.pdf'}[command]
        ref = ROOT / 'paper' / target
        if not ref.exists() and not ref.with_suffix(extension).exists():
            errors.append(f'{path.relative_to(ROOT)}: missing {target}')
manifest = ROOT / 'docs/FILE_MANIFEST.json'
if manifest.exists():
    for entry in json.loads(manifest.read_text()):
        path = ROOT / entry['path']
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            errors.append(f"manifest mismatch: {entry['path']}")
print(json.dumps({'passed': not errors, 'files_checked': len(files), 'errors': errors}, indent=2))
sys.exit(bool(errors))
