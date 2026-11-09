"""Full-dataset completion gate: candidate releases cannot satisfy this contract."""
import json
from pathlib import Path

def require_complete_dataset(audit_path):
 report=json.loads(Path(audit_path).read_text())
 counts=report.get('counts',{})
 if set(counts)!={'droid','libero','robotwin'}:raise ValueError('required three-source audit missing')
 for source,c in counts.items():
  total=c.get('source_episodes',0)
  if total<=0 or c.get('fully_annotated_episodes',0)!=total or c.get('audit_errors',0) or c.get('incomplete_episodes',0):
   raise ValueError(f'{source}: incomplete full-episode labels; partial candidates cannot be released as complete')
 if report.get('status')!='coverage_audit_completed_not_annotation_completed':raise ValueError('unexpected audit status')
 return report
