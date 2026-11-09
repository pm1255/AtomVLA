import json,tempfile,unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parents[1]))
from schema import Episode, load_manifest
from runner import Runner
from backends import SmokeBackend
from metrics import summarize

class Eval(unittest.TestCase):
 def test_smoke_runner_preserves_variants(self):
  with tempfile.TemporaryDirectory() as d:
   e=[Episode('e','libero','t','test',0,'x')]
   rows=Runner(SmokeBackend(True),d).run(e,['full','no_query'])
   self.assertEqual(len(rows),2);self.assertTrue(all(x['success'] for x in rows))
   self.assertEqual(json.loads((Path(d)/'summary.json').read_text())['n_trials'],2)
 def test_unknown_benchmark(self):
  with self.assertRaises(ValueError): Episode('e','bad','t','test',0,'x').validate()
 def test_manifest_and_error_are_preserved(self):
  manifest=Path(__file__).parents[1]/'manifests'/'robotwin_three_tasks.jsonl'
  eps=load_manifest(manifest)
  self.assertEqual(len(eps),9)
  class Unavailable:
   def run(self, episode, variant): return {'success':False,'error':'unavailable: no simulator'}
  with tempfile.TemporaryDirectory() as d:
   rows=Runner(Unavailable(),d).run(eps[:1],['full'])
   self.assertEqual(rows[0]['error'],'unavailable: no simulator')
   self.assertEqual(summarize(rows)['errors'],1)

if __name__=='__main__':unittest.main()
