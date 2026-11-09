import unittest
from source_events import binding,infer
import test_production
class SourceEvents(unittest.TestCase):
 def test_language_binding(self):
  self.assertEqual(binding('pick up the book on the left and place it on top of the shelf')['object'],'book_on_the_left')
  self.assertIsNone(binding(''));self.assertIsNone(binding('open the drawer'));self.assertIsNone(binding('put the bowl and the cup on the table'))
 def test_libero_aperture(self):
  p,g,s,i=test_production.Contracts().fixture();a,issues=infer(p[:,:1],g[:,:1]*.06+.02,'libero','put the bowl on the plate')
  self.assertEqual([x['verb'] for x in a],['pick','place']);self.assertFalse(a[0]['evidence']['physical_completion_verified'])
 def test_droid_missing_text_not_invented(self):
  p,g,s,i=test_production.Contracts().fixture();self.assertFalse(infer(p[:,:1],g[:,:1],'droid','')[0])
if __name__=='__main__':unittest.main()
