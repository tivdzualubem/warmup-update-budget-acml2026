from pathlib import Path
import sys,unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

class FinalEditorialAudit(unittest.TestCase):
 def test_displayed_numbers_match_the_source_records(self):
  from audit_displayed_tables import run
  result=run();self.assertEqual(result['displayed_numeric_cells_checked'],560);self.assertEqual(result['discrepancies'],0)

 def test_requested_scientific_content_and_removed_administrative_prose(self):
  from pypdf import PdfReader
  forbidden=['What is still unknown','No additional files are required','old-chat','earlier handoff','guarantee acceptance','submission readiness','separately authorized regeneration','No message has been sent']
  for name in ['Research_Manuscript','All_Seed_Results_and_Provenance']:
   md=(ROOT/'manuscript'/(name+'.md')).read_text(encoding='utf-8')
   pdf=' '.join(p.extract_text() for p in PdfReader(ROOT/'manuscript'/(name+'.pdf')).pages)
   for term in forbidden:
    self.assertNotIn(term,md);self.assertNotIn(term,pdf)
  main=(ROOT/'manuscript/Research_Manuscript.md').read_text(encoding='utf-8')
  for phrase in ['exploratory','0.0625','35/40','20/20','extreme injected fault','do not establish equivalence',"guard's incremental runtime and memory overhead were not measured",'guard enforcement ends after step 100','maximum validation macro-F1']:
   self.assertIn(phrase,main)
  title=main.splitlines()[0]
  # The public reproducibility package must not depend on private correspondence.
  self.assertEqual(title,'# Separating First-Step Learning-Rate Faults from Scheduler Delay in Adam Training')
  for word in ['First-Step','Scheduler Delay']:self.assertIn(word,title)
