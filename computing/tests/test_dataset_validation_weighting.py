"""Check actual dataset layouts and the weighting consequence independently."""
from pathlib import Path
import csv,json,zipfile,unittest,math
ROOT=Path(__file__).resolve().parents[1]

class DatasetValidationWeighting(unittest.TestCase):
    def test_all_archived_scratch_layouts_match_export_and_selection_scope(self):
        with (ROOT/'tables/archived_evaluator_provenance.csv').open(newline='',encoding='utf-8') as f:
            exports={r['source']:r for r in csv.DictReader(f)}
        self.assertEqual(len(exports),120)
        counts={}
        with zipfile.ZipFile(ROOT/'data/ACML_RESULTS_FOR_STUDENT.zip') as z:
            for path,row in exports.items():
                result=json.loads(z.read(path));n=result['dataset_manifest']['split']['validation_count']
                ds=result['dataset'];counts[ds]=counts.get(ds,0)+1
                self.assertEqual(int(row['validation_examples']),n)
                expected=(18000,562,16,563) if ds=='ag_news' else (84000,2625,32,2625)
                self.assertEqual((n,int(row['validation_full_batches']),int(row['validation_final_batch_examples']),int(row['validation_batches'])),expected)
                self.assertEqual(row['validation_batch_weighting_matches_example_weighting'],'False' if ds=='ag_news' else 'True')
                self.assertEqual(row['partial_batch_selection_effect_status'],'unknown_without_candidate_epoch_assets' if ds=='ag_news' else 'not_applicable_full_validation_batches')
                self.assertEqual(row['selection_rule'],'minimum_archived_validation_batch_mean_CE')
                self.assertEqual(row['historical_selection_preserved'],'True')
        self.assertEqual(counts,{'ag_news':60,'dbpedia_14':60})

    def test_dataset_layouts_produce_the_stated_example_weights(self):
        with (ROOT/'tables/validation_batch_layout.csv').open(newline='',encoding='utf-8') as f:
            layouts={r['dataset']:r for r in csv.DictReader(f)}
        # Each example in a batch has weight 1/(number_of_batches * batch_size).
        for ds,row in layouts.items():
            n=int(row['validation_examples']);b=int(row['validation_batch_size']);last=int(row['validation_final_batch_examples'])
            batches=int(row['validation_batches']);ratio=(1/(batches*last))/(1/n)
            self.assertAlmostEqual(float(row['final_batch_example_influence_ratio']),ratio,places=14)
            if ds=='ag_news':
                self.assertAlmostEqual(ratio,1.9982238010657194,places=14)
                self.assertNotEqual(1/(batches*b),1/n)
            else:
                self.assertEqual(ratio,1.)
                self.assertEqual(1/(batches*b),1/n)
                self.assertEqual(n%b,0)
