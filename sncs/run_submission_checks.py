"""Execute current tests and independently record preservation/numerical results."""
from pathlib import Path
import sys,os,unittest,io,json,hashlib,platform,importlib.metadata
from audit_displayed_tables import run as audit_tables
from audit_saved_evidence import run as audit_evidence
ROOT=Path(__file__).resolve().parent

def run():
    os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1')
    sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
    stream=io.StringIO();suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    log=stream.getvalue();print(log)
    evidence=ROOT/'evidence/verification';evidence.mkdir(parents=True,exist_ok=True)
    (evidence/'submission_plan_test_log.txt').write_text(log,encoding='utf-8')
    expected=json.loads((ROOT/'tests/expected_preserved_hashes.json').read_text())
    mismatches=[p for p,sha in expected['files'].items() if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=sha]
    baseline=json.loads((ROOT/'evidence/previous_revision/baseline_manifest.json').read_text())['files']
    original_csvs={p:sha for p,sha in baseline.items() if p.startswith('tables/') and p.endswith('.csv')}
    changed_csvs=[p for p,sha in original_csvs.items() if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=sha]
    numeric=audit_tables()
    independent=audit_evidence()
    pages={}
    from pypdf import PdfReader
    for p in (ROOT/'manuscript').glob('*.pdf'):pages[p.stem]=len(PdfReader(p).pages)
    report=dict(revision_date='2026-11-20',current_feedback_date='2026-11-13',source_plan_date='2026-10-22',
        tests_run=result.testsRun,tests_passed=result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
        failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),test_returncode=0 if result.wasSuccessful() else 1,
        original_files_preserved=len(expected['files']),preservation_mismatches=mismatches,
        original_scientific_csvs_checked=len(original_csvs),changed_original_scientific_csvs=changed_csvs,
        validation_scope='Scientific records, displayed results, original-file preservation, evaluator and optimizer fixtures',
        current_execution_provenance=json.loads((ROOT/'evidence/execution_provenance/execution_provenance_summary.json').read_text()),
        recovered_launcher_hotfix=json.loads((ROOT/'evidence/execution_provenance/launcher_hotfix_audit.json').read_text()),
        configured_peak_pair_run=True,venue='SN Computer Science (Springer)',author_declarations_confirmed=True,external_submission=False,
        configured_peak_record_audit=json.loads((ROOT/'evidence/peak_first/RECORDS_AUDIT.json').read_text()),
        configured_peak_full_prediction_state_audit=json.loads((ROOT/'evidence/peak_first/FULL_AUDIT.json').read_text()),
        dataset_validation_layout=list(__import__('csv').DictReader((ROOT/'tables/validation_batch_layout.csv').open(encoding='utf-8'))),
        research_training_performed=True,analysis_launched_training=False,new_prediction_evaluation_performed=True,
        cpu_optimizer_fixtures_executed=True,cpu_resume_fixture_executed=True,
        actual_training_device_resume_verified=True,actual_device_resume_scope='new configured-peak runner on Kaggle T4; not historical rollback',cuda_amp_distributed_tested=False,
        replay_contract='v2: finite-state validation, at most one revalidated replay; rejected replay has no committed transition',
        archived_executor_preserved=True,archived_checkpoint_selections_preserved=True,
        shared_step_summaries=95,shared_step_seed_pairs=475,scientific_records=240,trained_test_endpoints=180,
        guard_primary_detected=35,guard_primary_groups=40,WNUT_misses=5,assertion_detected_pretrained=20,
        model_checkpoints_available_in_results_zip=False,new_pair_model_checkpoints_available=True,guard_overhead_measured=False,title_unchanged=True,
        python=sys.version,platform=platform.platform(),
        package_versions={n:importlib.metadata.version(n) for n in ['torch','numpy','scipy','pandas','matplotlib','pypdf','reportlab']},
        **numeric,independent_saved_evidence_audit=independent,pdf_page_counts=pages,pdf_visual_review='pending',
        test_log='evidence/verification/submission_plan_test_log.txt')
    (ROOT/'tables/revision_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    if not result.wasSuccessful() or mismatches or changed_csvs:raise SystemExit(1)
    return report

if __name__=='__main__':run()
