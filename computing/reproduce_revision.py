"""Rebuild analyses/documents/tests from local saved evidence, without research training."""
from pathlib import Path
import sys,runpy,os,argparse
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1')
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--analysis-only',action='store_true',help='Rebuild saved-record analyses and Markdown; skip LaTeX compilation.')
args=parser.parse_args()
steps=[('reanalyse.py',['--archive',str(ROOT/'data/ACML_RESULTS_FOR_STUDENT.zip'),'--outdir',str(ROOT/'tables')]),
       ('analyse_condition_budgets.py',[]),
       ('extract_historical_sweep.py',['--ag',str(ROOT/'data/Transformer_Artifacts(AGNEWS).ipynb'),'--db',str(ROOT/'data/Transformer_Artifacts(DBPEDIA).ipynb'),'--out',str(ROOT/'tables')]),
       ('make_figures.py',[]),('analyse_matched_steps.py',[]),('make_submission_figures.py',[]),
       ('record_cpu_probes.py',[]),('audit_execution_provenance.py',[]),('audit_launcher_hotfix.py',[]),
       ('examples/first_two_steps.py',['--output',str(ROOT/'evidence/verification/first_two_steps.json')]),
       ('analyse_peak_pair.py',[]),('make_peak_figures.py',[]),
       ('revise_for_computing.py',[]),('make_computing_figures.py',[])]
if not args.analysis_only:steps.append(('build_computing_latex.py',[]))
steps.append(('run_submission_checks.py',[]))
for name,args in steps:
    print('Reproduce:',name,flush=True)
    sys.argv=[str(ROOT/name)]+args
    runpy.run_path(str(ROOT/name),run_name='__main__')
