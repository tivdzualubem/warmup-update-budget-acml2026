"""Persist our reproducible CPU counterexamples, separately from scientific runs."""
from pathlib import Path
import ast,json,csv,hashlib,math
import numpy as np
import torch
import torch.nn.functional as F
from future_runs.scratch_evaluator import evaluate as evaluate_v2
from future_runs.transition_executor import transactional_step as step_v2
import importlib.util

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'source/warmup-update-budget-acml2026/publication_expansion/src'

def fixture():
    m=torch.nn.Linear(1,1,bias=False,dtype=torch.float64)
    with torch.no_grad():m.weight.fill_(1.)
    o=torch.optim.Adam(m.parameters(),lr=1.,betas=(.9,.98),eps=1e-9)
    m.weight.grad=torch.ones_like(m.weight)
    return m,o

def build():
    source=(SOURCE/'scratch_job_runner.py').read_text(encoding='utf-8');tree=ast.parse(source)
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('evaluate','macro_f1')]
    ns={'torch':torch,'F':F,'np':np};exec(compile(ast.Module(body=nodes,type_ignores=[]),'archived_evaluator','exec'),ns)
    class Identity(torch.nn.Module):
        def forward(self,x):return x
    model=Identity();extreme=[{'input_ids':torch.tensor([[1000.,-1000.]]),'labels':torch.tensor([1])}]
    logits=torch.tensor([[0.,0.]]*32+[[0.,10.]],dtype=torch.float64);labels=torch.zeros(33,dtype=torch.long)
    partial=[{'input_ids':logits[:32],'labels':labels[:32]},{'input_ids':logits[32:],'labels':labels[32:]}]
    old_extreme=ns['evaluate'](model,extreme,'cpu',2);new_extreme=evaluate_v2(model,extreme,'cpu',2)
    old_partial=ns['evaluate'](model,partial,'cpu',2);new_partial=evaluate_v2(model,partial,'cpu',2)
    spec=importlib.util.spec_from_file_location('probe_archived_executor',SOURCE/'transition_executor.py')
    import sys
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    m,o=fixture();m.weight.grad.fill_(float('nan'))
    bad=module.transactional_step(model=m,optimizer=o,intended_lr=1e-4,threshold=.001,mode='rescue')
    m,o=fixture();legacy=module.transactional_step(model=m,optimizer=o,intended_lr=.1,threshold=.001,mode='rescue')
    m,o=fixture();new=step_v2(model=m,optimizer=o,intended_lr=.1,threshold=.001,mode='rescue')
    report=dict(scope='our tiny artificial CPU counterexamples, not historical training outcomes',torch=torch.__version__,
        extreme_logits=dict(archived_logit_ce=old_extreme['loss'],archived_clipped_probability_nll=old_extreme['nll'],v2_logit_ce=new_extreme['loss']),
        uneven_batch=dict(batch_sizes=[32,1],archived_mean_of_batch_means=old_partial['loss'],v2_sample_weighted_ce=new_partial['loss']),
        archived_nonfinite_candidate=dict(ratio_is_finite=math.isfinite(bad.candidate_update_ratio),guard_triggered=bad.guard_triggered,stopped=bad.stopped),
        replay_ceiling=dict(threshold=.001,intended_lr=.1,archived_committed_ratio=legacy.committed_update_ratio,archived_stopped=legacy.stopped,
            v2_replay_ratio=new.replay_update_ratio,v2_stopped=new.stopped,v2_restored_weight=float(m.weight.item()),v2_failure_reason=new.failure_reason,v2_optimizer_calls=new.optimizer_step_calls))
    assert old_extreme['loss']==2000. and new_extreme['loss']==2000.
    assert bad.guard_triggered is False and not math.isfinite(bad.candidate_update_ratio)
    assert legacy.committed_update_ratio>.001 and new.stopped and float(m.weight.item())==1.
    out=ROOT/'evidence/verification';out.mkdir(exist_ok=True,parents=True)
    (out/'our_cpu_counterexamples.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    findings=[]
    for file,needle,issue,replacement in [
        ('scratch_job_runner.py','clipped = np.clip','clipped-probability NLL floor1e-12','future_runs/scratch_evaluator.py'),
        ('scratch_job_runner.py','losses.append','equal weighting of batch means','future_runs/evaluation.py'),
        ('transition_executor.py','triggered = candidate_ratio >','nonfinite comparison can bypass threshold','future_runs/transition_executor.py'),
        ('transition_executor.py','committed_ratio = update_ratio','replay result is not revalidated','future_runs/transition_executor.py')]:
        p=SOURCE/file;lines=p.read_text(encoding='utf-8').splitlines();line=next(i for i,s in enumerate(lines,1) if needle in s)
        findings.append(dict(frozen_source=p.relative_to(ROOT).as_posix(),source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),line=line,
            source_issue=issue,separate_v2_implementation=replacement,observed_training_failure_claimed=False))
    with (out/'our_source_inspection.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(findings[0]));w.writeheader();w.writerows(findings)
    print(json.dumps(report,indent=2))

if __name__=='__main__':build()
