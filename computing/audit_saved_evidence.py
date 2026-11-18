"""Independent arithmetic and trace audit; reads original records, never trains.

This deliberately does not import the analysis builders. The fixed t(4) 97.5%
quantile is independently applied to directly reconstructed paired differences.
"""
from pathlib import Path
from collections import Counter
import csv, json, math, hashlib, zipfile

ROOT = Path(__file__).resolve().parent
SEEDS = (42, 123, 456, 789, 2026)
PAIRS = {
    'stable_minus_single_bad': ('stable_with_guard', 'single_bad_first_step_then_stable'),
    'rescue_minus_late': ('buggy_rescue_guard', 'buggy_no_guard'),
    'corrected_minus_rescue': ('fixed_order_with_guard', 'buggy_rescue_guard'),
}


def rows(name):
    with (ROOT / 'tables' / name).open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def moments(values):
    mean = math.fsum(values) / len(values)
    sd = math.sqrt(math.fsum((x-mean)**2 for x in values)/(len(values)-1))
    return mean, sd


def close(actual, expected):
    assert math.isclose(float(actual), expected, rel_tol=1e-10, abs_tol=1e-10), (actual, expected)


def nonfinite_paths(value, path=''):
    if isinstance(value, dict):
        return [p for k,v in value.items() for p in nonfinite_paths(v, path+'.'+k)]
    if isinstance(value, list):
        return [p for i,v in enumerate(value) for p in nonfinite_paths(v, path+f'[{i}]')]
    return [path] if isinstance(value,float) and not math.isfinite(value) else []


def run():
    archive = ROOT / 'data/ACML_RESULTS_FOR_STUDENT.zip'
    histories, results, trace_count, trace_rows, failures = {}, {}, 0, 0, []
    scientific_records = trained = 0
    with zipfile.ZipFile(archive) as z:
        for member in sorted(z.namelist()):
            if '/results/' not in member or not member.endswith('/result.json'):
                continue
            r=json.loads(z.read(member))
            if not r.get('job_complete') or 'condition' not in r:
                continue
            scientific_records += 1
            trained += bool(r.get('test_metrics'))
            trace_member=member.replace('result.json','trace.json')
            trace=json.loads(z.read(trace_member))
            trace_count += 1; trace_rows += len(trace)
            failures.extend({'source':trace_member,'field':p} for p in nonfinite_paths(trace))
            if '/results/scratch/' in member:
                k=(r['dataset'],r['model'],r['condition'],int(r['seed']))
                history=json.loads(z.read(member.replace('result.json','history.json')))
                mapped={int(h['global_step']):h for h in history}
                assert len(mapped)==len(history)
                histories[k]=mapped; results[k]=r
        summaries=rows('matched_step_summary.csv')
        seed_rows=rows('matched_step_per_seed.csv')
        coverage=rows('matched_step_coverage.csv')
        summary_index={(r['dataset'],r['model'],r['contrast'],int(r['global_step'])):r for r in summaries}
        seed_index={(r['dataset'],r['model'],r['contrast'],int(r['global_step']),int(r['seed'])):r for r in seed_rows}
        coverage_index={(r['dataset'],r['model'],r['contrast']):r for r in coverage}
        assert len(summary_index)==len(summaries) and len(seed_index)==len(seed_rows)
        expected_keys=set(); expected_seed_keys=set(); summary_checks=0
        for ds,model in sorted({k[:2] for k in histories}):
            for label,(a,b) in PAIRS.items():
                observed=[set(histories[ds,model,c,s]) for c in (a,b) for s in SEEDS]
                common=set.intersection(*observed); union=set.union(*observed)
                cov=coverage_index[ds,model,label]
                parsed=lambda text:set(map(int,text.split(';'))) if text else set()
                assert parsed(cov['shared_logged_steps'])==common
                assert parsed(cov['excluded_nonshared_steps'])==union-common
                for step in sorted(common):
                    key=(ds,model,label,step); expected_keys.add(key)
                    aa=[]; bb=[]; diffs=[]
                    for seed in SEEDS:
                        skey=key+(seed,); expected_seed_keys.add(skey)
                        out=seed_index[skey]
                        ah=histories[ds,model,a,seed][step]; bh=histories[ds,model,b,seed][step]
                        av=ah['validation']['accuracy']; bv=bh['validation']['accuracy']
                        d=100*(av-bv); aa.append(av); bb.append(bv); diffs.append(d)
                        for field,value in [('arm_a_validation_accuracy',av),('arm_b_validation_accuracy',bv),
                                            ('accuracy_difference_pp',d),('arm_a_archived_validation_loss',ah['validation']['loss']),
                                            ('arm_b_archived_validation_loss',bh['validation']['loss']),
                                            ('archived_validation_loss_difference',ah['validation']['loss']-bh['validation']['loss'])]:
                            close(out[field],value)
                        for arm in ('a','b'):
                            source=out['arm_'+arm+'_source']; raw=json.loads(z.read(source))
                            assert next(h for h in raw if int(h['global_step'])==step)==(ah if arm=='a' else bh)
                    out=summary_index[key]; mean,sd=moments(diffs)
                    radius=2.7764451051977987*sd/math.sqrt(5)
                    ma,sa=moments(aa); mb,sb=moments(bb)
                    pos=sum(v>0 for v in diffs); neg=sum(v<0 for v in diffs); n=pos+neg
                    sign_p=min(1.,2*sum(math.comb(n,i) for i in range(min(pos,neg)+1))/2**n) if n else 1.
                    for field,value in [('mean_accuracy_difference_pp',mean),('sample_sd_difference_pp',sd),
                                        ('descriptive_ci95_lower_pp',mean-radius),('descriptive_ci95_upper_pp',mean+radius),
                                        ('arm_a_validation_accuracy_mean',ma),('arm_a_validation_accuracy_sample_sd',sa),
                                        ('arm_b_validation_accuracy_mean',mb),('arm_b_validation_accuracy_sample_sd',sb),
                                        ('exact_sign_p',sign_p)]:
                        close(out[field],value); summary_checks+=1
        assert set(summary_index)==expected_keys and set(seed_index)==expected_seed_keys
    stable=[(k,h) for k,h in histories.items() if k[2] in PAIRS['stable_minus_single_bad']]
    crossed=Counter(); equal_postzero=0; selected_postzero=0
    for k,h in stable:
        ordered=[h[s] for s in sorted(h)]
        crossed[k[0]]+=max(h,default=0)>=50000
        if k[0]=='dbpedia_14':
            equal_postzero+=sum(a['global_step']>=50000 and b['global_step']>=50000 and
                a['validation']['loss']==b['validation']['loss'] for a,b in zip(ordered,ordered[1:]))
        best=min(ordered,key=lambda x:x['validation']['loss'])
        close(results[k]['best_validation_loss'],best['validation']['loss'])
        selected_postzero+=best['global_step']>=50000
    sensitivity=rows('retrospective_kappa_sensitivity.csv')
    wnut=[r for r in sensitivity if r['dataset']=='wnut_17' and float(r['kappa'])==10 and r['kind']=='pretrained']
    # Dataset spelling is preserved exactly as recorded in the original export.
    if not wnut:
        wnut=[r for r in sensitivity if 'wnut' in r['dataset'].lower() and float(r['kappa'])==10]
    assert len(wnut)==5
    factors=[float(r['candidate_update_ratio'])/float(r['safe_calibration_max']) for r in wnut]
    assert all(2 < f < 2.11 for f in factors)
    assert all(float(r['candidate_update_ratio']) < float(r['threshold']) for r in wnut)
    assert (scientific_records,trained,trace_count,trace_rows)==(240,180,240,55535)
    assert not failures
    assert (len(expected_keys),len(expected_seed_keys),len(coverage))==(95,475,12)
    assert dict(crossed)=={'ag_news':9,'dbpedia_14':20}
    assert equal_postzero==91 and selected_postzero==14
    report=dict(audit_date='2026-11-07',archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        independent_of_analysis_builders=True,shared_step_summary_rows=len(expected_keys),
        paired_seed_rows=len(expected_seed_keys),summary_numeric_fields_checked=summary_checks,
        source_pair_numeric_fields_checked=6*len(expected_seed_keys),coverage_groups=len(coverage),
        all_shared_steps_retained=True,scientific_records=scientific_records,trained_test_endpoints=trained,
        scientific_trace_files=trace_count,scientific_trace_rows=trace_rows,nonfinite_trace_fields=failures,
        stable_family_runs_crossing_zero=dict(crossed),DBpedia_equal_consecutive_postzero_losses=equal_postzero,
        stable_family_selected_epoch_at_or_after_zero=selected_postzero,
        WNUT_candidate_to_fold_safe_max_factors=factors,WNUT_mean_factor=math.fsum(factors)/len(factors),
        WNUT_primary_threshold_misses=5,discrepancies=[],research_training=False)
    out=ROOT/'evidence/verification/independent_saved_evidence_audit.json'
    out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2)); return report


if __name__=='__main__':run()
