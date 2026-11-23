"""Audit the returned fixed-budget pair. No training or checkpoint reselection.

Default: reconstruct all tables and verify complete trace/record evidence.
Optional --predictions and --states: independently audit saved logits and tensors.
"""
from pathlib import Path
import argparse,copy,csv,gc,hashlib,io,json,math,sys,zipfile
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'peak_pair_runner/frozen_source'))
from publication_expansion.src.anchor import object_sha256


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def jd(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def write_csv(name,rows):
    with (ROOT/'tables'/name).open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def close(a,b,tol=1e-10):assert math.isclose(a,b,rel_tol=tol,abs_tol=tol),(a,b)


def check_archive(path,group,expected=None):
    z=zipfile.ZipFile(path)
    manifest=json.loads(z.read('ARTIFACT_MANIFEST.json'))
    if expected is not None:assert manifest==expected
    expected_names={n for n,v in manifest.items() if v['bundle']==group}
    assert set(z.namelist())==expected_names|{'ARTIFACT_MANIFEST.json'}
    for name in sorted(expected_names):
        h=hashlib.sha256();count=0
        with z.open(name) as stream:
            for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block);count+=len(block)
        assert h.hexdigest()==manifest[name]['sha256'] and count==manifest[name]['bytes'],name
    return z,manifest,{'file':Path(path).name,'bytes':Path(path).stat().st_size,'sha256':sha(path),'verified_members':len(expected_names)}


def stable(t):
    if t<=8000:return 1e-4*t/8000
    progress=(t-8000)/42000
    return 1e-4*.5*(1+math.cos(math.pi*progress))


def audit(predictions=None,states=None):
    torch.set_num_threads(1)
    evidence=ROOT/'evidence/peak_first';evidence.mkdir(parents=True,exist_ok=True)
    records,manifest,archive_info=check_archive(ROOT/'data/peak_first/ACML_Peak_First_RESULTS.zip','RESULTS')
    read=lambda n:json.loads(records.read(n))
    result=read('run/result.json');config=read('run/protocol.json');history=read('run/evaluation_history.json')
    online=read('run/online_training_history.json');arms=('clean','peak_first')
    assert result['status']=='completed' and result['protocol']==config
    assert config['budget_steps']==49999 and config['seed']==42 and config['dataset']=='dbpedia_14' and config['model']=='medium-4'
    assert config['evaluation_steps']==[0,1,100,1000,8000,10000,20000,30000,40000,49999]
    assert result['fixed_positive_lr_steps_per_arm']==49999 and result['zero_lr_steps_per_arm']==0
    for key,name in [('protocol','protocol'),('code','source_hashes'),('data','data_manifest'),('environment','environment')]:
        assert jd(read('run/'+name+'.json'))==result['identity'][key]
    assert result['identity']==read('preflight/preflight.json')['identity']
    for name,h in read('run/source_hashes.json').items():
        assert hashlib.sha256(records.read('code/'+name)).hexdigest()==h
        assert sha(ROOT/'peak_pair_runner'/name)==h
    preflight=read('preflight/preflight.json')
    assert preflight['resume_probe_exact'] and preflight['first_transition_anchors_match']
    expected_keys={(a,p,t) for a in arms for p in ('validation','train_monitor_subset') for t in config['evaluation_steps']}|{(a,'test',49999) for a in arms}
    assert len(history)==42 and {(r['arm'],r['population'],r['step']) for r in history}==expected_keys
    bykey={(r['arm'],r['population'],r['step']):r for r in history}
    with np.load(io.BytesIO(records.read('run/split_indices.npz')),allow_pickle=False) as arrays:
        split={k:arrays[k] for k in arrays.files}
    rng=np.random.RandomState(42);permutation=rng.permutation(560000)
    assert np.array_equal(split['validation'],permutation[:84000]) and np.array_equal(split['train'],permutation[84000:])
    assert np.array_equal(split['test'],np.arange(70000)) and np.array_equal(split['train_monitor'],split['train'][:8192])
    h=hashlib.sha256();arr=split['validation'];h.update(str(arr.dtype).encode());h.update(str(arr.shape).encode());h.update(arr.tobytes())
    assert h.hexdigest()==config['expected_validation_indices_sha256']
    populations={'validation':84000,'train_monitor_subset':8192,'test':70000}
    index_arrays={'validation':split['validation'],'train_monitor_subset':split['train_monitor'],'test':split['test']}
    for row in history:
        assert row['num_examples']==populations[row['population']] and math.isfinite(row['logit_ce'])
        assert row['prediction_sha256']==manifest['run/predictions/'+row['prediction_file']]['sha256']
        if row['step']==49999:assert result['final_metrics'][row['arm']][row['population']]==row
    # Reconstruct the shared sampler independently of the runner; compare every
    # logged permutation/cursor and batch-index hash, including epoch boundaries.
    generator=torch.Generator().manual_seed(42)
    sampler_permutation=torch.randperm(476000,generator=generator);cursor=0;epoch=0;step=0
    sampler_states={0:dict(permutation=sampler_permutation.clone(),cursor=0,epoch=0,generator_state=generator.get_state().clone())}
    training_seconds={a:0. for a in arms};evaluation_seconds={a:0. for a in arms};checkpoint_seconds=0.;closed_seconds=0.
    windows={a:dict(sum_loss=0.,examples=0,first_step=1) for a in arms};rebuilt_online=[];eval_events=[];checkpoint_events=[];first_record=None
    logs=sorted(n for n in records.namelist() if n.startswith('run/logs/') and n.endswith('.jsonl'))
    assert len(logs)==1,'This archived audit expects the observed uninterrupted single segment.'
    for name in logs:
        for line in records.read(name).decode().splitlines():
            row=json.loads(line);event=row['event']
            assert event!='failure'
            if event=='start':assert row['start_step']==0 and not row['resume'] and row['identity']==result['identity']
            elif event=='paired_step':
                step+=1;assert row['step']==step
                if cursor==476000:
                    sampler_permutation=torch.randperm(476000,generator=generator);cursor=0;epoch+=1
                indices=sampler_permutation[cursor:cursor+32];cursor+=len(indices)
                assert row['batch_indices_sha256']==object_sha256(indices)
                assert row['epoch']==epoch and row['sampler_cursor']==cursor
                if step==1:first_record=row
                for arm in arms:
                    values=row['arms'][arm];expected=1e-4 if arm=='peak_first' and step==1 else stable(step)
                    assert len(values['lr_consumed'])==1 and math.isclose(values['lr_consumed'][0],expected,rel_tol=1e-12,abs_tol=0) and expected>0 and values['examples']==32
                    if step>1:assert values['lr_consumed']==row['arms']['clean']['lr_consumed']
                    assert math.isfinite(values['online_batch_ce']) and math.isfinite(values['instrumented_training_seconds'])
                    if values['candidate_and_committed_ratio'] is not None:assert math.isfinite(values['candidate_and_committed_ratio'])
                    training_seconds[arm]+=values['instrumented_training_seconds']
                    windows[arm]['sum_loss']+=values['online_batch_ce']*32;windows[arm]['examples']+=32
                if step%100==0 or step==49999:
                    for arm in arms:
                        w=windows[arm];rebuilt_online.append(dict(arm=arm,first_step=w['first_step'],last_step=step,num_examples=w['examples'],example_weighted_online_ce=w['sum_loss']/w['examples']))
                        windows[arm]=dict(sum_loss=0.,examples=0,first_step=step+1)
                if step in (1,8000,49999):sampler_states[step]=dict(permutation=sampler_permutation.clone(),cursor=cursor,epoch=epoch,generator_state=generator.get_state().clone())
            elif event=='evaluation':
                assert row['step']==step
                if row['population']=='test':assert step==49999
                eval_events.append((row['arm'],row['population'],row['step']));evaluation_seconds[row['arm']]+=row['seconds']
            elif event=='checkpoint':checkpoint_seconds+=row['seconds'];checkpoint_events.append(row['step'])
            elif event=='session_end':assert row['status']=='completed' and step==49999;closed_seconds+=row['active_seconds']
            else:raise AssertionError(event)
    assert step==49999 and set(eval_events)==expected_keys and len(eval_events)==42
    assert rebuilt_online==online and len(online)==1000
    costs=[]
    for arm in arms:
        cost=result['committed_trajectory_cost'][arm]
        assert cost['positive_lr_steps']==49999 and cost['zero_lr_steps']==0 and cost['examples']==1599968
        close(cost['instrumented_training_seconds'],training_seconds[arm]);close(cost['evaluation_seconds'],evaluation_seconds[arm])
        costs.append(dict(arm=arm,positive_lr_steps=49999,zero_lr_steps=0,examples=1599968,
            instrumented_training_seconds=training_seconds[arm],evaluation_seconds=evaluation_seconds[arm],
            peak_process_cuda_allocated_bytes=cost['peak_process_cuda_allocated_bytes'],peak_process_cuda_reserved_bytes=cost['peak_process_cuda_reserved_bytes']))
    close(checkpoint_seconds,result['shared_cost']['checkpoint_seconds'])
    close(closed_seconds,result['recorded_attempt_cost']['closed_session_active_seconds'])
    assert result['recorded_attempt_cost']['completed_pair_step_events']==49999
    assert all(result['recorded_attempt_cost'][k]==0 for k in ('failure_events','segments_without_end','truncated_final_log_lines'))
    curves=[]
    for population in populations:
        for t in ([49999] if population=='test' else config['evaluation_steps']):
            a,b=[bykey[arm,population,t] for arm in arms]
            curves.append(dict(population=population,step=t,clean_logit_ce=a['logit_ce'],peak_first_logit_ce=b['logit_ce'],
                ce_peak_first_minus_clean=b['logit_ce']-a['logit_ce'],clean_accuracy_pct=100*a['accuracy'],peak_first_accuracy_pct=100*b['accuracy'],
                accuracy_peak_first_minus_clean_pp=100*(b['accuracy']-a['accuracy']),num_examples=a['num_examples']))
    final=next(r for r in curves if r['population']=='validation' and r['step']==49999)
    close(final['ce_peak_first_minus_clean'],result['validation_ce_peak_first_minus_clean'])
    close(final['accuracy_peak_first_minus_clean_pp'],-100*result['validation_accuracy_clean_minus_peak_first'])
    write_csv('peak_first_curves.csv',curves);write_csv('peak_first_online_training.csv',online);write_csv('peak_first_costs.csv',costs)
    report={'status':'passed','scope':'complete saved-record/trace audit','archives':[archive_info],
        'paired_transitions':49999,'consumed_lr_fields_checked':99998,'sampler_batches_checked':49999,'evaluation_records':42,
        'online_training_blocks_checked':1000,'fixed_split_verified':True,'same_source_config_data_environment_as_preflight':True,
        'failure_or_resume_events':0,'closed_session_active_seconds':closed_seconds,'first_transition':first_record,
        'primary_validation_ce_difference':result['validation_ce_peak_first_minus_clean'],'research_training_by_this_analysis':False,
        'new_research_pair_executed_on_kaggle':True,'original_historical_checkpoint_reselection':False}
    write_json(evidence/'RECORDS_AUDIT.json',report)
    if predictions:
        pred,_,info=check_archive(predictions,'PREDICTIONS',manifest);report['archives'].append(info)
        checked=[];label_reference={}
        for row in history:
            name='run/predictions/'+row['prediction_file']
            with np.load(io.BytesIO(pred.read(name)),allow_pickle=False) as payload:
                logits=payload['logits'].astype(np.float64);labels=payload['labels'];ids=payload['example_indices']
            assert logits.shape==(populations[row['population']],14) and np.isfinite(logits).all()
            assert np.array_equal(ids,index_arrays[row['population']])
            assert labels.shape==(len(ids),) and np.issubdtype(labels.dtype,np.integer) and labels.min()>=0 and labels.max()<14
            if row['population'] in label_reference:assert np.array_equal(labels,label_reference[row['population']])
            else:label_reference[row['population']]=labels.copy()
            maximum=logits.max(axis=1);ce=float(np.mean(maximum+np.log(np.exp(logits-maximum[:,None]).sum(axis=1))-logits[np.arange(len(labels)),labels]))
            accuracy=float(np.mean(logits.argmax(axis=1)==labels))
            close(ce,row['logit_ce'],1e-11);assert accuracy==row['accuracy']
            checked.append(dict(arm=row['arm'],population=row['population'],step=row['step'],num_examples=len(labels),
                saved_logit_ce=row['logit_ce'],independent_logit_ce=ce,absolute_ce_difference=abs(ce-row['logit_ce']),
                saved_accuracy=row['accuracy'],independent_accuracy=accuracy))
        write_csv('peak_first_prediction_recheck.csv',checked)
        report['prediction_files_independently_recomputed']=len(checked)
        report['maximum_absolute_ce_recomputation_difference']=max(r['absolute_ce_difference'] for r in checked)
        pred.close()
    if states:
        checkpoints,_,info=check_archive(states,'STATES',manifest);report['archives'].append(info)
        verified=[];initial=None;final_digest=None
        checkpoint_names=['pair_step00000.pt','pair_step00001.pt','pair_step08000.pt','pair_step49999.pt','latest.pt']
        for filename in checkpoint_names:
            name='run/checkpoints/'+filename;payload=io.BytesIO(checkpoints.read(name))
            checksum=read('run/checkpoints/'+Path(filename).with_suffix('.sha256.json').name)
            assert checksum['sha256']==manifest[name]['sha256'] and checksum['bytes']==manifest[name]['bytes']
            unsafe=set(torch.serialization.get_unsafe_globals_in_checkpoint(payload))
            assert unsafe<={'numpy.ndarray','numpy.dtype','numpy._core.multiarray._reconstruct'}
            payload.seek(0)
            with torch.serialization.safe_globals([np._core.multiarray._reconstruct,np.ndarray,np.dtype,type(np.dtype('uint32'))]):
                state=torch.load(payload,map_location='cpu',weights_only=True)
            t=state['step'];assert t==(49999 if filename=='latest.pt' else int(filename[9:14]))
            assert state['identity']==result['identity'] and state['config']==config and state['device']=='cuda:0'
            assert state['sampler']['count']==476000 and state['sampler']['batch_size']==32
            for k,v in sampler_states[t].items():
                if torch.is_tensor(v):assert torch.equal(state['sampler'][k],v)
                else:assert state['sampler'][k]==v
            assert object_sha256(state['rng']['clean'])==object_sha256(state['rng']['peak_first'])
            for arm in arms:
                for v in state['models'][arm].values():assert torch.isfinite(v).all()
                optimizer=state['optimizers'][arm]
                assert len(optimizer['state'])==(0 if t==0 else 55)
                for group in optimizer['param_groups']:
                    assert group['betas']==(.9,.98) and group['eps']==1e-9 and group['weight_decay']==0
                    assert math.isclose(group['lr'],(1e-4 if t==0 or arm=='peak_first' and t==1 else stable(t)),rel_tol=1e-12,abs_tol=0)
                for value in optimizer['state'].values():
                    assert float(value['step'])==t
                    assert torch.isfinite(value['exp_avg']).all() and torch.isfinite(value['exp_avg_sq']).all()
            if t==0:
                assert object_sha256(state['models']['clean'])==object_sha256(state['models']['peak_first'])
                initial=copy.deepcopy(state['models']['clean'])
            else:
                assert len(state['anchors']['clean'])==7 and state['anchors']['clean']==state['anchors']['peak_first']
                assert state['anchors']['clean']['first_batch']==first_record['batch_sha256']
            if t==1:
                assert object_sha256(state['optimizers']['clean']['state'])==object_sha256(state['optimizers']['peak_first']['state'])
                for arm in arms:
                    names=[k for k in initial if k!='pos_encoding.pe']
                    norm=sum(initial[k].double().square().sum() for k in names).sqrt()
                    displacement=sum((state['models'][arm][k].double()-initial[k].double()).square().sum() for k in names).sqrt()
                    ratio=float(displacement/(norm+1e-12));close(ratio,first_record['arms'][arm]['candidate_and_committed_ratio'],1e-11)
            if t==49999:
                assert state['history']==history and state['online_history']==online
                for arm in arms:assert state['cost'][arm]==result['committed_trajectory_cost'][arm]
                scientific={k:state[k] for k in ('models','optimizers','rng','sampler','anchors','history','online_history','windows')}
                if final_digest is None:final_digest=object_sha256(scientific)
                else:assert object_sha256(scientific)==final_digest
            verified.append(dict(file=filename,step=t,finite_model_and_optimizer=True,adam_counter_matches=True,sampler_matches_trace=True,paired_rng_matches=True))
            del state,payload;gc.collect()
        checkpoints.close();write_json(evidence/'CHECKPOINT_AUDIT.json',verified)
        report['full_paired_checkpoint_files_checked']=5
        report['first_step_moments_equal_weights_differ']=True
        report['final_and_latest_scientific_states_match']=True
    if predictions and states:
        report['scope']='full returned records, predictions and restricted tensor-state audit'
        write_json(evidence/'FULL_AUDIT.json',report)
    records.close();print(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions',type=Path);parser.add_argument('--states',type=Path)
    args=parser.parse_args();audit(args.predictions,args.states)
