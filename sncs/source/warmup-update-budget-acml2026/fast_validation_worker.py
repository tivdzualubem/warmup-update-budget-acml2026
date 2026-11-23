"""Run one UNMODIFIED frozen experiment runner on runtime-only small REAL datasets.
Patch only the data-loading boundary in this process; publication source never changes.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path('validation_runtime_data')
OUT = Path('validation_runtime_results')

def run(args):
    import torch
    from datasets import load_from_disk
    from publication_expansion.src import pretrained_job_runner as pre
    from publication_expansion.src import scratch_job_runner as scr
    from publication_expansion.src.pretrained_model_adapter import (
        build_sequence_classifier, build_token_classifier, build_image_classifier,
        load_deit_processor,
    )
    from publication_expansion.src.pretrained_data_adapter import CifarImageDataset
    from publication_expansion.src.scratch_dataset_adapter import (
        ScratchTextDataset, PreparedScratchDataset, build_vocab,
    )
    # CPU threading small; avoid oversubscription on two Kaggle T4 processes.
    torch.set_num_threads(2)
    data_dir=ROOT/args.family/args.dataset
    if args.family=='scratch':
        def prepare(dataset,* ,max_length,split_seed,val_ratio,train_fraction,revision):
            del max_length,split_seed,val_ratio,train_fraction,revision
            raw={k: json.loads((data_dir/(k+'.json')).read_text()) for k in ('train','validation','test')}
            vocab=build_vocab(raw['train']['texts'], min_freq=2,max_vocab_size=30000)
            ds={k:ScratchTextDataset(v['texts'],v['labels'],vocab,max_length=128) for k,v in raw.items()}
            return PreparedScratchDataset(ds['train'],ds['validation'],ds['test'],vocab,
                4 if dataset=='ag_news' else 14,
                {'validation_only':True,'source':'frozen real HF subset',
                 'subset_counts':{k:len(v['texts']) for k,v in raw.items()},
                 'seed':42,'data_override':'integration test; NOT publication dataset'})
        scr.load_scratch_hf_dataset=prepare
        runner=scr
    else:
        def prepare(dataset,*,train_fraction=1.0):
            del train_fraction
            data={}
            for k in ('train','validation','test'):
                d=load_from_disk(str(data_dir/k))
                if dataset!='cifar10':
                    d.set_format(type='torch',columns=['input_ids','attention_mask','labels'])
                data[k]=d
            if dataset in ('ag_news','dbpedia_14'):
                n=4 if dataset=='ag_news' else 14
                model=build_sequence_classifier(num_labels=n)
                data['num_labels']=n
            elif dataset=='wnut_17':
                names=json.loads((data_dir/'labels.json').read_text())
                id2label={i:v for i,v in enumerate(names)}
                label2id={v:i for i,v in id2label.items()}
                model=build_token_classifier(num_labels=len(names),id2label=id2label,label2id=label2id)
                data.update(num_labels=len(names),id2label=id2label,label2id=label2id)
            elif dataset=='cifar10':
                names=json.loads((data_dir/'labels.json').read_text())
                id2label={i:v for i,v in enumerate(names)}
                label2id={v:i for i,v in id2label.items()}
                processor=load_deit_processor()
                for k in ('train','validation','test'):
                    d=data[k]
                    data[k]=CifarImageDataset(d,range(len(d)),processor=processor)
                model=build_image_classifier(num_labels=len(names),id2label=id2label,label2id=label2id)
                data.update(num_labels=len(names),id2label=id2label,label2id=label2id)
            else:raise RuntimeError('unknown task')
            return data,model
        # Original module's run_job calls prepare_task, which we override only here.
        pre.prepare_task=prepare
        runner=pre

    # Real step-level progress; delegate actual transitions unchanged to frozen runner.
    original_step=runner.transactional_step
    count=[0]
    def logged_step(**kwargs):
        output=original_step(**kwargs)
        count[0]+=1
        if count[0]==1 or count[0]%2==0 or output.stopped:
            print(f'[STEP] {count[0]} mode={kwargs["mode"]} '
                  f'candidate_ratio={output.candidate_update_ratio:.7g} '
                  f'guard={output.guard_triggered} stopped={output.stopped}',flush=True)
        return output
    runner.transactional_step=logged_step
    result_root=OUT/args.family
    result_root.mkdir(parents=True,exist_ok=True)
    controls=argparse.Namespace(dataset=args.dataset,seed=42,condition=args.condition,
         output_root=str(result_root),device='cuda',num_workers=0,pilot_mode=True,
         train_fraction=1.0,epochs=3,max_optimizer_steps=None,force=False,
         threshold=args.threshold,model=args.model)
    print(f'[RUN] family={args.family} data={args.dataset} model={args.model} '
          f'condition={args.condition} real_reduced_subset=YES',flush=True)
    rc=runner.run_job(controls)
    result_dir=result_root/args.dataset
    if args.family=='scratch':result_dir=result_dir/args.model
    result_dir=result_dir/'seed42'/args.condition
    result=json.loads((result_dir/'result.json').read_text())
    # The original manifest's 1.0 means 100% of the externally supplied subset,
    # NOT 100% of publication data. Record override on *every* result explicitly.
    override={'non_scientific':True,'external_subset':True,'subset_manifest':
              'validation_runtime_data/subset_manifest.json',
              'description':'The frozen runner was executed on reduced real HF datasets injected at the data boundary; publication input counts and validation/test are NOT exercised.'}
    (result_dir/'VALIDATION_ONLY_DATA_OVERRIDE.json').write_text(json.dumps(override,indent=2)+'\n')
    result['validation_only_data_override']=override
    (result_dir/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(f'[FINISH] {args.dataset}/{args.model}/{args.condition} '
          f'status={result["terminal_reason"]} steps={result["global_steps"]} '
          f'test={result.get("test_metrics")}',flush=True)
    return rc

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--family',choices=('scratch','pretrained'),required=True)
    p.add_argument('--dataset',required=True)
    p.add_argument('--model',default='distilbert')
    p.add_argument('--condition',required=True)
    p.add_argument('--threshold',type=float,default=None)
    args=p.parse_args()
    sys.exit(run(args))
