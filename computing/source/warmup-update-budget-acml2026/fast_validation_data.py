"""Prepare tiny deterministic *real* HF subsets for runtime-only whole-matrix validation.
Never a scientific sample or a replacement for frozen publication splits.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from datasets import load_dataset, DatasetDict

from publication_expansion.src.pretrained_data_adapter import (
    load_frozen_dataset, tokenize_sequence_batch, tokenize_wnut_batch,
)
from publication_expansion.src.pretrained_model_adapter import load_distilbert_tokenizer
from publication_expansion.src.scratch_data_contract import make_split_indices

C = {'ag_news': 4, 'dbpedia_14':14}
SPECS = {'ag_news':'text','dbpedia_14':'content'}
ROOT = Path('validation_runtime_data')

def choose_balanced(ds, count_per_class, seed=42):
    labels = ds['label']
    rng = np.random.RandomState(seed)
    idx = []
    for c in sorted(set(labels)):
        candidates = np.flatnonzero(np.asarray(labels)==c)
        if len(candidates)<count_per_class:
            raise RuntimeError(f'not enough examples for class {c}')
        idx.extend(rng.choice(candidates, count_per_class, replace=False).tolist())
    rng.shuffle(idx)
    return ds.select(idx)

def save_tokenized(parts, dest, tokenizer, dataset, wnut=False):
    dest.mkdir(parents=True, exist_ok=True)
    cols = ['input_ids','attention_mask','labels']
    for split, raw in parts.items():
        if wnut:
            ds = raw.map(lambda b: tokenize_wnut_batch(b,tokenizer=tokenizer),
                         batched=True,remove_columns=raw.column_names)
        else:
            ds = raw.map(lambda b: tokenize_sequence_batch(b,tokenizer=tokenizer,dataset_name=dataset),
                         batched=True,remove_columns=raw.column_names)
        if not all(x in ds.column_names for x in cols):
            raise RuntimeError('tokenization schema mismatch')
        ds.save_to_disk(str(dest / split))

def build():
    ROOT.mkdir(exist_ok=True)
    tokenizer = load_distilbert_tokenizer()
    meta={'schema_version':1,'intent':'non-scientific, deterministic real-data integration validation',
          'seed':42,'source_splits':'frozen HF revisions; original train split policy', 'datasets':{}}
    for dataset in ('ag_news','dbpedia_14'):
        print(f'[DATA] {dataset}: loading pinned real HF data',flush=True)
        raw=load_frozen_dataset(dataset)
        train=raw['train']
        hf_split=train.train_test_split(test_size=.15,seed=42,shuffle=True)
        nclass=C[dataset]
        ntrain=32 if nclass==4 else 10
        nval=8 if nclass==4 else 4
        parts={'train':choose_balanced(hf_split['train'],ntrain),
               'validation':choose_balanced(hf_split['test'],nval),
               'test':choose_balanced(raw['test'],nval)}
        save_tokenized(parts,ROOT/'pretrained'/dataset,tokenizer,dataset)
        # Scratch uses its own original split contract (not pretrained split).
        split=make_split_indices(len(train),split_seed=42,val_ratio=.15,train_fraction=1.0)
        scratch_parts={
            'train':choose_balanced(train.select(list(split.train)),16 if nclass==4 else 6),
            'validation':choose_balanced(train.select(list(split.validation)),8 if nclass==4 else 3),
            'test':choose_balanced(raw['test'],8 if nclass==4 else 3)}
        dest=ROOT/'scratch'/dataset
        dest.mkdir(parents=True,exist_ok=True)
        for label,ds in scratch_parts.items():
            (dest/f'{label}.json').write_text(json.dumps({
                'texts':[str(row[SPECS[dataset]]) for row in ds], 'labels':[int(row['label']) for row in ds]},ensure_ascii=False))
        meta['datasets'][dataset]={
            'pretrained_counts':{k:len(v) for k,v in parts.items()},
            'scratch_counts':{k:len(v) for k,v in scratch_parts.items()},
            'class_balanced':True}
        print(f'[DATA] {dataset}: pretrained={meta["datasets"][dataset]["pretrained_counts"]}; '
              f'scratch={meta["datasets"][dataset]["scratch_counts"]}',flush=True)
    print('[DATA] wnut_17: loading pinned real HF data',flush=True)
    raw=load_frozen_dataset('wnut_17')
    # Select 64/24/24 actual official examples; preserve first-subtoken labeling.
    rng=np.random.RandomState(42)
    parts={split: raw[split].select(rng.choice(len(raw[split]),size=n,replace=False).tolist())
           for split,n in [('train',64),('validation',24),('test',24)]}
    save_tokenized(parts,ROOT/'pretrained'/'wnut_17',tokenizer,'wnut_17',wnut=True)
    names=list(raw['train'].features['ner_tags'].feature.names)
    dest=ROOT/'pretrained'/'wnut_17'
    (dest/'labels.json').write_text(json.dumps(names))
    meta['datasets']['wnut_17']={'pretrained_counts':{k:len(v) for k,v in parts.items()}}
    print('[DATA] cifar10: loading pinned real HF data',flush=True)
    raw=load_frozen_dataset('cifar10')
    # Frozen publication stratified split (5000 validation), THEN reduce to 13/class.
    from publication_expansion.src.pretrained_data_adapter import make_cifar_stratified_split
    split=make_cifar_stratified_split(raw['train']['label'],validation_size=5000,split_seed=42)
    parts={'train':choose_balanced(raw['train'].select(split.train_indices),13),
           'validation':choose_balanced(raw['train'].select(split.validation_indices),4),
           'test':choose_balanced(raw['test'],4)}
    dest=ROOT/'pretrained'/'cifar10'
    dest.mkdir(parents=True,exist_ok=True)
    for name,ds in parts.items():
        ds.save_to_disk(str(dest/name))
    names=list(raw['train'].features['label'].names)
    (dest/'labels.json').write_text(json.dumps(names))
    meta['datasets']['cifar10']={'pretrained_counts':{k:len(v) for k,v in parts.items()},'class_balanced':True}
    (ROOT/'subset_manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    (ROOT/'READY').write_text('ok\n')
    print('[DATA] ALL 4 REAL TASKS AND BOTH SCRATCH DATASETS: READY',flush=True)

if __name__=='__main__':
    if not (ROOT/'READY').exists():build()
    else: print('[DATA] existing deterministic subsets reused',flush=True)
