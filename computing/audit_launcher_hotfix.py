"""Verify the supplied patch by applying its pure text function in memory only.

Never calls main(), launches training, or mutates the frozen original launcher.
"""
from pathlib import Path
import ast,hashlib,json,zipfile,difflib
ROOT=Path(__file__).resolve().parent
E=ROOT/'evidence/execution_provenance'

def digest(b):return hashlib.sha256(b).hexdigest()

def run():
    hotfix=E/'ACML_LAUNCHER_HOTFIX_RESUME.zip'
    assert digest(hotfix.read_bytes())=='d05ace110f0d702aee6842a81d215a9aa1ffbc9eb279079178a6b46243479a7d'
    with zipfile.ZipFile(hotfix) as z,zipfile.ZipFile(E/'original_source_launcher.zip') as original:
        assert z.testzip() is None
        payload=z.read('apply_acml_fullrun_hotfix.py').decode('utf-8')
        tree=ast.parse(payload)
        # Execute only the already-inspected pure string transformer; not module imports or main.
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='patch_script')
        scope={}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'<verified_patch_script>','exec'),scope)
        patch=scope['patch_script']
        out=E/'reconstructed_launchers';out.mkdir(exist_ok=True)
        rows=[]
        for folder,file in {'AG_News':'run_full_agnews.py','DBPedia':'run_full_dbpedia.py','WNUT_17':'run_full_wnut17.py','CIFAR_10':'run_full_cifar10.py'}.items():
            member=f'ACML_ALL_DATASETS_ONE_COMMAND/datasets/{folder}/{file}'
            old=original.read(member).decode('utf-8');new=patch(old)
            assert patch(new)==new
            assert 'scratch.DATASETS = (DATASET,)' in new and 'pre.DATASETS = (DATASET,)' in new
            assert 'job_queue.put(job[0] if isinstance(job, tuple) else job)' in new
            compile(new,file,'exec') # syntax only
            (out/file).write_text(new,encoding='utf-8',newline='\n')
            diff=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile=file+' original',tofile=file+' reconstructed'))
            (out/(file+'.diff')).write_text(diff,encoding='utf-8')
            rows.append({'dataset':folder,'original_member':member,'original_sha256':digest(old.encode()),'reconstructed_sha256':digest(new.encode()),'idempotent':True,'syntax_valid':True,'dataset_scope_before_enumeration':True,'queue_unwrap_present':True})
        record={'supplied_zip_sha256':digest(hotfix.read_bytes()),'patch_script_sha256':digest(payload.encode()),'status':'passed','wrappers':rows,'scope':'Pure patch function applied to original launcher text in memory; syntax and idempotence checks only. No launcher, original hotfix main, or training executed.','source_modules_changed':False,'observed_log_failures_addressed':['Two tuple-versus-dictionary queue failures','Cross-dataset DBpedia threshold lookup from AG News launcher'],'not_proven':'No final original-host patch application log or execution-time source digest was supplied; reconstructed wrappers are not recovered executed files.'}
        (E/'launcher_hotfix_audit.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
        print(json.dumps(record,indent=2))

if __name__=='__main__':run()
