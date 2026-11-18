"""Run synthetic CPU tests and save a reproducible verification record."""
from pathlib import Path
import contextlib
import io
import json
import os
import platform
import time
import unittest

os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import torch
import numpy as np
import peak_pair

ROOT=Path(__file__).resolve().parent


def main():
    torch.set_num_threads(1)
    start=time.perf_counter()
    stream=io.StringIO()
    with contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
        suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    report={'scope':'CPU synthetic fixtures only. No real dataset downloaded and no research experiment run.',
            'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
            'passed':result.wasSuccessful(),'python':platform.python_version(),
            'torch':torch.__version__,'numpy':np.__version__,'torch_threads':torch.get_num_threads(),
            'source_hashes':peak_pair.source_hashes(),'elapsed_seconds':time.perf_counter()-start,
            'command':'python -B check_cpu.py'}
    evidence=ROOT/'evidence';evidence.mkdir(exist_ok=True)
    (evidence/'cpu_test_log.txt').write_text(stream.getvalue(),encoding='utf-8')
    (evidence/'cpu_test_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(stream.getvalue())
    print(json.dumps({k:v for k,v in report.items() if k!='source_hashes'},indent=2))
    raise SystemExit(not result.wasSuccessful())


if __name__=='__main__':main()
