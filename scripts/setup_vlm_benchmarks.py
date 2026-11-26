#!/usr/bin/env python3
"""Install the pinned benchmark harness in an isolated system-site-packages venv."""
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--venv',type=Path,default=Path('/tmp/research_benchmark_venv'))
    args=p.parse_args();venv=args.venv.absolute()
    if venv.resolve()==Path(sys.prefix).resolve():raise ValueError('Refusing to install into the active training environment')
    spec=json.loads((ROOT/'configs/eval/vlm_benchmarks_v1.json').read_bytes());harness=ROOT/spec['harness']['path']
    def run(command):return subprocess.run(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True)
    if not harness.exists():
        run(['git','clone','--filter=blob:none','--no-checkout',spec['harness']['repository'],str(harness)])
        run(['git','-C',str(harness),'checkout','--detach',spec['harness']['revision']])
    actual=subprocess.check_output(['git','-C',str(harness),'rev-parse','HEAD'],text=True).strip()
    if actual!=spec['harness']['revision']:raise ValueError('Existing harness is at a different revision; preserve it and use the pinned checkout')
    if not (venv/'pyvenv.cfg').exists():run([sys.executable,'-m','venv','--system-site-packages',str(venv)])
    run(['uv','pip','install','--python',str(venv/'bin/python'),'--no-deps','-r',str(ROOT/'configs/eval/vlmeval_extra_requirements_v1.txt')])
    run([str(venv/'bin/python'),str(ROOT/'scripts/run_vlm_benchmarks.py'),'--check-harness'])


if __name__=='__main__':main()
