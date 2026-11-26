#!/usr/bin/env python3
"""Wait for the active GPU worker naturally, then activate the tested fix."""
import json,os,signal,subprocess,sys,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import research_budget18_queue_v1 as q
OUT=ROOT/'research_results/capacity_eval_v5/handoff'

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 info=q.supervisor_info('research-additive-v1');parent=q.proc(info['pid'])
 state=q.read(ROOT/'runs/research/additive_visual_v1/queue_state.json');job=state['active_job'];row=state['jobs'][job];worker=q.proc(row['pid'])
 if not parent or 'scripts/research_capacity_queue_v4.py' not in parent['argv'] or not worker or worker['ppid']!=parent['pid'] or parent['state']=='T':raise ValueError('Exact owned V4 CPU owner and active GPU worker required')
 lease=dict(active=True,controller=parent,coordinator=q.proc(os.getpid()),worker=worker,previous_job=job,GPU_worker_signals_sent=0)
 path=OUT/'lease.json';q.atomic(path,lease);paused=False
 guard=subprocess.Popen([sys.executable,'scripts/run_qwen25_native_baseline_parallel_v1.py','--guardian',str(path)],cwd=ROOT,env=os.environ,start_new_session=True)
 def stop(_n,_f):raise InterruptedError('Restore CPU owner')
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 try:
  os.kill(parent['pid'],signal.SIGSTOP);paused=True
  deadline=time.monotonic()+3600
  while q.same(worker,q.proc(worker['pid'])) or not (OUT/'activate.json').exists():
   if time.monotonic()>deadline:raise TimeoutError('Fix preparation/natural boundary exceeded one hour')
   time.sleep(2)
  request=q.read(OUT/'activate.json')
  subprocess.run([sys.executable,'scripts/research_capacity_queue_v5.py','--plan-sha',request['plan_sha256']],cwd=ROOT,check=True,env=os.environ)
  os.kill(parent['pid'],signal.SIGTERM);os.kill(parent['pid'],signal.SIGCONT);paused=False
  deadline=time.monotonic()+900
  while q.same(parent,q.proc(parent['pid'])):
   if time.monotonic()>deadline:raise TimeoutError('Old CPU owner still finishing its job')
   time.sleep(1)
  q.atomic(OUT/'completed.json',dict(status='natural_worker_boundary',previous_job=job,worker=worker,GPU_worker_signals_sent=0,plan_sha256=request['plan_sha256']))
  conf=Path('/etc/supervisor/conf.d/research-additive-v1.conf')
  lines=[('command=/venv/main/bin/python -u scripts/research_capacity_queue_v5.py --run --plan-sha '+request['plan_sha256'] if l.startswith('command=') else 'stdout_logfile=/workspace/gigachat_vl/logs/research/capacity_eval_queue_v5.log' if l.startswith('stdout_logfile=') else l) for l in conf.read_text().splitlines()]
  conf.write_text('\n'.join(lines)+'\n');subprocess.run(['supervisorctl','reread'],check=True);subprocess.run(['supervisorctl','update','research-additive-v1'],check=True)
 finally:
  if paused and q.same(parent,q.proc(parent['pid'])):os.kill(parent['pid'],signal.SIGCONT)
  q.atomic(path,dict(lease,active=False,released_unix=time.time()))
  try:guard.wait(timeout=5)
  except subprocess.TimeoutExpired:guard.terminate();guard.wait(timeout=5)

if __name__=='__main__':main()
