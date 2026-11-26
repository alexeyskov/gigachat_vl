#!/usr/bin/env python3
"""Secondary stock-deterministic3b numerical diagnostic; never legacy validation."""
from __future__ import annotations
import hashlib,json,os
from pathlib import Path
import sys
from types import ModuleType
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OWNER='smallthinker3b_stock_deterministic_diagnostic_v2'
READY=ROOT/'research_results/selection_deterministic_3b_ready.json'
NATIVE_SOURCE='scripts/probe_research_selection_replay.py'
NATIVE_SHA='fd10b96c3a9346a4d68d8d52988f5ab2a61228a3387d296d369100ec81dc980e'
NATIVE_READY_SHA='8493a9139faa64bac6ff37da6c1d93727634e497a470e2fab30566ae356c777e'
OUTPUT=ROOT/'research_results/selection_deterministic_3b_best170_v2'
PROFILE={'name':'secondary_stock_torch_deterministic_v2','CUBLAS_WORKSPACE_CONFIG':':4096:8',
 'torch_use_deterministic_algorithms':True,'warn_only':False,'matmul_allow_tf32':True,'torch_threads':8,
 'full116_generation_passes':3,'repeated_unique_records':116,'restoration_polling':'fast_check_per_tensor_full_resource_thread_0.5s',
 'legacy_quality_validation':False,'scientific_report_replacement':False,'automatic_fallback':False}

def clone_native():
 raw=(ROOT/NATIVE_SOURCE).read_bytes()
 if hashlib.sha256(raw).hexdigest()!=NATIVE_SHA:raise ValueError('Frozen native diagnostic changed')
 # A new isolated module: canonical native globals/source/READY stay unchanged.
 d=ModuleType('scripts._stock_deterministic_selection_v2');d.__file__=str(Path(__file__).resolve())
 d.__package__='scripts';exec(compile(raw,str(ROOT/NATIVE_SOURCE),'exec'),d.__dict__)
 original_verify=d.verify_release
 # The frozen release is checked with its original namespace before rebinding.
 if d.sha(ROOT/'research_results/selection_replay_3b_ready.json')!=NATIVE_READY_SHA:raise ValueError('Frozen native READY changed')
 original_verify()
 d.OWNER=OWNER;d.READY=READY;d.OUTPUT=OUTPUT
 d.SOURCES=(*d.SOURCES,'scripts/probe_research_selection_deterministic.py','tests/test_research_selection_deterministic.py')
 def verify():
  r=d.read(READY)
  if (r.get('owner')!=OWNER or r.get('status')!='cpu_tests_passed_actual_gpu_not_run' or
      r.get('source_sha256')!=d.source_binding() or r.get('profile')!=PROFILE or
      r.get('native_source_sha256')!=NATIVE_SHA or r.get('native_ready_sha256')!=NATIVE_READY_SHA):
   raise ValueError('Exact secondary deterministic diagnostic release required')
  return {'path':str(READY),'sha256':d.sha(READY)}
 d.verify_release=verify
 native_prepare=d.prepare_plan
 def prepare(*a,**k):
  p=native_prepare(*a,**k);p['numerical_profile']=dict(PROFILE);p['expected_internal_profile'].update(deterministic_algorithms=True,CUBLAS_WORKSPACE_CONFIG=':4096:8')
  return p
 d.prepare_plan=prepare
 native_runtime=d.NativeRuntime
 class StockRuntime(native_runtime):
  def load(self,plan,check):
   if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise RuntimeError('Stock deterministic workspace must precede Torch import')
   import torch
   torch.use_deterministic_algorithms(True,warn_only=False)
   owner=getattr(check,'__self__',None)
   if owner is None or not callable(getattr(owner,'fast_check',None)):raise ValueError('Reviewed live guard fast_check required')
   return super().load(plan,owner.fast_check)
  def repeat(self,model,selection,mismatches,check):
   owner=getattr(check,'__self__',None)
   if owner is not None:owner.phase='full116_repeat_'+str(getattr(self,'repeat_number',1))
   return super().repeat(model,selection,[{'record_id':r['record_id']} for r in selection['records']],check)
  def observed_generate(self,model,record,predict,*,phase,**kwargs):
   phase=phase.replace('mismatch_repeat_','full116_repeat_')
   return super().observed_generate(model,record,predict,phase=phase,**kwargs)
 d.NativeRuntime=StockRuntime
 native_environment=d.runtime_environment
 def environment():
  result=native_environment();result.update(numerical_profile=dict(PROFILE),CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))
  return result
 d.runtime_environment=environment
 native_write=d.write
 def write(path,value):
  path=Path(path)
  if path.name.startswith('mismatch_repeat_'):path=path.with_name(path.name.replace('mismatch_repeat_','full116_repeat_',1))
  if isinstance(value,dict) and value.get('owner')==OWNER and path.name=='report.json':
   value.update(numerical_profile=dict(PROFILE),secondary_numerical_diagnostic=True,legacy_validation_passed=False)
   if 'repeated_mismatches' in value:value['repeated_all116']=value.pop('repeated_mismatches')
   if value.get('status')=='completed_diagnostic_only':
    full=d.read(path.parent/'replayed_selection.json')['predictions'];repeats=value['repeated_all116']
    if len(repeats)!=2 or any(len(rows)!=116 for rows in repeats):raise ValueError('Both full116 repeated passes are mandatory')
    ids=[r['record_id'] for r in full]
    if any([r['record_id'] for r in rows]!=ids for rows in repeats):raise ValueError('Repeated full116 record order differs')
    value['three_pass_token_mismatch_records']=sum(any(full[i]['generation']['generated_token_ids']!=rows[i]['generation']['generated_token_ids'] for rows in repeats) for i in range(116))
    value['three_pass_prompt_mismatch_records']=sum(any(full[i]['generation']['prompt_sha256']!=rows[i]['generation']['prompt_sha256'] for rows in repeats) for i in range(116))
  return native_write(path,value)
 d.write=write
 return d

def main(argv=None):
 argv=list(sys.argv[1:] if argv is None else argv)
 if '--allow-gpu' in argv and 'torch' in sys.modules:raise RuntimeError('Fresh process must set deterministic workspace before Torch import')
 os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
 if '--mismatch-repeats' in argv:raise ValueError('Secondary profile requires fixed full116 repeats, no mismatch-only option')
 if '--full116-repeats' in argv:
  index=argv.index('--full116-repeats')
  if index+1>=len(argv) or argv[index+1]!='2':raise ValueError('Exactly two full116 repeats are required')
  del argv[index:index+2]
 argv+=['--mismatch-repeats','2']
 return clone_native().main(argv)

if __name__=='__main__':raise SystemExit(main())
