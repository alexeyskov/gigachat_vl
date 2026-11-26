import io,json,random
from types import ModuleType
import numpy as np
import pytest
import torch
from src.utils.research_storage import BestLastCheckpointManager,DiskBudget
from src.utils.research_budget_storage_v1 import install_manager as storage_manager
from src.utils.research_budget_stream_save_v1 import install_manager,_Writer


def test_actual_native_save_load_full_payload_optimizer_RNG_alias_and_bestlast(tmp_path):
 module=ModuleType('_private_stream_fixture');module.BestLastCheckpointManager=BestLastCheckpointManager;calls=[]
 def guard(extra):calls.append(extra);return {'available_bytes':100*2**30}
 install_manager(module,guard=guard);storage_manager(module)
 base=torch.arange(10000,dtype=torch.float32);payload={'model':{'base':base,'view':base[100:200],'tie':base},'optimizer':{'mom':torch.arange(10000,dtype=torch.uint8)},'rng_numpy':np.random.get_state(),'rng_python':random.getstate(),'rng_torch':torch.get_rng_state(),'config':{'step':56},'step':56}
 manager=module.BestLastCheckpointManager(tmp_path,disk_budget=DiskBudget(tmp_path,0));manager.save_torch(payload,step=56,metric=1.)
 assert (tmp_path/'best').resolve()==(tmp_path/'last').resolve()
 restored=torch.load(tmp_path/'last/state.pt',map_location='cpu',mmap=True,weights_only=False)
 assert torch.equal(restored['model']['base'],base) and restored['model']['base'].untyped_storage()._cdata==restored['model']['view'].untyped_storage()._cdata
 assert torch.equal(restored['optimizer']['mom'],payload['optimizer']['mom']) and torch.equal(restored['rng_torch'],payload['rng_torch']) and np.array_equal(restored['rng_numpy'][1],payload['rng_numpy'][1]) and restored['rng_python']==payload['rng_python']
 receipt=json.loads((tmp_path/'checkpoint_stream_write.json').read_bytes());assert receipt['only_new_staging_output_advised'] and receipt['assumed_cache_credit_bytes']==0 and calls


def test_small_windows_output_exact_bytes_and_each_guard_after_fsync(tmp_path):
 path=tmp_path/'own';events=[]
 class Budget:
  def ensure_space(self,n):events.append('disk')
 def guard(n):events.append('host');return {'available_bytes':99}
 data=b'0123456789'*10
 with path.open('xb') as handle:
  sink=_Writer(handle,guard=guard,budget=Budget(),chunk=17);assert sink.write(data)==len(data);sink.commit()
 assert path.read_bytes()==data and sink.commits==6 and events==['disk','host']*6


def test_memory_refusal_does_not_publish_partial_or_destroy_previous_last(tmp_path):
 module=ModuleType('_private_stream_failure');module.BestLastCheckpointManager=BestLastCheckpointManager;install_manager(module,guard=lambda extra:(_ for _ in ()).throw(MemoryError('floor retained')));storage_manager(module)
 manager=module.BestLastCheckpointManager(tmp_path,disk_budget=DiskBudget(tmp_path,0))
 with pytest.raises(MemoryError):manager.save_torch({'model':torch.ones(3),'optimizer':{}},step=1,metric=1.)
 assert not (tmp_path/'last').exists() and not list((tmp_path/'.checkpoint_objects').glob('.stage-*'))
