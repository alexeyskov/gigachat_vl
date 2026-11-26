import io
from types import ModuleType
import numpy as np
import pytest
import torch
from src.utils.research_budget_storage_v1 import METADATA_ALLOWANCE,payload_storage_budget,install_manager

def test_all_model_optimizer_rng_backings_and_aliases_charged_exactly_once():
 base=torch.arange(100,dtype=torch.float32);mom=torch.arange(100,dtype=torch.uint8);rng=np.arange(624,dtype=np.uint32)
 payload={'model':{'weight':base,'alias':base,'slice':base[:5]},'optimizer':{'state1':mom,'state2':mom.clone()},'rng_numpy':('MT19937',rng,2,0,.1),'rng_torch':torch.arange(8,dtype=torch.uint8),'metadata':{'dtype':'float32','step':56}}
 report=payload_storage_budget(payload)
 assert report['unique_tensor_storage_bytes']==400+100+100+8
 assert report['anticipated_bytes']==608+rng.nbytes+METADATA_ALLOWANCE
 buffer=io.BytesIO();torch.save(payload,buffer);assert len(buffer.getvalue())<report['anticipated_bytes']
 buffer.seek(0);restored=torch.load(buffer,weights_only=False)
 assert torch.equal(restored['model']['weight'],base) and restored['model']['weight'].untyped_storage()._cdata==restored['model']['slice'].untyped_storage()._cdata

@pytest.mark.parametrize('bad',[torch.empty(2,device='meta'),torch.sparse_coo_tensor(torch.tensor([[0]]),torch.tensor([1.]),(2,)),object()])
def test_unknown_or_unmaterialized_payload_refused(bad):
 with pytest.raises(ValueError):payload_storage_budget({'model':bad})

def test_private_manager_payload_and_optimizer_are_not_filtered(tmp_path):
 module=ModuleType('_private_budget_fixture');seen={}
 class Native:
  def __init__(self):self.root=tmp_path
  def save_torch(self,payload,**kwargs):seen.update(payload=payload,kwargs=kwargs);return {'saved':True}
 module.BestLastCheckpointManager=Native;model=torch.arange(3.);payload={'model':model,'optimizer':{'moment':torch.ones(3,dtype=torch.uint8)},'rng':torch.arange(4,dtype=torch.uint8)};install_manager(module)
 module.BestLastCheckpointManager().save_torch(payload,anticipated_bytes=9999999,step=56,metric=1.)
 assert seen['payload'] is payload and 'optimizer' in seen['payload'] and seen['kwargs']['anticipated_bytes']==payload_storage_budget(payload)['anticipated_bytes']
 assert (tmp_path/'checkpoint_storage_budget.json').is_file()
