"""Real tiny native CLI/token/scorer parity; no CUDA or trained weight reads."""
import copy,hashlib,json,random,sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from PIL import Image
from src.eval import research_archived_native_evaluation_v1 as s

@pytest.fixture
def current(tmp_path,monkeypatch):
 import torch
 from tests.test_research_inference import TinyResearchModel
 from src.eval import research_inference as native
 from scripts import eval_research_checkpoint as original
 run=tmp_path/'run';target=run/'.checkpoint_objects/object-owned';target.mkdir(parents=True)
 (target/'state.pt').write_bytes(b'synthetic owned checkpoint, not a real model')
 marker=target/'.research_checkpoint.json';marker.write_text('{}');receipt=run/'checkpoint_archival.json';receipt.write_text('{"synthetic":true}')
 for role in ('best','last'):(run/role).symlink_to(Path('.checkpoint_objects')/target.name)
 context={'run':str(run),'role':'best','checkpoint_path':str(target/'state.pt'),
  'artifact':{'sha256':native.sha256_file(target/'state.pt')},'files':{'checkpoint_archival.json':{'path':str(receipt),'stat':s.fileio.stat(receipt),'sha256':s.fileio.file_sha(receipt)}},
  'roles':{role:{'object':str(target),'marker_file':{'path':str(marker),'stat':s.fileio.stat(marker),'sha256':s.fileio.file_sha(marker)}} for role in ('best','last')},
  'member_stats':{str(target/'state.pt'):s.fileio.stat(target/'state.pt')}}
 groups=['general_vqa','ocr_documents','charts_diagrams','knowledge_reasoning'];records=[]
 for i in range(8):
  path=tmp_path/f'image{i}.png';rgb=Image.new('RGB',(3+i%2,2),(10+i,20,30));rgb.save(path)
  sha=hashlib.sha256(f'RGB:{rgb.width}:{rgb.height}:'.encode()+rgb.tobytes()).hexdigest()
  records.append({'record_id':str(i),'question':'question red' if i%2 else 'question blue','image_paths':[str(path)],'image_hashes':[sha],
   'answers':['blue'],'scorer':'exact_match','task_group':groups[i%4],'benchmark':'synthetic_tiny','source':{'repo':'fixture'}})
 manifest=tmp_path/'manifest.jsonl';manifest.write_text(''.join(json.dumps(r)+'\n' for r in records))
 counts={'loads':0,'inference':0};models=[]
 def load(*a,**k):
  counts['loads']+=1;torch.manual_seed(35);model=native.freeze_for_generation(TinyResearchModel(wrapped=True));models.append(model)
  return model,{'numerical_profile':s.stock.definition(),'archived_evaluation':s.shared.execution_profile(),'archived_physical_sha256':context['artifact']['sha256'],
   'archived_publication_sha256':context['files']['checkpoint_archival.json']['sha256']}
 def inference():
  counts['inference']+=1;return SimpleNamespace(GENERATION_SCHEMA_VERSION=native.GENERATION_SCHEMA_VERSION,generate_prediction=native.generate_prediction,
   load_research_checkpoint=load,sha256_file=native.sha256_file)
 monkeypatch.setattr(s.shared,'inspect',lambda *a,**k:copy.deepcopy(context));monkeypatch.setattr(s.shared,'inference',inference)
 monkeypatch.setattr(original,'assert_gpu_idle',lambda *a:None)
 flags=(torch.are_deterministic_algorithms_enabled(),torch.is_deterministic_algorithms_warn_only_enabled(),torch.backends.cuda.matmul.allow_tf32,torch.get_num_threads())
 s.stock.activate()
 yield SimpleNamespace(run=run,context=context,records=records,manifest=manifest,original=original,load=load,counts=counts,models=models,root=tmp_path,native=native)
 torch.use_deterministic_algorithms(flags[0],warn_only=flags[1]);torch.backends.cuda.matmul.allow_tf32=flags[2];torch.set_num_threads(flags[3])

@pytest.mark.parametrize('variant',['none','shuffle','blank','text_only'])
def test_frozen_native_cli_prompt_tokens_scores_RNG_and_all4_groups_exact(current,monkeypatch,variant):
 import torch
 oldout=current.root/f'ordinary-{variant}';newout=s.output_path(current.run,variant,'best')
 monkeypatch.setattr(current.original,'load_research_checkpoint',current.load)
 current.original.main(['--run',str(current.run),'--manifest',str(current.manifest),'--output',str(oldout),'--max-new-tokens','2','--image-ablation',variant])
 expected=copy.deepcopy(current.models[-1].state_dict());rng=torch.get_rng_state().clone();python_rng=random.getstate()
 module=s.evaluator(current.context);module.assert_gpu_idle=lambda *a:None
 module.main(['--run',str(current.run),'--manifest',str(current.manifest),'--output',str(newout),'--max-new-tokens','2','--image-ablation',variant])
 a=[json.loads(x) for x in (oldout/'predictions.jsonl').read_text().splitlines()];b=[json.loads(x) for x in (newout/'predictions.jsonl').read_text().splitlines()]
 for old,new in zip(a,b):
  for key in ('record_id','prediction','generated_token_ids','generated_tokens','prompt_sha256','prefill_tokens','visual_tokens','logits_to_keep','score','answers','scorer','task_group'):
   assert old[key]==new[key]
 protocol=json.loads((newout/'protocol.json').read_text());assert protocol['single_decode_execution']['CPU_payload_decodes_per_fresh_load']==1
 assert protocol['numerical_profile']==s.stock.definition() and protocol['source_sha256'][s.OWN[0]]==s.sha(s.ROOT/s.OWN[0])
 summary=json.loads((newout/'summary.json').read_text());oldsummary=json.loads((oldout/'summary.json').read_text())
 assert summary['group_macro_score']==oldsummary['group_macro_score'] and summary['complete_four_group_coverage']
 assert current.counts=={'loads':2,'inference':1} and torch.equal(rng,torch.get_rng_state()) and python_rng==random.getstate()
 for name,value in current.models[-1].state_dict().items():torch.testing.assert_close(value,expected[name],atol=0,rtol=0)
 assert not torch.cuda.is_initialized()

def test_no_stock_eval_or_double_peek_called_single_private_native_loader(current,monkeypatch):
 monkeypatch.setattr(s.stock,'eval_main',Mock(side_effect=AssertionError('Old eval prepeek used')))
 monkeypatch.setattr(s.stock,'inference',Mock(side_effect=AssertionError('Old double-peek inference used')))
 module=s.evaluator(current.context);module.assert_gpu_idle=lambda *a:None;output=s.output_path(current.run,'none','best')
 module.main(['--run',str(current.run),'--manifest',str(current.manifest),'--output',str(output),'--max-new-tokens','2','--max-records','2'])
 assert current.counts=={'loads':1,'inference':1}
 result=json.loads((output/'partial_summary.json').read_text());assert result['group_macro_score'] is None

def test_resume_completed_own_protocol_never_reload_or_decode(current):
 module=s.evaluator(current.context);module.assert_gpu_idle=lambda *a:None;out=s.output_path(current.run,'none','best')
 argv=['--run',str(current.run),'--manifest',str(current.manifest),'--output',str(out),'--max-new-tokens','2']
 module.main(argv);before=(out/'predictions.jsonl').read_bytes();module.main(argv+['--resume'])
 assert current.counts['loads']==1 and (out/'predictions.jsonl').read_bytes()==before
 value=json.loads((out/'protocol.json').read_text());value['single_decode_execution']['owner']='foreign';(out/'protocol.json').write_text(json.dumps(value))
 with pytest.raises(ValueError,match='Resume protocol'):module.main(argv+['--resume'])
 assert current.counts['loads']==1

def test_wrong_checkpoint_or_archive_provenance_refused_before_generation(current,monkeypatch):
 module=s.evaluator(current.context)
 with pytest.raises(ValueError,match='before loading'):module.load_research_checkpoint(current.root/'foreign.pt')
 assert current.counts['loads']==0
 changed=copy.deepcopy(current.context);changed['artifact']['sha256']='0'*64;other=s.evaluator(changed)
 with pytest.raises(ValueError,match='provenance'):other.load_research_checkpoint(current.context['checkpoint_path'])

def test_gold_annotations_never_accessed_by_private_native_generation(current):
 class Guard(dict):
  def __getitem__(self,k):
   if k in ('answers','answer','scorer'):raise AssertionError('Gold accessed')
   return super().__getitem__(k)
  def get(self,k,*a):
   if k in ('answers','answer','scorer'):raise AssertionError('Gold accessed')
   return super().get(k,*a)
 module=s.evaluator(current.context);model,_=module.load_research_checkpoint(current.context['checkpoint_path'])
 actual=module.generate_prediction(model,Guard(current.records[0]),max_new_tokens=2)
 assert actual['generated_tokens']==2

def test_new_namespace_and_fresh_process_cli_guards(tmp_path,monkeypatch):
 with pytest.raises(ValueError,match='namespace'):s.output_path(tmp_path,'none','best',tmp_path/'evaluation/heldout_v3')
 from scripts import eval_research_archived_native_v1 as cli
 monkeypatch.setitem(sys.modules,'torch',object())
 with pytest.raises(RuntimeError,match='fresh process'):cli.main(['--allow-gpu','--run','x','--manifest','x'])
