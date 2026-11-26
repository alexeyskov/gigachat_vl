"""Native loading-info adapter for installed packed MTP conversion; no math change."""
from __future__ import annotations
import ast,hashlib,json,types
from importlib.metadata import distribution
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
BASE='src/model/research_giga_loading_v1.py';BASE_SHA='c354198662aab43b64bf494cc98390cd973912b28c52c041ed9321cf47be6d02'
CONVERTED_MTP={'model.layers.26.mlp.experts.gate_up_proj','model.layers.26.mlp.experts.down_proj'}
CONVERSION_SHA='126a342d8be1942f76207ed30f809b659709c71618ab2b57066bfde19a0a1f83'
_PRIVATE=None
def normalize_loading_info(value):
 if isinstance(value,dict):return {k:normalize_loading_info(v) for k,v in value.items()}
 if isinstance(value,set):return [normalize_loading_info(v) for v in sorted(value,key=repr)]
 if isinstance(value,(tuple,list)):return [normalize_loading_info(v) for v in value]
 if value is None or isinstance(value,(str,int,float,bool)):return value
 raise TypeError('Unsupported native loading-info type: '+type(value).__name__)
def implementation():
 global _PRIVATE
 if hashlib.sha256((ROOT/BASE).read_bytes()).hexdigest()!=BASE_SHA:raise ValueError('Pinned Giga loadingV1 changed')
 conversion=distribution('transformers').locate_file('transformers/conversion_mapping.py')
 if hashlib.sha256(Path(conversion).read_bytes()).hexdigest()!=CONVERSION_SHA:raise ValueError('Installed native packed-expert conversion mapping changed')
 if _PRIVATE is not None:return _PRIVATE
 tree=ast.parse((ROOT/BASE).read_bytes());counts={'info':0,'allow':0}
 class Extend(ast.NodeTransformer):
  def visit_Assign(self,node):
   if len(node.targets)==1 and isinstance(node.targets[0],ast.Tuple) and [n.id for n in node.targets[0].elts if isinstance(n,ast.Name)]==['model','info']:
    counts['info']+=1;return [node,ast.Assign([ast.Name('info',ast.Store())],ast.Call(ast.Name('_normalize_loading_info',ast.Load()),[ast.Name('info',ast.Load())],[]))]
   return node
  def visit_Compare(self,node):
   if len(node.ops)==1 and isinstance(node.ops[0],ast.NotIn) and len(node.comparators)==1 and isinstance(node.comparators[0],ast.Name) and node.comparators[0].id=='unused':
    counts['allow']+=1;node.comparators[0]=ast.BinOp(node.comparators[0],ast.BitOr(),ast.Name('_converted_MTP',ast.Load()))
   return self.generic_visit(node)
 tree=Extend().visit(tree)
 if counts!={'info':1,'allow':1}:raise ValueError('Expected two Giga loading adaptation sites changed')
 m=types.ModuleType('_native_giga_loading_info_v2');m.__file__=str(ROOT/BASE);m.__dict__.update(_normalize_loading_info=normalize_loading_info,_converted_MTP=CONVERTED_MTP)
 exec(compile(ast.fix_missing_locations(tree),m.__file__,'exec'),m.__dict__);_PRIVATE=m;return m
def install_loader(module):
 m=implementation();m.install_loader(module);Native=module.AutoModelForCausalLM
 class Model:
  @staticmethod
  def from_pretrained(*a,**k):
   model=Native.from_pretrained(*a,**k);report=model._giga_budget_loading
   report.update(unused_MTP_index_keys=210,allowed_native_converted_MTP_target_names=sorted(CONVERTED_MTP),native_conversion_mapping_sha256=CONVERSION_SHA,loading_info_containers='native sets recursively normalized to JSON lists; names/values unchanged',loading_info_adapter_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
   json.dumps(report,allow_nan=False);return model
 module.AutoModelForCausalLM=Model;return module
def tokenizer_proxy(native=None):return implementation().tokenizer_proxy(native)
def pinned_config(path):return implementation().pinned_config(path)
def normalize_config(value):return implementation().normalize_config(value)
class ScopedPEFT:
 def __new__(cls,*a,**k):return implementation().ScopedPEFT(*a,**k)
