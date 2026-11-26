"""Prospective target-context deduplication; no actual eligible data or training.

The frozen v1 release is imported without modification. Structural receipts here
do not perform source/language/context/heldout audits. Real immutable audit files
and a complete candidate graph remain mandatory future primary preparation.
"""
import copy
import hashlib
import importlib.util
from pathlib import Path
import re
from collections import Counter,defaultdict

# Loading this stdlib helper by file must not execute src.dataset.__init__,
# which imports the training stack. Keep the previous release byte-frozen.
_V1_PATH=Path(__file__).with_name('research_primary_protocol.py')
_V1_SHA256='3f33083610382bb4d0ce9f72ab91010ad34267a0d88d1c808275ac7aa8bd8eb4'
if hashlib.sha256(_V1_PATH.read_bytes()).hexdigest()!=_V1_SHA256:raise ValueError('Frozen primary protocol v1 changed')
_spec=importlib.util.spec_from_file_location('_frozen_primary_protocol_v1',_V1_PATH)
v1=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(v1)

PROFILE='primary_train_en_v2_target_context_dedup'
TARGET_RULE='canonical_RGB_ordered_context_target_NFKC_casefold_whitespace_v1'
OWNER_RULE='seed42_hash_semantic_key_unit_id_original_turn_lexical_tie_v1'
DEDUP_PROTOCOL='primary_global_target_ownership_v2'
ELIGIBILITY_PROTOCOL='primary_external_eligibility_binding_v2'
GRAPH_PROTOCOL='primary_complete_component_content_graph_v2'
FIELDS=('unit_id','source_id','task_group','source','original_id','kind','turns',
    'image_hashes','document_ids','dialogue_key','independent_QA_no_sibling_history')
HEX=re.compile('[a-f0-9]{64}')
SEED=v1.SEED
AUDITS=v1.AUDITS


def _hex(value):return isinstance(value,str) and HEX.fullmatch(value) is not None
def _seed(seed):
    if type(seed) is not int or seed!=SEED:raise ValueError('Fixed integer data seed42 required')
def _text(value):
    if not isinstance(value,str) or not value.strip():raise ValueError('Nonempty actual source text required')
    return value
def _links(images,documents,*,text_only=False):
    if (not isinstance(images,list) or len(images)!=(0 if text_only else 1) or any(not _hex(h) for h in images) or
        not isinstance(documents,list) or documents!=sorted(set(documents)) or
        any(not isinstance(x,str) or not x for x in documents)):
        raise ValueError('Canonical RGB/document identities differ')


def emitted_target(unit,turn_index):
    """Actual ordered model input, retaining even non-owned earlier answers."""
    turns=unit['turns']
    if type(turn_index) is not int or not 0<=turn_index<len(turns):raise ValueError('Original target turn index differs')
    if unit['kind']=='independent_qa' and (len(turns)!=1 or turn_index!=0):raise ValueError('Independent QA cannot inherit sibling history')
    messages=[]
    if unit['kind']=='genuine_dialogue':
        for earlier in turns[:turn_index]:
            messages.extend([{'role':'user','content':_text(earlier['user'])},
                {'role':'assistant','content':_text(earlier['assistant'])}])
    current=turns[turn_index];messages.append({'role':'user','content':_text(current['user'])})
    return {'image_hashes':list(unit['image_hashes']),'messages':messages,
        'question':current['user'],'answer':_text(current['assistant'])}


def target_context_key(unit,index):
    actual=emitted_target(unit,index)
    # Document IDs are leakage/provenance metadata, never seen by the model.
    # Same RGB cannot escape dedup through different document/source metadata;
    # different pages of one document are not collapsed into one model input.
    return v1.digest({'rule':TARGET_RULE,'RGB':actual['image_hashes'],
        'messages':[{'role':m['role'],'content':v1.normalise(m['content'])} for m in actual['messages']],
        'target':v1.normalise(actual['answer'])})


def _raw(unit):
    if unit.get('kind') not in ('independent_qa','genuine_dialogue'):raise ValueError('Declared sampling-unit kind required')
    v1.require_train(unit['source']);_links(unit['image_hashes'],unit['document_ids'])
    if unit.get('task_group')!=v1.group_for(unit['source_id']):raise ValueError('Source/group differs')
    if unit['source_id'] in ('docvqa','infographic_vqa') and not unit['document_ids']:raise ValueError('Original document mapping missing')
    identity={'source_id':unit['source_id'],'source':unit['source'],'original_id':unit['original_id'],'kind':unit['kind']}
    if not isinstance(unit['original_id'],str) or not unit['original_id'] or unit['unit_id']!=v1.digest(identity):
        raise ValueError('Stable original source/unit identity differs')
    if not isinstance(unit['turns'],list) or not unit['turns']:raise ValueError('Original turn list missing')
    for i in range(len(unit['turns'])):emitted_target(unit,i)
    return {key:unit.get(key) for key in FIELDS}


def _bind(unit):
    result=copy.deepcopy(unit);raw=_raw(result)
    keys=[target_context_key(result,i) for i in range(len(result['turns']))]
    exact=[v1.digest(emitted_target(result,i)) for i in range(len(result['turns']))]
    result.update(protocol=PROFILE,emitted_context_target_keys=keys,emitted_context_target_exact_sha256=exact,dedupe_key=v1.digest(keys),
        unit_content_sha256=v1.digest({'schema':2,'raw':raw,'target_keys':keys}))
    return result


def validate_unit(unit):
    raw=_raw(unit);keys=[target_context_key(unit,i) for i in range(len(unit['turns']))]
    if (unit.get('protocol')!=PROFILE or unit.get('emitted_context_target_keys')!=keys or
        unit.get('emitted_context_target_exact_sha256')!=[v1.digest(emitted_target(unit,i)) for i in range(len(unit['turns']))] or
        unit.get('dedupe_key')!=v1.digest(keys) or
        unit.get('unit_content_sha256')!=v1.digest({'schema':2,'raw':raw,'target_keys':keys})):
        raise ValueError('Actual unit content/target key changed after binding')


def normalise_finevision(*args,**kwargs):
    if kwargs.get('original_id') is None:raise ValueError('Actual original source ID required')
    return [_bind(u) for u in v1.normalise_finevision(*args,**kwargs)]
def normalise_gqa(*args,**kwargs):
    if kwargs.get('question_id') is None:raise ValueError('Actual original GQA question ID required')
    return _bind(v1.normalise_gqa(*args,**kwargs))


def bind_component_node(*,source_id,source,original_id,content,image_hashes=(),document_ids=(),
                        dialogue_key=None,conversation_key=None):
    """Bind exact external A0/text payload for graph links, never eligibility.

    This does not implement a caption/text sampler or certify original aliases.
    Future source audits must prove the supplied document/conversation links.
    """
    v1.require_train(source)
    if original_id is None or not str(original_id):raise ValueError('Actual external source ID required')
    if source_id not in ('pixmo_repack','smoltalk'):raise ValueError('Only declared external A0/text nodes supported')
    text_only=source_id=='smoltalk';images=list(image_hashes);docs=sorted(set(document_ids));_links(images,docs,text_only=text_only)
    if text_only and not _hex(conversation_key):raise ValueError('Stable text conversation component required')
    for key in (dialogue_key,conversation_key):
        if key is not None and not _hex(key):raise ValueError('Canonical dialogue/conversation component identity required')
    node={'protocol':PROFILE,'source_id':source_id,'source':copy.deepcopy(source),'original_id':str(original_id),
        'component_only':True,'content':copy.deepcopy(content),'image_hashes':images,'document_ids':docs,
        'dialogue_key':dialogue_key,'conversation_key':conversation_key}
    node['unit_id']=v1.digest([PROFILE,'component_only',source_id,source,str(original_id)])
    node['component_content_sha256']=v1.digest(node)
    return node


def _validate_component(node):
    if not node.get('component_only') or node.get('protocol')!=PROFILE:raise ValueError('Bound external component node required')
    raw={k:v for k,v in node.items() if k!='component_content_sha256'}
    if node.get('component_content_sha256')!=v1.digest(raw):raise ValueError('External source/link/content changed')
    v1.require_train(node['source']);_links(node['image_hashes'],node['document_ids'],text_only=node['source_id']=='smoltalk')
    if (node['source_id'] not in ('pixmo_repack','smoltalk') or
        node['unit_id']!=v1.digest([PROFILE,'component_only',node['source_id'],node['source'],node['original_id']])):
        raise ValueError('Stable declared external source/component identity differs')


def split_components(units,*,component_nodes=(),seed=SEED,validation_basis_points=200):
    _seed(seed)
    for unit in units:validate_unit(unit)
    for node in component_nodes:_validate_component(node)
    # Never filter out units whose targets lose dedup ownership: their original
    # RGB/document/dialogue links still belong to the complete candidate graph.
    return v1.split_components([*units,*component_nodes],seed=seed,validation_basis_points=validation_basis_points)


def _sealed(body):return dict(body,sha256=v1.digest(body))
def _verify(receipt,protocol):
    body={k:v for k,v in receipt.items() if k!='sha256'}
    if receipt.get('protocol')!=protocol or receipt.get('sha256')!=v1.digest(body):raise ValueError('Immutable receipt content/protocol differs')
    return body


def seal_component_graph(units,*,component_nodes=(),seed=SEED,validation_basis_points=200):
    """Structural exact-content census of ALL stages; no source audit claim."""
    partitions=split_components(units,component_nodes=component_nodes,seed=seed,
        validation_basis_points=validation_basis_points)
    census=[{'unit_id':u['unit_id'],'scope':'multimodal_target_unit','content_sha256':u['unit_content_sha256']} for u in units]
    census.extend({'unit_id':n['unit_id'],'scope':'external_A0_or_text_node','content_sha256':n['component_content_sha256']} for n in component_nodes)
    census.sort(key=lambda row:row['unit_id'])
    body={'schema_version':1,'protocol':GRAPH_PROTOCOL,'profile':PROFILE,'data_seed':seed,
        'validation_basis_points':validation_basis_points,'content_census':census,'assignments':partitions,
        'status':'complete_supplied_graph_content_only_real_source_audits_pending','primary_training_authorized':False}
    body['joint_graph_sha256']=v1.digest({'seed':seed,'validation_basis_points':validation_basis_points,
        'content_census':census,'assignments':partitions})
    return _sealed(body)


def deduplicate_targets(units,*,source_registry_sha256,candidate_turn_indices=None,seed=SEED):
    """Global across all MM sources BEFORE source quotas. No input mutation."""
    _seed(seed)
    if not _hex(source_registry_sha256):raise ValueError('Pinned source registry SHA required')
    ordered=sorted(units,key=lambda u:u['unit_id'])
    if len({u['unit_id'] for u in ordered})!=len(ordered):raise ValueError('Duplicate original sampling-unit identities')
    candidates=defaultdict(list);rows=[]
    if candidate_turn_indices is not None and set(candidate_turn_indices)!={u['unit_id'] for u in ordered}:
        raise ValueError('Explicit candidate turn mapping must cover the full joint MM pool')
    for unit in ordered:
        validate_unit(unit)
        indices=list(range(len(unit['turns']))) if candidate_turn_indices is None else candidate_turn_indices[unit['unit_id']]
        if (not isinstance(indices,list) or indices!=sorted(set(indices)) or
            any(type(i) is not int or not 0<=i<len(unit['turns']) for i in indices)):
            raise ValueError('Candidate original-turn mapping differs')
        rows.append({'unit_id':unit['unit_id'],'unit_content_sha256':unit['unit_content_sha256'],
            'candidate_turn_indices':indices,'target_keys':list(unit['emitted_context_target_keys']),
            'exact_target_sha256':list(unit['emitted_context_target_exact_sha256'])})
        for index in indices:candidates[unit['emitted_context_target_keys'][index]].append((unit['unit_id'],index))
    owners={}
    for key,options in sorted(candidates.items()):
        uid,index=min(options,key=lambda x:(v1.digest([seed,OWNER_RULE,key,x[0],x[1]]),x[0],x[1]))
        owners[key]={'unit_id':uid,'turn_index':index}
    for row in rows:
        row['eligible_turn_indices']=[i for i in row['candidate_turn_indices'] if owners[row['target_keys'][i]]=={'unit_id':row['unit_id'],'turn_index':i}]
    receipt=_sealed({'schema_version':1,'protocol':DEDUP_PROTOCOL,'profile':PROFILE,'data_seed':seed,
        'normalization_rule':TARGET_RULE,'owner_rule':OWNER_RULE,'source_registry_sha256':source_registry_sha256,
        'joint_pool_sha256':v1.digest([{k:r[k] for k in ('unit_id','unit_content_sha256','candidate_turn_indices')} for r in rows]),
        'joint_pool_scope':'all_multimodal_target_units_external_stage_content_bound_by_complete_graph_receipt',
        'units':rows,'target_owners':owners,'status':'structural_target_ownership_only_source_audits_pending',
        'real_source_audits_executed':False,'primary_training_authorized':False})
    by_id={r['unit_id']:r for r in rows};result=[]
    for unit in ordered:
        value=copy.deepcopy(unit);value.update(eligible_turn_indices=by_id[unit['unit_id']]['eligible_turn_indices'],
            target_ownership_receipt_sha256=receipt['sha256']);result.append(value)
    return result,receipt


def verify_target_receipt(units,receipt,*,full_pool=False):
    _verify(receipt,DEDUP_PROTOCOL);_seed(receipt.get('data_seed'))
    if receipt.get('normalization_rule')!=TARGET_RULE or receipt.get('owner_rule')!=OWNER_RULE:
        raise ValueError('Preregistered semantic/winner rule differs')
    rows={r['unit_id']:r for r in receipt['units']}
    if len(rows)!=len(receipt['units']):raise ValueError('Duplicate bound unit rows')
    for unit in units:
        validate_unit(unit);row=rows.get(unit['unit_id'])
        if (not row or row['unit_content_sha256']!=unit['unit_content_sha256'] or
            row['target_keys']!=unit['emitted_context_target_keys'] or
            row['exact_target_sha256']!=unit['emitted_context_target_exact_sha256'] or
            unit.get('eligible_turn_indices')!=row['eligible_turn_indices'] or
            unit.get('target_ownership_receipt_sha256')!=receipt['sha256']):
            raise ValueError('Candidate content/owned turn map/immutable global receipt differs')
    if full_pool:
        _,expected=deduplicate_targets(units,source_registry_sha256=receipt['source_registry_sha256'],
            candidate_turn_indices={k:r['candidate_turn_indices'] for k,r in rows.items()})
        if expected!=receipt:raise ValueError('Complete global ownership differs')


def bind_external_eligibility(units,receipt,component_graph,*,component_nodes=(),audit_receipts):
    """Receipt interface for FUTURE already-executed, externally verified audits.

    It cannot execute or certify an audit from a boolean. Each audit must bind
    the same immutable source registry and joint raw-content pool; production
    preparation still must verify every referenced audit file's bytes/results.
    """
    verify_target_receipt(units,receipt,full_pool=True)
    _verify(component_graph,GRAPH_PROTOCOL)
    if component_graph!=seal_component_graph(units,component_nodes=component_nodes):
        raise ValueError('Complete graph/source/content census differs from supplied actual candidates')
    partitions=component_graph['assignments']
    expected_ids={u['unit_id'] for u in units}
    if not expected_ids<=set(partitions):raise ValueError('Complete candidate component split missing')
    for audit in AUDITS:
        item=audit_receipts.get(audit,{})
        if (item.get('status')!='passed' or not _hex(item.get('receipt_sha256')) or
            item.get('joint_pool_sha256')!=receipt['joint_pool_sha256'] or
            item.get('joint_graph_sha256')!=component_graph['joint_graph_sha256'] or
            item.get('source_registry_sha256')!=receipt['source_registry_sha256']):
            raise ValueError('Actual immutable source/content-bound audit receipt missing: '+audit)
    assigned=[]
    for unit in sorted(units,key=lambda u:u['unit_id']):
        entry=partitions[unit['unit_id']]
        if set(entry)!={'partition','component_id'} or entry.get('partition') not in ('train','validation') or not _hex(entry.get('component_id')):
            raise ValueError('Verified complete component split differs')
        assigned.append({'unit_id':unit['unit_id'],'unit_content_sha256':unit['unit_content_sha256'],
            'eligible_turn_indices':unit['eligible_turn_indices'],**entry})
    return _sealed({'schema_version':1,'protocol':ELIGIBILITY_PROTOCOL,'profile':PROFILE,
        'target_ownership_receipt_sha256':receipt['sha256'],'source_registry_sha256':receipt['source_registry_sha256'],
        'joint_pool_sha256':receipt['joint_pool_sha256'],'component_assignment_map_sha256':v1.digest(partitions),
        'joint_graph_sha256':component_graph['joint_graph_sha256'],'component_graph_receipt':copy.deepcopy(component_graph),
        'audits':copy.deepcopy(audit_receipts),
        'assignments':assigned,'status':'external_audit_descriptors_bound_not_executed_here',
        'primary_training_authorized':False})


def select_exposures(units,quota,*,target_receipt,eligibility_receipt,seed=SEED):
    _seed(seed);verify_target_receipt(units,target_receipt)
    _verify(eligibility_receipt,ELIGIBILITY_PROTOCOL)
    graph=eligibility_receipt['component_graph_receipt'];_verify(graph,GRAPH_PROTOCOL)
    if (eligibility_receipt.get('target_ownership_receipt_sha256')!=target_receipt['sha256'] or
        eligibility_receipt.get('joint_pool_sha256')!=target_receipt['joint_pool_sha256'] or
        eligibility_receipt.get('source_registry_sha256')!=target_receipt['source_registry_sha256'] or
        eligibility_receipt.get('joint_graph_sha256')!=graph['joint_graph_sha256'] or
        eligibility_receipt.get('component_assignment_map_sha256')!=v1.digest(graph['assignments'])):
        raise ValueError('Eligibility and joint raw-content/source ownership receipt differ')
    assignments={row['unit_id']:row for row in eligibility_receipt['assignments']}
    graph_census={row['unit_id']:row for row in graph['content_census']}
    if len(assignments)!=len(eligibility_receipt['assignments']) or len(graph_census)!=len(graph['content_census']):
        raise ValueError('Duplicate immutable graph/eligibility projection rows')
    selected=[]
    for unit in units:
        row=assignments.get(unit['unit_id'])
        projection=graph['assignments'].get(unit['unit_id']);content=graph_census.get(unit['unit_id'],{})
        if (not row or projection!={'partition':row.get('partition'),'component_id':row.get('component_id')} or
            content.get('scope')!='multimodal_target_unit' or content.get('content_sha256')!=unit['unit_content_sha256']):
            raise ValueError('Eligibility component assignment/raw content differs from bound complete graph')
        if (not row or row['unit_content_sha256']!=unit['unit_content_sha256'] or row['eligible_turn_indices']!=unit['eligible_turn_indices'] or
            row['partition']!='train' or any(unit.get('audit_status',{}).get(a) is not True for a in AUDITS)):
            raise ValueError('Candidate actual content/partition/audit binding is incomplete')
        if not unit['eligible_turn_indices']:continue
        selected.append(unit)
    if len({u['unit_id'] for u in selected})!=len(selected) or len({u['source_id'] for u in selected})!=1:
        raise ValueError('One deduplicated source pool required')
    if type(quota) is not int or not 0<quota<=v1.MAX_REPEAT*len(selected):raise ValueError('Insufficient post-dedup audited source pool/cap4')
    result=[];occurrences=Counter()
    for cycle in range((quota+len(selected)-1)//len(selected)):
        ordered=sorted(selected,key=lambda u:(v1.digest([seed,'primary_units',cycle,u['unit_id']]),u['unit_id']))
        for unit in ordered:
            occurrence=occurrences[unit['unit_id']];occurrences[unit['unit_id']]+=1
            indices=unit['eligible_turn_indices'];phase=int(v1.digest([seed,'dialogue_turn_phase',unit['unit_id']]),16)%len(indices)
            turn=indices[(phase+occurrence)%len(indices)];actual=emitted_target(unit,turn)
            result.append({**actual,'exposure_id':v1.digest([PROFILE,unit['unit_id'],occurrence]),
                'sampling_unit_id':unit['unit_id'],'unit_occurrence':occurrence,'source_id':unit['source_id'],
                'task_group':unit['task_group'],'source':copy.deepcopy(unit['source']),
                'component_id':assignments[unit['unit_id']]['component_id'],'partition':'train',
                'turn_index':turn,'eligible_original_turn_indices':list(indices),
                'emitted_context_target_key':unit['emitted_context_target_keys'][turn],
                'unit_content_sha256':unit['unit_content_sha256'],'target_ownership_receipt_sha256':target_receipt['sha256'],
                'eligibility_receipt_sha256':eligibility_receipt['sha256'],'sampling_unit_weight':1,'supervised_target_turns':1})
            if len(result)==quota:return result
    raise AssertionError('Exposure accounting failed')


def policy():
    result=copy.deepcopy(v1.policy());result.update(profile=PROFILE,
        target_identity_rule=TARGET_RULE,target_owner_rule=OWNER_RULE,
        global_emitted_target_dedup_before_source_quota=True,
        remaining_dialogue_targets_keep_full_original_prior_history=True,
        source_registry_revision=3,external_immutable_audit_files_verified=False,
        actual_primary_source_audits_executed=False,
        complete_external_A0_text_raw_content_graph_receipt_required=True,
        status='prospective_protocol_primary_P01_P02_partial')
    return result


primary_quotas=v1.primary_quotas
proportional_quotas=v1.proportional_quotas
