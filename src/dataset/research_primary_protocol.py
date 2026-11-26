"""Prospective primary data units, leakage components and bounded sampling.

Pure CPU/stdlib. This is not a downloader, a completed training manifest, or an
authorization to change the frozen pilot. Eligibility audits are mandatory.
"""
from collections import Counter,defaultdict
import hashlib
import json
import math
import re
import unicodedata

PROFILE='primary_train_en_v1_pool_proportional'
SEED=42
GROUPS={'general_vqa':{'fraction_basis_points':3500,'sources':('vqav2','gqa','llava_instruct')},
    'ocr_documents':{'fraction_basis_points':3000,'sources':('textvqa','docvqa','infographic_vqa')},
    'charts_diagrams':{'fraction_basis_points':2000,'sources':('chartqa','ai2d')},
    'knowledge_reasoning':{'fraction_basis_points':1500,'sources':('aokvqa','scienceqa')}}
FINEVISION_CONFIGS={'vqav2':'vqav2','llava_instruct':'LLaVA_Instruct_150K','textvqa':'textvqa',
    'docvqa':'docvqa','infographic_vqa':'infographic_vqa','chartqa':'chartqa','ai2d':'ai2d_merged',
    'aokvqa':'aokvqa','scienceqa':'scienceqa'}
MAX_REPEAT=4
AUDITS=('source_bytes_and_original_train_split_verified','language_english_passed',
    'all_four_backbone_native_context_census_passed','record_and_conversation_dedup_passed',
    'all_declared_heldout_RGB_and_source_ids_disjoint','all_official_heldout_coverage_complete',
    'near_image_and_document_review_complete','all_15902_text_heldout_inputs_screened',
    'document_and_dialogue_component_links_verified','licenses_and_source_lineage_reviewed')


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
def digest(value):return hashlib.sha256(canonical(value).encode()).hexdigest()
def normalise(value):
    if not isinstance(value,str):raise ValueError('Expected source text')
    return ' '.join(unicodedata.normalize('NFKC',value).casefold().split())
def group_for(source):
    matches=[g for g,v in GROUPS.items() if source in v['sources']]
    if len(matches)!=1:raise ValueError('Undeclared primary multimodal source')
    return matches[0]
def require_train(source):
    if (source.get('split')!='train' or not isinstance(source.get('repo'),str) or not source['repo'] or
            not isinstance(source.get('revision'),str) or not re.fullmatch('[a-f0-9]{40,64}',source['revision'])):
        raise ValueError('Explicit pinned original TRAIN provenance required')


def _turn(user,assistant):
    if not isinstance(user,str) or not user.strip() or not isinstance(assistant,str) or not assistant.strip():
        raise ValueError('Both public user and supervised assistant text are required')
    return {'user':user,'assistant':assistant}


def _unit(source_id,source,image_hashes,turns,*,kind,original_id,document_ids=(),dialogue_verified=False):
    require_train(source)
    if len(image_hashes)!=1 or any(not isinstance(h,str) or not re.fullmatch('[a-f0-9]{64}',h) for h in image_hashes):
        raise ValueError('Primary scope requires one decoded canonical RGB image identity')
    if kind=='genuine_dialogue' and not dialogue_verified:
        raise ValueError('Original ordered dialogue provenance has not been verified')
    if source_id in ('docvqa','infographic_vqa') and not document_ids:
        raise ValueError('Original document component mapping required before primary eligibility')
    if not turns:raise ValueError('Empty primary sampling unit')
    identity={'source_id':source_id,'source':source,'original_id':str(original_id),'kind':kind}
    return {'unit_id':digest(identity),'source_id':source_id,'task_group':group_for(source_id),
        'source':dict(source),'original_id':str(original_id),'kind':kind,'turns':turns,
        'image_hashes':list(image_hashes),'document_ids':sorted(set(document_ids)),
        'dialogue_key':digest([source['repo'],source['revision'],source.get('config'),str(original_id)]) if kind=='genuine_dialogue' else None,
        'independent_QA_no_sibling_history':kind=='independent_qa','audit_status':{k:False for k in AUDITS},
        'dedupe_key':digest({'images':image_hashes,'kind':kind,
            'turns':[{k:normalise(v) for k,v in t.items()} for t in turns]})}


def normalise_finevision(row,*,source_id,source,original_id,image_hashes,document_ids=(),dialogue_verified=False):
    """Independent QA per texts entry, except explicitly verified LLaVA order.

    No image decoding here: a future preparer supplies audited canonical RGB
    identity and original document/dialogue links. Do not assert eligibility.
    """
    if source_id not in FINEVISION_CONFIGS or source.get('config')!=FINEVISION_CONFIGS[source_id]:
        raise ValueError('FineVision source alias/config mismatch')
    if not isinstance(row.get('images'),list) or len(row['images'])!=1:raise ValueError('Single-image source row required')
    texts=row.get('texts')
    if not isinstance(texts,list) or not texts:raise ValueError('FineVision texts sequence missing')
    turns=[_turn(t.get('user'),t.get('assistant')) for t in texts]
    if source_id=='llava_instruct':
        return [_unit(source_id,source,image_hashes,turns,kind='genuine_dialogue',original_id=original_id,
            document_ids=document_ids,dialogue_verified=dialogue_verified)]
    return [_unit(source_id,source,image_hashes,[turn],kind='independent_qa',
        original_id=f'{original_id}:qa:{i}',document_ids=document_ids) for i,turn in enumerate(turns)]


def normalise_gqa(instruction,image,*,source,question_id,image_hashes):
    """Strict balanced-TRAIN instruction→original-image-ID join, no zip order."""
    require_train(source)
    if source.get('config')!='train_balanced_instructions' or source.get('image_config')!='train_balanced_images':
        raise ValueError('GQA balanced TRAIN instructions/images required')
    if str(instruction.get('id'))!=str(question_id) or instruction.get('isBalanced') is not True:
        raise ValueError('GQA original balanced question identity differs')
    image_id=str(instruction.get('imageId'))
    if not image_id or image_id=='None' or str(image.get('id'))!=image_id:
        raise ValueError('GQA exact original image-ID join failed')
    turn=_turn(instruction.get('question'),instruction.get('answer'))
    result=_unit('gqa',source,image_hashes,[turn],kind='independent_qa',original_id=question_id)
    result['original_image_id']=image_id
    return result


def split_components(units,*,seed=SEED,validation_basis_points=200):
    """Union all image/document/dialogue links before any flatten or budget.

    Transitive source links are retained even for later deduplicated units.
    Caller must assemble the joint A0/A1/SFT/text candidate graph first.
    """
    if type(seed) is not int or seed!=SEED or type(validation_basis_points) is not int or validation_basis_points!=200:
        raise ValueError('Preregistered data seed42 and2% component holdout required')
    by_id={u['unit_id']:u for u in units}
    if len(by_id)!=len(units):raise ValueError('Duplicate primary unit IDs')
    parent={key:key for key in by_id}
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    def union(x,y):
        x,y=find(x),find(y)
        if x!=y:parent[max(x,y)]=min(x,y)
    seen={}
    for key,u in by_id.items():
        links=['rgb:'+h for h in u.get('image_hashes',[])]+['document:'+s for s in u.get('document_ids',[])]
        for name in ('dialogue_key','conversation_key'):
            if u.get(name):links.append(name+':'+u[name])
        if not links:raise ValueError('A sampling unit needs a stable leakage component')
        for link in links:
            if link in seen:union(key,seen[link])
            else:seen[link]=key
    components=defaultdict(list)
    for key in by_id:components[find(key)].append(key)
    result={}
    for keys in components.values():
        identity=digest(sorted(keys));heldout=int(digest([seed,'primary_component_split',identity]),16)%10000<validation_basis_points
        for key in keys:result[key]={'component_id':identity,'partition':'validation' if heldout else 'train'}
    return result


def proportional_quotas(counts,target,*,repeat_cap=MAX_REPEAT):
    """Largest remainder from actual eligible pools; never observed scores."""
    if type(target) is not int or target<=0 or repeat_cap!=MAX_REPEAT:raise ValueError('Fixed positive budget and preregistered cap4 required')
    if not counts or any(type(n) is not int or n<=0 for n in counts.values()):raise ValueError('Every declared source requires a positive audited eligible pool')
    total=sum(counts.values())
    if target>repeat_cap*total:raise ValueError('Insufficient audited pool at cap4; never silently oversample or redistribute groups')
    base={k:target*n//total for k,n in counts.items()}
    order=sorted(counts,key=lambda k:(-(target*counts[k]%total),k))
    for key in order[:target-sum(base.values())]:base[key]+=1
    if any(base[k]==0 or base[k]>repeat_cap*counts[k] for k in counts):raise ValueError('Source coverage/cap cannot satisfy the declared budget')
    return base


def primary_quotas(pool_counts):
    expected={s for g in GROUPS.values() for s in g['sources']}
    if set(pool_counts)!=expected:raise ValueError('All nine FineVision configs plus direct GQA are required')
    result={}
    for group,spec in GROUPS.items():
        budget=500000*spec['fraction_basis_points']//10000
        result[group]={'exposures':budget,'sources':proportional_quotas({s:pool_counts[s] for s in spec['sources']},budget)}
    assert sum(v['exposures'] for v in result.values())==500000
    return result


def select_exposures(units,quota,*,seed=SEED):
    """Select only audited TRAIN units; one genuine-dialogue turn per exposure."""
    if type(seed) is not int or seed!=SEED:raise ValueError('Preregistered data seed42 is fixed across model seeds')
    if type(quota) is not int or quota<=0 or quota>MAX_REPEAT*len(units):raise ValueError('Insufficient pool/cap4')
    if (len({u['unit_id'] for u in units})!=len(units) or
            len({u['dedupe_key'] for u in units})!=len(units)):
        raise ValueError('Units must already be globally deduplicated by identity AND normalized content')
    if len({u['source_id'] for u in units})!=1:raise ValueError('One explicit source pool per sampling call')
    for u in units:
        if u.get('partition')!='train' or any(u.get('audit_status',{}).get(k) is not True for k in AUDITS):
            raise ValueError('Primary eligibility audit or train split is incomplete')
    result=[];occurrences=Counter()
    for cycle in range(math.ceil(quota/len(units))):
        ordered=sorted(units,key=lambda u:(digest([seed,'primary_units',cycle,u['unit_id']]),u['unit_id']))
        for u in ordered:
            occurrence=occurrences[u['unit_id']];occurrences[u['unit_id']]+=1
            phase=int(digest([seed,'dialogue_turn_phase',u['unit_id']]),16)%len(u['turns'])
            turn=(phase+occurrence)%len(u['turns']) if u['kind']=='genuine_dialogue' else 0
            messages=[]
            if u['kind']=='genuine_dialogue':
                for earlier in u['turns'][:turn]:messages.extend([{'role':'user','content':earlier['user']},{'role':'assistant','content':earlier['assistant']}])
            selected=u['turns'][turn];messages.append({'role':'user','content':selected['user']})
            result.append({'exposure_id':digest([PROFILE,u['unit_id'],occurrence]),'sampling_unit_id':u['unit_id'],
                'unit_occurrence':occurrence,'source_id':u['source_id'],'task_group':u['task_group'],
                'component_id':u['component_id'],'partition':'train','source':u['source'],
                'image_hashes':u['image_hashes'],'question':selected['user'],'answer':selected['assistant'],
                'messages':messages,'turn_index':turn,'eligible_dialogue_turns':len(u['turns']),
                'sampling_unit_weight':1,'supervised_target_turns':1})
            if len(result)==quota:return result
    raise AssertionError('Exposure accounting failed')


def policy():
    return {'schema_version':1,'profile':PROFILE,'status':'prospective_protocol_primary_P01_P02_partial',
        'data_seed':SEED,'model_seeds_do_not_change_manifests':True,'groups':GROUPS,
        'sft_mm_train_exposures':500000,'sft_text_train_exposures':88236,'text_fraction':88236/588236,
        'text_count_rule':'ceil(500000*15/85); record fraction rounded upward by <1 record',
        'A0_train_exposures':200000,'A0_unique_canonical_RGB_images':200000,'A0_repeat_cap':1,
        'A1_train_exposures':10000,'A1_source':'deterministic 10K subset of A0 TRAIN, same across all11 branches',
        'A1_repeat_cap_within_stage':1,'A1_additional_unique_images':0,'multimodal_repeat_cap_per_unit_per_epoch':MAX_REPEAT,
        'text_repeat_cap':1,'internal_validation_component_basis_points':200,
        'within_group_allocation':'largest remainder proportional to actual post-audit eligible sampling-unit counts; lexical source-ID remainder ties',
        'source_policy_revision_reason':'ScienceQA6149 prefilter assistant entries cap4 capacity24596 <37500 old equal-source quota; data-capacity evidence, no pilot scores',
        'one_exposure':'one supervised target turn; verified dialogue preserves only genuine preceding history',
        'LLaVA_unit':'one verified original ordered dialogue, uniform unit mass; cyclic eligible target turn chosen by fixed hashed phase',
        'other_VQA_unit':'one independent QA, never inherit sibling questions/answers',
        'refill':'scan pinned sources in fixed hash-ranked shard/row order; apply all audits first; allocate from actual eligible pools; fail on insufficient capacity',
        'unique_counts_required':['source parent rows','sampling units','dialogues','supervised target turns','RGB images','document components','exposures','repeated units','target tokens'],
        'required_audits':list(AUDITS),'primary_manifest_exists':False,'actual_eligible_pool_counts_known':False,
        'frozen_pilot_or_runtime_modified':False,'GPU_training_authorized':False}
