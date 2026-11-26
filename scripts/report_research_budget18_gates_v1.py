#!/usr/bin/env python3
"""Export the complete R3 gate census from finalized A1/SFT routing logs."""
from __future__ import annotations
import argparse,csv,hashlib,io,json,math,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RELEASE_SHA='910448c7fd680ec949d6521e32716aee2832a99fc534d185f0a06b226c0ec1c0'


def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def publish(p,raw):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists() and p.read_bytes()!=raw:raise ValueError('Preserve existing census: '+str(p))
    if not p.exists():p.write_bytes(raw)


def collect(run):
    run=Path(run).resolve(strict=True);config=read(run/'resolved_config.json');done=read(run/'completed.json')
    if (not run.name.endswith('__budget18_s42') or config.get('recipe')!='1c' or config.get('stage')!='SFT'
            or config.get('seed')!=42 or done.get('steps')!=56 or done.get('records')!=3529):
        raise ValueError('Only completed budget18 R3 supported')
    data=config.get('budget18_data_proof',config.get('budget_data_proof',{}))
    if data.get('sha256')!=RELEASE_SHA:raise ValueError('Exact reduced data release required')
    initial=Path(config['initial_checkpoint']);a1=(initial if initial.is_absolute() else ROOT/initial).parent.parent
    paths=[run/'resolved_config.json',run/'completed.json',run/'routing.jsonl',a1/'routing.jsonl',
           ROOT/'src/model/research_moe.py',ROOT/'src/eval/research_selection.py',Path(__file__).resolve()]
    source={str(p):sha(p) for p in paths};snapshots={}
    for stage,path,step in [('A1',a1/'routing.jsonl',4),('SFT',run/'routing.jsonl',56)]:
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        if len(rows)!=1 or rows[0]['step']!=step:raise ValueError('Exact single final snapshot required')
        snapshots[stage]=rows[0]
    count=len(snapshots['SFT']['layers'])
    if not count or len(snapshots['A1']['layers'])!=count:raise ValueError('MoE block census differs')
    records=[];summaries={};roles={}
    owner=read(run/'.checkpoint_manager.json')['owner']
    for role in ('best','last'):
        marker=read(run/role/'.research_checkpoint.json')
        if marker.get('owner')!=owner or marker.get('step')!=56:raise ValueError('Current role ownership/step differs')
        roles[role]={'step':56,'logical_checkpoint':str(run/role/'state.pt')}
    for stage,snapshot in snapshots.items():
        grouped={'old':[],'new':[]}
        dispatch={mod:{'all_slots':0,'new_slots':0} for mod in ('text','visual')}
        for ordinal,layer in enumerate(snapshot['layers']):
            old=layer['old_experts'];new=layer['new_experts'];gates=layer['modality_gates_text_visual'];prior=snapshots['A1']['layers'][ordinal]
            if (layer['family']!=1 or layer['policy']!='c' or new!=2 or old<=0 or len(gates)!=old+new
                    or (old,new)!=(prior['old_experts'],prior['new_experts'])):raise ValueError('Family/order/shape differs')
            for expert,values in enumerate(gates):
                if len(values)!=2 or any(not math.isfinite(x) or not 0<x<1 for x in values):raise ValueError('Invalid sigmoid gate')
                t,v=values;pt,pv=prior['modality_gates_text_visual'][expert];origin='old' if expert<old else 'new'
                record={'stage':stage,'MoE_block_ordinal':ordinal,'expert':expert,'origin':origin,
                        'text_gate':t,'visual_gate':v,'visual_minus_text':v-t,
                        'text_minus_A1':t-pt,'visual_minus_A1':v-pv}
                records.append(record);grouped[origin].append(record)
            for i,mod in enumerate(('text','visual')):
                slots=layer['dispatch_counts_text_visual'][i]
                if len(slots)!=old+new or any(x<0 for x in slots):raise ValueError('Dispatch shape/value differs')
                dispatch[mod]['all_slots']+=sum(slots);dispatch[mod]['new_slots']+=sum(slots[old:])
        summary={}
        for origin,values in grouped.items():
            gates=[r[k] for r in values for k in ('text_gate','visual_gate')]
            summary[origin]={'expert_block_pairs':len(values),'gate_values':len(gates),
                'text_mean':statistics.mean(r['text_gate'] for r in values),'visual_mean':statistics.mean(r['visual_gate'] for r in values),
                'mean_visual_minus_text':statistics.mean(r['visual_minus_text'] for r in values),
                'min_gate':min(gates),'max_gate':max(gates),
                'visual_larger_pairs':sum(r['visual_minus_text']>0 for r in values),
                'text_larger_pairs':sum(r['visual_minus_text']<0 for r in values),
                'max_abs_change_from_A1':max(abs(r[k]) for r in values for k in ('text_minus_A1','visual_minus_A1'))}
        for mod,d in dispatch.items():d['new_fraction']=d['new_slots']/d['all_slots'] if d['all_slots'] else None
        summaries[stage]={'parameters':summary,'dispatch':dispatch,
            'dispatch_scope':'final caption validation microbatch' if stage=='A1' else 'last training microbatch; selection mode restores diagnostic buffers'}
    if any(sha(p)!=s for p,s in source.items()):raise ValueError('Source changed during census')
    report={'status':'completed_log_census','run':str(run),'model':config['model'],'vision':config['vision'],
        'MoE_blocks':count,'roles':roles,'initial_gate':.9,'columns':['text','visual'],
        'application':'g * Expert(g * hidden_state), after top-k selection',
        'interpretation':'Static parameter census and explicitly scoped dispatch snapshots; no epoch-wide traffic, causal specialization or accuracy attribution.',
        'activation_cast_note':'Selected sigmoids are cast to dispatch-state dtype. Small FP32 preferences can round to the same BF16 value; effective activation casts are not runtime-measured by this log census.',
        'SmallThinker_scale_note':'Its biasless ReGLU expert has ideal real-arithmetic g^3 scaling. BF16 rounding is not that exact identity; other architectures are not assumed homogeneous.',
        'checkpoint_tensor_verification_claimed':False,'source_sha256':source,'snapshots':snapshots,'summary':summaries,'CSV_rows':len(records)}
    return report,records


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--output',required=True);p.add_argument('--plot',action='store_true');a=p.parse_args()
    report,rows=collect(a.run);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    publish(out/'report.json',json.dumps(report,ensure_ascii=False,indent=2).encode()+b'\n')
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows);publish(out/'gates.csv',stream.getvalue().encode())
    if a.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.family':'DejaVu Sans','svg.hashsalt':'budget18-gates-v1'})
        final=report['snapshots']['SFT']['layers'];total=final[0]['old_experts']+2
        values=[[1000*(layer['modality_gates_text_visual'][e][1]-layer['modality_gates_text_visual'][e][0]) for layer in final] for e in range(total)]
        if any(layer['old_experts']+2!=total for layer in final):raise ValueError('Heatmap requires uniform expert counts')
        bound=max(1e-9,max(abs(x) for r in values for x in r));fig,ax=plt.subplots(figsize=(10,6),layout='constrained')
        im=ax.imshow(values,origin='lower',aspect='auto',interpolation='nearest',cmap='RdBu_r',vmin=-bound,vmax=bound)
        title=report['model'].split('/')[-1]+' / '+report['vision'].split('/')[-1]+'\nR3 · static modality gates · SFT56'
        ax.axhline(total-2.5,color='#222222',linewidth=1);ax.set(xlabel='MoE block ordinal',ylabel='Expert index',title=title)
        fig.colorbar(im,ax=ax,label='Visual gate − text gate (×10⁻³)');fig.supxlabel('Final two experts are new. Parameter preference does not establish causal specialization.',fontsize=9)
        for suffix in ('png','svg','pdf'):
            metadata={'Creator':'Budget18 gate census'}
            if suffix=='pdf':metadata.update(CreationDate=None,ModDate=None)
            fig.savefig(out/('gates.'+suffix),dpi=180,metadata=metadata)
        plt.close(fig)
    print(json.dumps({'status':report['status'],'blocks':report['MoE_blocks'],'rows':len(rows),'output':str(out),'GPU_used':False}))


if __name__=='__main__':main()
