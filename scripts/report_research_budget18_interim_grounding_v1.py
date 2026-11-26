#!/usr/bin/env python3
"""Audit the available preplanned image/shuffle contrast for all three SigLIP methods."""
from __future__ import annotations
import hashlib,json,math,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));os.environ['CUDA_VISIBLE_DEVICES']=''
from src.utils.research_storage import _atomic_json
SOURCE='scripts/research_grounding_queue.py'
SOURCE_SHA='b99dfd413edd4176288734262197a72b3df4720e01d562e4b76f625b70d0a1d9'


def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return {'path':str(p),'sha256':sha(p),'bytes':Path(p).stat().st_size}


def collect():
    if sha(ROOT/SOURCE)!=SOURCE_SHA:raise ValueError('Frozen image-cluster bootstrap changed')
    from scripts.research_grounding_queue import paired_image_bootstrap
    from src.eval.research_metrics import score_prediction,aggregate_scores
    state=read(ROOT/'runs/research/budget18_queue_state_v1.json')
    manifest=ROOT/'datasets/research/budget18-en-v1/heldout_evaluation.jsonl'
    records=[json.loads(line) for line in manifest.read_text().splitlines()]
    by_id={r['record_id']:r for r in records}
    if len(records)!=512 or len(by_id)!=512:raise ValueError('Exact unique512 slice required')
    results=[]
    for recipe in ('1b','4f','1c'):
        name=f'smallthinker4__siglip2__{recipe}__SFT__budget18_s42'
        row=state['jobs'][name];validation=row['checkpoint_validation']
        replay=read(validation['report'])
        if (validation.get('status')!='completed' or replay.get('status')!='passed'
                or replay.get('mode')!='full_gpu_replay' or sha(validation['report'])!=validation['report_sha256']):
            raise ValueError('Source-bound full GPU replay required')
        for entry in replay['checkpoints']:
            st=Path(entry['checkpoint_path']).stat()
            if st.st_size!=entry['checkpoint_size_bytes'] or st.st_mtime_ns!=entry['checkpoint_mtime_ns']:
                raise ValueError('Replayed checkpoint stat changed')
        views={};scores={};bindings={};checkpoint=None
        for mode,key in [('none','evaluation'),('shuffle','grounding_shuffle')]:
            proof=row[key]
            if proof.get('status')!='completed' or proof.get('records')!=512:
                raise ValueError('Two complete verified512 views required')
            out=Path(proof['output']);protocol=read(out/'protocol.json');summary=read(out/'summary.json')
            predictions=[json.loads(line) for line in (out/'predictions.jsonl').read_text().splitlines()]
            # Native protocol identity uses sorted JSON with default separators.
            protocol_sha=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
            if protocol_sha!=proof['protocol_sha256'] or protocol_sha!=summary['protocol_sha256']:
                raise ValueError('Native protocol semantic hash differs')
            cp=proof['checkpoint_sha256']
            if cp!=summary['checkpoint_sha256'] or cp!=protocol['checkpoint_sha256']:
                raise ValueError('Checkpoint identity differs')
            if checkpoint is None:checkpoint=cp
            if cp!=checkpoint:raise ValueError('Contrast must use one checkpoint')
            if proof['manifest_sha256']!=sha(manifest):raise ValueError('Contrast source slice differs')
            if protocol.get('image_ablation',{}).get('variant','none')!=mode:
                raise ValueError('Intervention variant differs')
            if len(predictions)!=512 or len({r['record_id'] for r in predictions})!=512 or {r['record_id'] for r in predictions}!=set(by_id):
                raise ValueError('Unique full512 raw rows required')
            for prediction in predictions:
                expected=score_prediction(by_id[prediction['record_id']],prediction['prediction'])
                if prediction['protocol_sha256']!=protocol_sha or not math.isclose(float(prediction['score']),expected,abs_tol=1e-12):
                    raise ValueError('Raw prediction protocol/score differs')
            aggregate=aggregate_scores(predictions,require_four_groups=True)
            if aggregate['group_macro_score']!=summary['group_macro_score'] or summary.get('complete_manifest') is not True:
                raise ValueError('Summary differs from actual predictions')
            views[mode]=predictions;scores[mode]=aggregate['group_macro_score']
            bindings[mode]={n:bind(out/n) for n in ('protocol.json','summary.json','predictions.jsonl')}
        paired=paired_image_bootstrap(views['none'],{'shuffle':views['shuffle']},records,replicates=1000,seed=42)['shuffle']
        if recipe!='1c':
            existing=read(row['paired_summary']['path'])['paired_image_group_bootstrap']['shuffle']
            if paired!=existing:raise ValueError('Interim computation differs from completed native shuffle interval')
        results.append({'run':name,'recipe':recipe,'primary_scores':scores,'paired_image_cluster_bootstrap':paired,
                        'checkpoint_sha256':checkpoint,'source_bindings':bindings,
                        'full_four_view_configuration_completed':row['status']=='completed'})
    return {'status':'interim_three_method_image_shuffle_audit','created_unix':time.time(),'rows':results,
            'manifest':bind(manifest),'bootstrap_source':bind(ROOT/SOURCE),'source':bind(Path(__file__)),
            'scope':'preplanned image-minus-shuffle contrast, all three currently available SmallThinker/SigLIP methods',
            'limitations':['Single frozen512 slice, one training seed; no cross-seed interval.',
                'Intervals are image-cluster resampling uncertainty, without multiple-configuration correction.',
                'R3 blank/text-only controls and full18-cell comparison remain pending.',
                'No mask-versus-R1 superiority or causal specialization claim.']}


def main():
    report=collect();directory=ROOT/'research_results/budget18_interim_grounding_v1';directory.mkdir(parents=True,exist_ok=True)
    _atomic_json(directory/'image_vs_shuffle.json',report)
    lines=['# Предварительный контроль зависимости от изображения\n\n',
        'SmallThinker + SigLIP2,512 вопросов, один seed42. Повторно проверены raw scores и provenance обеих полных оценок.\n\n',
        '| Вариант | Правильная картинка | Перемешанная | Разность | Парный95% CI |\n',
        '|---|---:|---:|---:|---|\n']
    names={'1b':'QLoRA + новые эксперты','4f':'Полное дообучение','1c':'QLoRA + эксперты + маски'}
    for row in report['rows']:
        scores=row['primary_scores'];paired=row['paired_image_cluster_bootstrap'];lo,hi=paired['paired_95_percentile_ci']
        lines.append(f"| {names[row['recipe']]} | {scores['none']:.2f} | {scores['shuffle']:.2f} | {paired['original_minus_variant_group_macro_points']:.2f} | [{lo:.2f}; {hi:.2f}] |\n")
    lines.append('\nЭто интервалы по image clusters данной выборки, без поправки на сравнение нескольких конфигураций и без оценки вариации между training seeds. R3 blank/text-only ещё не завершены; полный18-клеточный анализ ожидается. Превосходство масок над другими методами и причинная специализация экспертов этим не доказаны.\n')
    (directory/'image_vs_shuffle.md').write_text(''.join(lines))
    print(json.dumps([{k:r[k] for k in ('recipe','primary_scores','paired_image_cluster_bootstrap')} for r in report['rows']]))


if __name__=='__main__':main()
