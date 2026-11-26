#!/usr/bin/env python3
"""Offline CPU export/strict parity of Qwen2.5/Gemma4 vision-only donor assets."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import torch
from huggingface_hub import hf_hub_download
from src.utils.research_storage import _atomic_json
from src.utils.research_vision_export import (
    DONORS,export_vision_snapshot,file_sha,native_cpu_parity,plan_export,
)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--donor',choices=['all',*DONORS],default='all')
    parser.add_argument('--output-root',default=str(ROOT/'models/research-vision-derived'))
    parser.add_argument('--report',default=str(ROOT/'research_results/vision_only_derived_export.json'))
    parser.add_argument('--reserve-gb',type=float,default=120)
    args=parser.parse_args(argv)
    if args.reserve_gb<120:raise ValueError('The research disk reserve remains at least120GiB')
    if torch.cuda.is_initialized():raise RuntimeError('Vision export must not initialize CUDA')
    torch.set_num_threads(8);torch.set_num_interop_threads(1)
    inventory_path=ROOT/'research_results/model_inventory.json'
    inventory=json.loads(inventory_path.read_text())
    fixtures=[ROOT/'research_results/vision_probes/qwen25'/name for name in ('landscape.png','portrait.png')]
    if any(not path.is_file() for path in fixtures):raise FileNotFoundError('Existing two RGB fixture images are required')
    started=time.monotonic()
    def progress(event,**fields):
        print(json.dumps({'time':datetime.now(timezone.utc).isoformat(),'event':event,**fields}),flush=True)
    report={'schema_version':1,'status':'running','scope':'derived vision-only CPU export and native parity',
            'started_at':datetime.now(timezone.utc).isoformat(),'donors':[],
            'inventory_sha256':file_sha(inventory_path),'reserve_gb':args.reserve_gb,
            'source_hf_files_deleted':False,'production_resolver_modified':False,'cache_namespace_modified':False,
            'llm_instantiated':False,'weights_downloaded_bytes':0,'gpu_initialized':False}
    _atomic_json(Path(args.report),report)
    try:
        for donor_id in DONORS if args.donor=='all' else [args.donor]:
            item=next(row for row in inventory if row['repo']==DONORS[donor_id]['repo'])
            source=Path(hf_hub_download(item['repo'],'config.json',revision=item['sha'],local_files_only=True)).parent
            if source.name!=item['sha']:raise ValueError('Local source snapshot does not match pinned inventory commit')
            progress('planning_vision_export',donor=donor_id,source_repo=item['repo'],source_revision=item['sha'])
            plan=plan_export(source,donor_id,item['sha'],source_repo=item['repo'],progress=progress)
            progress('vision_export_preflight_passed',donor=donor_id,tensor_count=len(plan['tensor_headers']),
                     tensor_bytes=plan['tensor_bytes'],anticipated_new_peak_bytes=plan['anticipated_new_peak_bytes'])
            result=export_vision_snapshot(plan,args.output_root,reserve_gb=args.reserve_gb,progress=progress,
                validator=lambda source,derived,plan:native_cpu_parity(source,derived,plan,fixtures,progress))
            manifest=result.pop('manifest')
            original_weight_bytes=sum(plan['source_observations'][name]['bytes'] for name in plan['weight_files'])
            result.update(donor=donor_id,source_repo=item['repo'],source_revision=item['sha'],
                derivation_sha256=file_sha(Path(result['path'])/'derivation.json'),
                tensor_count=manifest['tensor_count'],tensor_elements=manifest['tensor_elements'],
                tensor_bytes=manifest['tensor_bytes'],artifact_total_bytes=manifest['artifact_total_bytes'],
                original_weight_files_bytes=original_weight_bytes,
                potential_net_reclaim_bytes=original_weight_bytes-manifest['artifact_total_bytes'],
                actual_reclaimed_bytes=0,native_validation=manifest['native_validation'],
                export_code_sha256=manifest['export_code_sha256'])
            report['donors'].append(result);_atomic_json(Path(args.report),report)
        if torch.cuda.is_initialized():raise RuntimeError('Vision export unexpectedly initialized CUDA')
        report.update(status='passed',elapsed_seconds=time.monotonic()-started,
            derived_total_bytes=sum(item['artifact_total_bytes'] for item in report['donors']),
            potential_net_reclaim_bytes=sum(item['potential_net_reclaim_bytes'] for item in report['donors']),
            limits='Original upstream weights remain intact. Reclaim is potential only, after a separate production derived-resolver/cache-identity review. Native CPU vision parity is not VLM accuracy or GPU validation.')
        _atomic_json(Path(args.report),report)
        progress('vision_exports_complete',report=args.report,total_derived_bytes=report['derived_total_bytes'],
                 potential_net_reclaim_bytes=report['potential_net_reclaim_bytes'],actual_reclaimed_bytes=0)
        return report
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error),elapsed_seconds=time.monotonic()-started)
        _atomic_json(Path(args.report),report);progress('vision_export_failed',error_type=type(error).__name__,error=str(error))
        raise


if __name__=='__main__':main()
