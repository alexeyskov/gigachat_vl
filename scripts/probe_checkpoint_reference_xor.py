#!/usr/bin/env python3
"""Bounded lossless reference-XOR/byte-shuffle measurement, not a checkpoint format.

Uses the exact tensor prefixes selected by checkpoint_compression_probe.json.
Unmatched/new expert/router tensors are excluded explicitly, never approximated.
No active writer, reader, scientific checkpoint or environment is modified.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def xor_shuffle(tensor,reference):
    """Serialize exact fp32 XOR bits as four consecutive byte lanes."""
    import torch
    if tensor.dtype!=torch.float32 or reference.dtype!=torch.bfloat16:
        raise ValueError('Probe requires trained fp32 and pinned original bf16 tensors')
    if tensor.shape!=reference.shape:
        raise ValueError('Reference sample shape mismatch')
    values=tensor.detach().cpu().contiguous().reshape(-1)
    anchor=reference.detach().cpu().float().contiguous().reshape(-1)
    delta=torch.bitwise_xor(values.view(torch.int32),anchor.view(torch.int32))
    return delta.view(torch.uint8).reshape(-1,4).t().contiguous().numpy().tobytes()


def decode_xor_shuffle(data,reference):
    """Inverse byte shuffle and XOR; no arithmetic subtraction or quantization."""
    import torch
    if reference.dtype!=torch.bfloat16 or len(data)!=reference.numel()*4:
        raise ValueError('Reference dtype/byte length mismatch')
    lanes=torch.frombuffer(bytearray(data),dtype=torch.uint8).reshape(4,-1)
    delta=lanes.t().contiguous().view(torch.int32).reshape(-1)
    original=reference.detach().cpu().float().contiguous().reshape(-1).view(torch.int32)
    return torch.bitwise_xor(delta,original).view(torch.float32).reshape(reference.shape)


def run_probe(sample_report,output_report,max_sample_bytes=128*1024**2):
    import torch
    from safetensors import safe_open
    from huggingface_hub import snapshot_download
    from src.utils.research_storage import DiskBudget
    if not 0<max_sample_bytes<=1024**3:
        raise ValueError('Probe must remain bounded to at most 1 GiB')
    DiskBudget(ROOT,120).ensure_space(1024**3)
    torch.set_num_threads(4)
    report=json.loads(Path(sample_report).read_text())
    source=Path(report['source_checkpoint'])
    if source.stat().st_size!=report['source_size_bytes']:
        raise ValueError('Source checkpoint changed since original sample probe')
    inventory=json.loads((ROOT/'research_results/model_inventory.json').read_text())
    anchor=next(row for row in inventory if row['repo']=='Tiiny/SmallThinker-4BA0.6B-Instruct')
    revision=anchor['sha']
    snapshot=Path(snapshot_download(anchor['repo'],revision=revision,local_files_only=True,allow_patterns=['model*.safetensors','model.safetensors.index.json']))
    index=json.loads((snapshot/'model.safetensors.index.json').read_text())
    checkpoint=torch.load(source,map_location='cpu',mmap=True,weights_only=False)
    matched=[];novel=[];blocks=[];raw_parts=[];used=0
    started=time.monotonic()
    with ExitStack() as handles:
        opened={}
        for selected in report['slices']:
            name=selected['name'];current=checkpoint['model'][name]
            reference_name=name.removeprefix('llm.')
            shard=index['weight_map'].get(reference_name)
            if shard is None:
                novel.append({'name':name,'reason':'no_original_parameter_key'});continue
            if shard not in opened:
                opened[shard]=handles.enter_context(safe_open(snapshot/shard,framework='pt',device='cpu'))
            reference=opened[shard].get_tensor(reference_name)
            if tuple(reference.shape)!=tuple(current.shape) or tuple(current.shape)!=tuple(selected['original_shape']):
                novel.append({'name':name,'reason':'full_parameter_shape_mismatch','current_shape':list(current.shape),'reference_shape':list(reference.shape)});continue
            if current.dtype!=torch.float32 or reference.dtype!=torch.bfloat16:
                novel.append({'name':name,'reason':'unsupported_sample_dtype','current_dtype':str(current.dtype),'reference_dtype':str(reference.dtype)});continue
            elements=selected['sample_elements'];size=elements*4
            if used+size>max_sample_bytes:
                raise ValueError('Original sample would exceed explicitly bounded probe budget')
            trained=current.reshape(-1)[:elements]
            original=reference.reshape(-1)[:elements]
            if trained.numel()!=elements or original.numel()!=elements:
                raise ValueError('Sample element count exceeds matched full tensor')
            block=xor_shuffle(trained,original)
            blocks.append(block);raw_parts.append((trained.clone(),original.clone()))
            matched.append({'name':name,'reference_name':reference_name,'shard':shard,'full_shape':list(current.shape),
                            'sample_elements':elements,'offset_bytes':used,'bytes':size})
            used+=size
        if not matched:raise ValueError('No original bf16 parameters matched the selected trained tensors')
        transform_seconds=time.monotonic()-started
        encoded=b''.join(blocks)
        del blocks
        measurements=[]
        codecs=['gzip']
        try:
            import zstandard
            codecs.append('zstd')
        except ImportError:zstandard=None
        for codec in codecs:
            start=time.monotonic()
            compressed=gzip.compress(encoded,compresslevel=3,mtime=0) if codec=='gzip' else zstandard.ZstdCompressor(level=3,write_checksum=True).compress(encoded)
            compression_seconds=time.monotonic()-start
            start=time.monotonic()
            decoded=gzip.decompress(compressed) if codec=='gzip' else zstandard.ZstdDecompressor().decompress(compressed)
            decompression_seconds=time.monotonic()-start
            if decoded!=encoded:raise AssertionError('Codec changed XOR delta bytes')
            start=time.monotonic()
            for row,(trained,original) in zip(matched,raw_parts):
                offset=row['offset_bytes'];restored=decode_xor_shuffle(decoded[offset:offset+row['bytes']],original)
                if not torch.equal(trained.contiguous().view(torch.uint8),restored.contiguous().view(torch.uint8)):
                    raise AssertionError(f"Reconstructed tensor bits changed: {row['name']}")
            decode_verify_seconds=time.monotonic()-start
            measurements.append({'compression':codec,'raw_matched_fp32_bytes':used,'encoded_xor_shuffled_bytes':len(encoded),
                                 'compressed_bytes':len(compressed),'compressed_to_raw_matched_ratio':len(compressed)/used,
                                 'compression_seconds':compression_seconds,'decompression_seconds':decompression_seconds,
                                 'xor_unshuffle_and_exact_verification_seconds':decode_verify_seconds,
                                 'compression_mib_per_second':used/1024**2/compression_seconds,
                                 'decompression_mib_per_second':used/1024**2/decompression_seconds,'exact_all_matched_tensor_bits':True})
            del compressed,decoded
        result={'schema_version':1,'status':'measured_sample_only','source_checkpoint':str(source),
                'source_checkpoint_size_bytes':source.stat().st_size,'source_sample_report':str(Path(sample_report)),
                'anchor_repo':anchor['repo'],'anchor_revision_sha':revision,
                'anchor_index_sha256':hashlib.sha256((snapshot/'model.safetensors.index.json').read_bytes()).hexdigest(),
                'reference_conversion':'original bf16 exactly promoted to fp32; uint32 bit XOR; four byte-lane shuffle',
                'endianness':sys.byteorder,'selected_sample_count':len(report['slices']),'matched_count':len(matched),
                'novel_or_unmatched_count':len(novel),'matched_sample_bytes':used,'transform_seconds':transform_seconds,
                'source_optimizer_included':False,'zstd_available_in_probe_process':zstandard is not None,
                'measurements':measurements,'matched':matched,'excluded':novel,
                'caveats':['Sample only: <=128MiB prefixes from completed two-step technical smoke; no full-checkpoint size/fit guarantee.',
                           'New experts, new router/adapter tensors and shape changes have no old reference and were excluded, not discarded from any checkpoint.',
                           'A real format must retain novel tensors, optimizer state, RNG, metadata and the exact pinned base anchor.',
                           'This probe does not measure trained Adam-state compressibility or convergence-stage weights.',
                           'No active format integration, no quantization, no dtype change of reconstructed trained weights.',
                           'All396 retained final checkpoints still require storage far beyond500GiB; the experiment matrix remains incomplete.']}
        Path(output_report).write_text(json.dumps(result,indent=2)+'\n')
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample-report',default=str(ROOT/'research_results/checkpoint_compression_probe.json'))
    parser.add_argument('--output-report',default=str(ROOT/'research_results/checkpoint_reference_xor_probe.json'))
    args=parser.parse_args()
    result=run_probe(args.sample_report,args.output_report)
    print(json.dumps({key:result[key] for key in ('status','anchor_revision_sha','matched_count','novel_or_unmatched_count','matched_sample_bytes','measurements')}))


if __name__=='__main__':main()
