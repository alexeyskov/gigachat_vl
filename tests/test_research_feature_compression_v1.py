"""Lossless BF16 bit transforms, independent of floating-point comparisons."""
import json
from pathlib import Path
import numpy as np
import pytest
from scripts import probe_research_feature_compression_v1 as p

@pytest.mark.parametrize('kind',['identity','byte_shuffle','bit_shuffle','xor_previous_token_byte_shuffle'])
@pytest.mark.parametrize('shape',[(1,1),(3,5),(7,18)])
def test_arbitrary_BF16_bits_including_padding_roundtrip(kind,shape):
 a=np.random.default_rng(42).integers(0,65536,size=shape,dtype=np.uint16);raw=a.astype('<u2',copy=False).tobytes();encoded=p.transform(raw,shape,kind)
 assert p.transform(encoded,shape,kind,True)==raw

def test_signed_zero_inf_NaN_payloads_remain_exact_bits():
 raw=np.array([0x0000,0x8000,0x7f80,0xff80,0x7fc1,0x7fff,0x0001,0xffff],dtype='<u2').tobytes()
 for kind in ('identity','byte_shuffle','bit_shuffle','xor_previous_token_byte_shuffle'):
  assert p.transform(p.transform(raw,(2,4),kind),(2,4),kind,True)==raw

def test_actual_bounded_sample_source_bits_and_no_active_adoption():
 r=json.loads((p.ROOT/'research_results/feature_lossless_compression_sample_v1.json').read_bytes())
 assert r['sample_file_bytes_read']<=64*2**20 and r['sample_records']==32 and not r['new_format_adopted'] and not r['active_cache_or_decoder_or_scientific_files_changed']
 assert r['source_code_sha256']['scripts/probe_research_feature_compression_v1.py']==p.sha(Path(p.__file__).read_bytes())
 assert all(tuple(row['shape'])==(729,1152) and row['storage_dtype']=='torch.bfloat16' and all(v['all_original_bytes_or_BF16bits_exact'] and v['metadata_shape_dtype_unchanged'] for v in row['algorithms'].values()) for row in r['rows'])
