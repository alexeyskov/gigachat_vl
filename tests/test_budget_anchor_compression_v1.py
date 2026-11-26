import struct
import pytest
from scripts.probe_budget_anchor_compression_v1 import transform,inverse,measurements
def test_exact_all_FP32_bits_including_nan_payload_signedzero_infinity():
 raw=struct.pack('<8I',0,0x80000000,0x7fc00042,0xff800000,0x7f800000,1,0x3f800001,0xffffffff);anchor=struct.pack('<8I',0x3f800000,0,0x7fc00000,0,0,0,0,0)
 assert inverse(transform(raw))==raw and inverse(transform(raw,anchor),anchor)==raw
 assert all(x['exact_all_bits'] for x in measurements(raw,anchor))
def test_malformed_or_mismatched_byte_counts_refused():
 for raw,anchor in [(b'123',None),(b'1234',b'12')]:
  with pytest.raises(ValueError):transform(raw,anchor)
  with pytest.raises(ValueError):inverse(raw,anchor)
