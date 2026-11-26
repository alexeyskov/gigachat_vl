"""Lossless model, optimizer and RNG roundtrip; streaming and mmap safeguards."""
import gzip
import importlib.util
import io
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch
from torch import nn

from src.utils.research_checkpoint_io import (
    CheckpointFormatError, _PrefixReplayReader, checkpoint_format,
    load_checkpoint, save_checkpoint,
)


class TestResearchCheckpointIO(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()
    def payload(self):
        torch.manual_seed(7)
        model=nn.Linear(4,3)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
        loss=model(torch.randn(8,4)).square().sum();loss.backward();optimizer.step()
        base=torch.arange(24,dtype=torch.float32).reshape(4,6)
        special=torch.tensor([0.,-0.,float('inf'),float('-inf'),float('nan')])
        return {'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                'scheduler':{'last_epoch':3,'_last_lr':[.01]},'rng_torch':torch.get_rng_state(),
                'rng_python':random.getstate(),'rng_numpy':np.random.get_state(),
                'base':base,'view':base[:,1::2],'special':special,
                'bf16':torch.linspace(-1,1,32,dtype=torch.bfloat16),
                'metadata':{'step':3,'model_revision':'fixed','nested':[None,False,b'bytes']}}
    def assert_payload_equal(self,left,right):
        if isinstance(left,torch.Tensor):
            self.assertIsInstance(right,torch.Tensor)
            self.assertEqual(left.dtype,right.dtype)
            self.assertEqual(tuple(left.shape),tuple(right.shape))
            self.assertEqual(left.stride(),right.stride())
            # Compare raw representations, including signed zero and NaN bits.
            self.assertTrue(torch.equal(left.contiguous().reshape(-1).view(torch.uint8),right.contiguous().reshape(-1).view(torch.uint8)))
        elif isinstance(left,np.ndarray):np.testing.assert_array_equal(left,right)
        elif isinstance(left,dict):
            self.assertEqual(set(left),set(right))
            for key in left:self.assert_payload_equal(left[key],right[key])
        elif isinstance(left,(list,tuple)):
            self.assertEqual(type(left),type(right));self.assertEqual(len(left),len(right))
            for a,b in zip(left,right):self.assert_payload_equal(a,b)
        else:self.assertEqual(left,right)

    def test_native_and_gzip_preserve_all_model_optimizer_rng_bits_and_views(self):
        payload=self.payload()
        for codec in ('none','gzip'):
            with self.subTest(codec=codec):
                path=self.root/f'state-{codec}.pt'
                save_checkpoint(payload,path,codec)
                restored=load_checkpoint(path,map_location='cpu',mmap=True)
                self.assert_payload_equal(payload,restored)
                self.assertEqual(restored['base'].untyped_storage().data_ptr(),restored['view'].untyped_storage().data_ptr())
                self.assertEqual(checkpoint_format(path)['mmap_supported'],codec=='none')

    def test_gzip_saves_directly_to_one_compressed_stream_without_raw_staging(self):
        path=self.root/'state.pt.gz';original=torch.save;observed=[]
        def saving(payload,stream,**kwargs):
            observed.append((isinstance(stream,gzip.GzipFile),kwargs.get('_use_new_zipfile_serialization'),len(list(self.root.iterdir()))))
            return original(payload,stream,**kwargs)
        with patch.object(torch,'save',side_effect=saving):save_checkpoint(self.payload(),path,'gzip')
        self.assertEqual(observed,[(True,False,1)])
        self.assertEqual(list(self.root.iterdir()),[path])
        self.assertEqual(path.read_bytes()[:2],b'\x1f\x8b')

    def test_compressed_load_explicitly_disables_mmap(self):
        path=self.root/'state.gz';save_checkpoint(self.payload(),path,'gzip')
        original=torch.load;seen=[]
        def loading(*args,**kwargs):
            seen.append(kwargs.get('mmap'));return original(*args,**kwargs)
        with patch.object(torch,'load',side_effect=loading):load_checkpoint(path,mmap=True)
        self.assertEqual(seen,[False])

    def test_deflated_torch_zip_never_maps_compressed_storage(self):
        raw=self.root/'raw.pt';compressed=self.root/'deflated.pt';payload=self.payload()
        save_checkpoint(payload,raw)
        with zipfile.ZipFile(raw) as source,zipfile.ZipFile(compressed,'w',compression=zipfile.ZIP_DEFLATED) as destination:
            for info in source.infolist():destination.writestr(info.filename,source.read(info.filename))
        self.assertEqual(checkpoint_format(compressed)['compression'],'zip_deflate')
        original=torch.load;seen=[]
        def loading(*args,**kwargs):seen.append(kwargs.get('mmap'));return original(*args,**kwargs)
        with patch.object(torch,'load',side_effect=loading):restored=load_checkpoint(compressed,mmap=True)
        self.assertEqual(seen,[False]);self.assert_payload_equal(payload,restored)

    def test_external_gzip_wrapped_native_torch_zip_is_auto_decoded(self):
        raw=self.root/'native.pt';compressed=self.root/'external.pt.gz';payload=self.payload()
        save_checkpoint(payload,raw)
        with gzip.open(compressed,'wb') as handle:handle.write(raw.read_bytes())
        fmt=checkpoint_format(compressed)
        self.assertEqual(fmt['serialization'],'torch_zip')
        self.assertFalse(fmt['mmap_supported'])
        self.assert_payload_equal(payload,load_checkpoint(compressed,mmap=True))

    def test_atomic_serialization_failure_leaves_existing_target_unchanged(self):
        path=self.root/'checkpoint.pt';path.write_bytes(b'old-checkpoint')
        def failing(payload,stream,**kwargs):
            stream.write(b'partial-data');raise RuntimeError('simulated serializer failure')
        with patch.object(torch,'save',side_effect=failing):
            with self.assertRaisesRegex(RuntimeError,'simulated'):save_checkpoint({},path,'gzip')
        self.assertEqual(path.read_bytes(),b'old-checkpoint')
        self.assertEqual(list(self.root.iterdir()),[path])

    def test_unknown_corrupted_and_truncated_gzip_refused(self):
        unknown=self.root/'unknown';unknown.write_bytes(b'not a checkpoint')
        with self.assertRaises(CheckpointFormatError):load_checkpoint(unknown)
        good=self.root/'good.gz';save_checkpoint(self.payload(),good,'gzip')
        truncated=self.root/'truncated.gz';truncated.write_bytes(good.read_bytes()[:-4])
        with self.assertRaises((EOFError,OSError,RuntimeError)):load_checkpoint(truncated)
        corrupt=self.root/'corrupted.gz';data=bytearray(good.read_bytes());data[-8]^=1;corrupt.write_bytes(data)
        with self.assertRaises((EOFError,OSError,RuntimeError)):load_checkpoint(corrupt)

    def test_raw_legacy_checkpoint_disables_mmap_and_loads(self):
        path=self.root/'legacy.pt';payload=self.payload()
        torch.save(payload,path,_use_new_zipfile_serialization=False)
        self.assertFalse(checkpoint_format(path)['mmap_supported'])
        self.assert_payload_equal(payload,load_checkpoint(path,mmap=True))

    def test_prefix_replay_reader_supports_torch_legacy_without_full_buffering(self):
        buffer=io.BytesIO();payload=self.payload()
        torch.save(payload,buffer,_use_new_zipfile_serialization=False)
        class ForwardOnly:
            def __init__(self,data):self.data=io.BytesIO(data)
            def read(self,size=-1):return self.data.read(size)
            def close(self):self.data.close()
        reader=_PrefixReplayReader(ForwardOnly(buffer.getvalue()),prefix_limit=64)
        restored=torch.load(reader,map_location='cpu',weights_only=False,mmap=False)
        self.assert_payload_equal(payload,restored)
        self.assertLessEqual(len(reader.prefix),64)
        reader.seek(0)
        with self.assertRaises(io.UnsupportedOperation):reader.read(128)
        reader.close()

    def test_zstd_missing_dependency_is_explicit(self):
        with patch('src.utils.research_checkpoint_io._zstandard',side_effect=RuntimeError('optional zstandard missing')):
            with self.assertRaisesRegex(RuntimeError,'zstandard'):save_checkpoint({},self.root/'state.zst','zstd')
        self.assertEqual(list(self.root.iterdir()),[])

    @unittest.skipUnless(importlib.util.find_spec('zstandard'),'optional zstandard unavailable')
    def test_optional_zstd_lossless_roundtrip_and_truncation(self):
        path=self.root/'state.zst';payload=self.payload()
        save_checkpoint(payload,path,'zstd');self.assert_payload_equal(payload,load_checkpoint(path,mmap=True))
        self.assertFalse(checkpoint_format(path)['mmap_supported'])
        truncated=self.root/'truncated.zst';truncated.write_bytes(path.read_bytes()[:-3])
        with self.assertRaises((EOFError,OSError,RuntimeError,ValueError)):load_checkpoint(truncated)


if __name__=='__main__':unittest.main()
