"""Lossless streaming checkpoint I/O prototype, not enabled in active training.

Native files use the mmap-compatible Torch ZIP format. Compressed files stream
Torch's sequential legacy format through gzip (or optional zstandard), avoiding
an entire uncompressed staging checkpoint. No model/optimizer dtype changes are
performed. Loading compressed files explicitly disables mmap; compressed ZIP
members are also never mapped as raw tensor storage.

Only load trusted local research checkpoints: optimizer and Python/NumPy RNG
state require the ordinary pickle loader (weights_only=False). The caller must
preflight disk headroom including old and new retained checkpoint objects.
"""
from __future__ import annotations

import contextlib
import gzip
import io
import os
from pathlib import Path
import time
import uuid
import zipfile

GZIP_MAGIC = b'\x1f\x8b'
ZSTD_MAGIC = b'\x28\xb5\x2f\xfd'
ZIP_MAGIC = b'PK\x03\x04'


class CheckpointFormatError(ValueError):
    """Unrecognized or unsupported checkpoint compression/serialization format."""


def _zstandard():
    try:
        import zstandard
    except ImportError as error:
        raise RuntimeError("zstd checkpoint compression requires optional package 'zstandard'") from error
    return zstandard


class _SequentialWriteWrapper:
    """Hide a codec writer's raw fileno so Torch cannot bypass compression."""
    def __init__(self, stream): self.stream = stream
    def write(self, data): return self.stream.write(data)
    def flush(self): return self.stream.flush()
    def fileno(self): raise io.UnsupportedOperation('Compressed checkpoint has no directly writable raw descriptor')


class _PrefixReplayReader(io.RawIOBase):
    """A bounded initial-rewind adapter for sequential Torch legacy loading.

    Torch sniffs four bytes and rewinds before reading legacy records in order.
    zstd stream readers cannot rewind natively. Cache only the initial 64 KiB,
    never the whole decompressed checkpoint. Arbitrary backward seeks outside
    this prefix are explicitly refused instead of returning incorrect bytes.
    """
    def __init__(self, source, prefix_limit=65536):
        super().__init__()
        self.source = source
        self.prefix_limit = prefix_limit
        self.prefix = bytearray()
        self.position = 0
        self.source_position = 0

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position

    def seek(self, offset, whence=os.SEEK_SET):
        if whence == os.SEEK_CUR:
            offset += self.position
        elif whence != os.SEEK_SET:
            raise io.UnsupportedOperation('Sequential compressed checkpoint does not support end-relative seeking')
        if offset < 0:
            raise ValueError('Negative checkpoint stream position')
        if offset == self.position:
            return offset
        if offset <= len(self.prefix) and offset <= self.source_position:
            self.position = offset
            return offset
        if offset >= self.source_position:
            while self.position < offset:
                chunk = self.read(min(offset-self.position,1024*1024))
                if not chunk:
                    raise EOFError('Cannot seek beyond compressed checkpoint stream')
            return self.position
        raise io.UnsupportedOperation('Backward seek exceeds compressed checkpoint prefix buffer')

    def read(self, size=-1):
        if size == 0: return b''
        replay = b''
        if self.position < self.source_position:
            available = min(self.source_position,len(self.prefix))-self.position
            if available <= 0:
                raise io.UnsupportedOperation('Compressed checkpoint rewind exceeds prefix buffer')
            take = available if size < 0 else min(available,size)
            replay = bytes(self.prefix[self.position:self.position+take])
            self.position += take
            if size >= 0:
                size -= take
                if size == 0: return replay
            if self.position < self.source_position:
                raise io.UnsupportedOperation('Compressed checkpoint rewind exceeds prefix buffer')
        data = self.source.read(size)
        if data is None: data = b''
        if self.source_position < self.prefix_limit:
            self.prefix.extend(data[:self.prefix_limit-self.source_position])
        self.source_position += len(data)
        self.position += len(data)
        return replay+data

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)

    def readline(self, size=-1):
        result = bytearray()
        while size < 0 or len(result) < size:
            value = self.read(1)
            if not value: break
            result.extend(value)
            if value == b'\n': break
        return bytes(result)

    def close(self):
        if not self.closed:
            self.source.close()
        super().close()


def checkpoint_format(path):
    """Inspect headers/ZIP directory only; no tensor storage is read or mapped."""
    path = Path(path)
    with path.open('rb') as handle:
        prefix = handle.read(8)
    if prefix.startswith(GZIP_MAGIC):
        with gzip.open(path,'rb') as decoded:
            inner=decoded.read(4)
        if not (inner.startswith(ZIP_MAGIC) or inner.startswith(b'\x80')):
            raise CheckpointFormatError('gzip payload is not a Torch ZIP/legacy checkpoint')
        return {'compression':'gzip','serialization':'torch_zip' if inner.startswith(ZIP_MAGIC) else 'legacy_stream','mmap_supported':False}
    if prefix.startswith(ZSTD_MAGIC):
        return {'compression':'zstd','serialization':'legacy_stream','mmap_supported':False}
    if prefix.startswith(ZIP_MAGIC):
        with zipfile.ZipFile(path) as archive:
            codecs = {entry.compress_type for entry in archive.infolist()}
        if not codecs.issubset({zipfile.ZIP_STORED,zipfile.ZIP_DEFLATED}):
            raise CheckpointFormatError('Torch ZIP contains unsupported compression methods')
        compressed = zipfile.ZIP_DEFLATED in codecs
        return {'compression':'zip_deflate' if compressed else 'none','serialization':'torch_zip','mmap_supported':not compressed}
    if prefix.startswith(b'\x80'):
        return {'compression':'none','serialization':'legacy_stream','mmap_supported':False}
    raise CheckpointFormatError('Unrecognized checkpoint header; expected Torch ZIP/legacy, gzip or zstd')


def save_checkpoint(payload,path,compression='none',compression_level=None):
    """Atomically stream a lossless checkpoint to its final file.

    At most one new serialized file exists; there is no uncompressed disk
    staging for gzip/zstd. Native serialization is kept for compression='none'.
    Existing targets remain untouched if serialization/compression fails.
    Returns small timing/size metadata, never model tensors.
    """
    import torch
    if compression not in {'none','gzip','zstd'}:
        raise ValueError("compression must be 'none', 'gzip' or 'zstd'")
    if compression=='gzip' and compression_level is not None and not 0 <= compression_level <= 9:
        raise ValueError('gzip compression_level must be between 0 and 9')
    zstd = _zstandard() if compression=='zstd' else None
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    started = time.monotonic()
    try:
        with temporary.open('xb') as raw:
            if compression=='none':
                torch.save(payload,raw)
            elif compression=='gzip':
                # Suppress temporary-file names and timestamps in the outer
                # gzip header; neither belongs in checkpoint provenance.
                with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0,
                                   compresslevel=3 if compression_level is None else compression_level) as stream:
                    torch.save(payload,stream,_use_new_zipfile_serialization=False)
            else:
                compressor = zstd.ZstdCompressor(level=3 if compression_level is None else compression_level,
                                                 write_checksum=True)
                with compressor.stream_writer(raw,closefd=False) as stream:
                    torch.save(payload,_SequentialWriteWrapper(stream),_use_new_zipfile_serialization=False)
            raw.flush()
            os.fsync(raw.fileno())
        size = temporary.stat().st_size
        os.replace(temporary,path)
        directory_fd = os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try: os.fsync(directory_fd)
        finally: os.close(directory_fd)
        return {'path':str(path),'compression':compression,
                'serialization':'torch_zip' if compression=='none' else 'legacy_stream',
                'bytes':size,'seconds':time.monotonic()-started}
    finally:
        temporary.unlink(missing_ok=True)


def _validate_zstd_frame(path, zstd):
    """Check complete single-frame block boundaries without decompressing twice.

    zstandard's streaming reader can silently accept a missing final checksum.
    The frame layout permits a bounded header/seek pass that rejects truncated
    blocks/checksums before tensor reconstruction. The decoder still validates
    the actual frame checksum during its ordinary streaming read.
    """
    size = path.stat().st_size
    with path.open('rb') as raw:
        prefix = raw.read(18)
        header_size = zstd.frame_header_size(prefix)
        parameters = zstd.get_frame_parameters(prefix)
        raw.seek(header_size)
        while True:
            header = raw.read(3)
            if len(header) != 3:
                raise EOFError('Truncated zstd checkpoint block header')
            block = int.from_bytes(header,'little')
            last, kind, length = bool(block & 1), (block >> 1) & 3, block >> 3
            if kind == 3:
                raise CheckpointFormatError('Reserved zstd checkpoint block type')
            stored_length = 1 if kind == 1 else length
            if raw.tell()+stored_length > size:
                raise EOFError('Truncated zstd checkpoint block')
            raw.seek(stored_length,os.SEEK_CUR)
            if last: break
        if parameters.has_checksum and len(raw.read(4)) != 4:
            raise EOFError('Truncated zstd checkpoint checksum')
        if raw.tell() != size:
            raise CheckpointFormatError('Expected a single zstd checkpoint frame without trailing data')


def load_checkpoint(path,map_location='cpu',mmap=True):
    """Autodetect compression and load exact tensors without unsafe mmap use."""
    import torch
    path = Path(path)
    fmt = checkpoint_format(path)
    if fmt['compression'] in {'none','zip_deflate'}:
        return torch.load(path,map_location=map_location,weights_only=False,
                          mmap=bool(mmap and fmt['mmap_supported']))
    if fmt['compression']=='gzip':
        # GzipFile is seekable, so externally gzip-wrapped Torch ZIP files also
        # work (less efficiently than our sequential legacy stream format).
        with gzip.open(path,'rb') as stream:
            result = torch.load(stream,map_location=map_location,weights_only=False,mmap=False)
            # Force trailer/CRC validation even if torch stopped after its last
            # tensor before reading the complete outer compression stream.
            while stream.read(1024*1024): pass
            return result
    zstd = _zstandard()
    _validate_zstd_frame(path,zstd)
    with path.open('rb') as raw:
        with zstd.ZstdDecompressor().stream_reader(raw,closefd=False) as decompressed:
            with _PrefixReplayReader(decompressed) as stream:
                if stream.read(4)==ZIP_MAGIC:
                    raise CheckpointFormatError('zstd-wrapped Torch ZIP is not a sequential checkpoint; save with this helper')
                stream.seek(0)
                result = torch.load(stream,map_location=map_location,weights_only=False,mmap=False)
                while stream.read(1024*1024): pass
                return result
