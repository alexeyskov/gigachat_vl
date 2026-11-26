"""Deterministic image interventions for held-out grounding measurements.

Only image fields determine an intervention. Questions, answers, record IDs,
source references and scoring protocols are preserved. This module does not
encode images, load a language model, or choose data using labels/predictions.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

from PIL import Image,ImageOps
from src.dataset.vision_cache import image_content_hash,ensure_disk_space

VARIANTS=('none','shuffle','blank','text_only')
ABLATION_SCHEMA_VERSION=1


def canonical_manifest_bytes(records):
    return ''.join(json.dumps(row,sort_keys=True,separators=(',',':'),ensure_ascii=False)+'\n' for row in records).encode()


def _image_info(path,cache):
    key=str(Path(path).expanduser().resolve())
    if key not in cache:
        with Image.open(key) as raw:
            rgb=ImageOps.exif_transpose(raw).convert('RGB')
            cache[key]={'path':key,'hash':image_content_hash(rgb),'size':list(rgb.size)}
    return cache[key]


def _black_image(size,output_dir,reserve_bytes):
    width,height=size
    # Conservative raw-pixel + metadata allocation reservation, checked before
    # creating the image or writing its atomic temporary PNG.
    directory=Path(output_dir)/'ablation_images'
    if directory.is_symlink():raise ValueError('Ablation image directory may not be a symlink')
    ensure_disk_space(directory,width*height*3+65536,reserve_bytes)
    black=Image.new('RGB',(width,height),(0,0,0))
    content_hash=image_content_hash(black)
    directory.mkdir(parents=True,exist_ok=True)
    path=directory/f'black-rgb-{width}x{height}-{content_hash}.png'
    if path.exists() or path.is_symlink():
        if path.is_symlink():raise ValueError('Blank ablation image may not be a symlink')
        with Image.open(path) as existing:
            if existing.mode!='RGB' or existing.size!=(width,height) or image_content_hash(existing)!=content_hash:
                raise ValueError('Existing blank ablation asset has different pixels or dimensions')
        return str(path.resolve()),content_hash
    fd,temporary=tempfile.mkstemp(prefix='.blank-',suffix='.tmp',dir=directory)
    try:
        with os.fdopen(fd,'wb') as handle:
            black.save(handle,format='PNG')
            handle.flush();os.fsync(handle.fileno())
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)
    return str(path.resolve()),content_hash


def transform_image_ablation(records,*,variant='none',seed=42,output_dir=None,
                              source_manifest_sha256=None,reserve_bytes=120*1024**3):
    """Return immutable transformed rows and a reproducible protocol descriptor.

    Shuffle permutes unique canonical image hashes in one seeded cycle, rather
    than shuffling question rows (which could retain the same duplicate image).
    The current generation profile supports one image per record; shuffle/blank
    refuse multiple-image records explicitly. Text-only can remove any image list.
    """
    if variant not in VARIANTS:raise ValueError('Unknown image ablation variant')
    if not isinstance(seed,int) or isinstance(seed,bool):raise ValueError('Ablation seed must be an integer')
    if variant=='blank' and output_dir is None:raise ValueError('Blank ablation requires its own output directory')
    source_hash=source_manifest_sha256 or hashlib.sha256(canonical_manifest_bytes(records)).hexdigest()
    transformed=copy.deepcopy(records)
    if variant=='none':
        return transformed,{'schema_version':ABLATION_SCHEMA_VERSION,'variant':'none','seed':seed,
                            'source_manifest_sha256':source_hash,'transformed_manifest_sha256':source_hash,
                            'records':len(records),'transformation_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    cache={};by_hash={};images_by_record=[]
    for record in records:
        paths=record.get('image_paths',[])
        if not isinstance(paths,list) or not all(isinstance(path,str) for path in paths):raise ValueError('Ablation image_paths must be a list of paths')
        if variant in ('shuffle','blank') and len(paths)>1:raise NotImplementedError('Current shuffle/blank profile requires one image per record')
        infos=[_image_info(path,cache) for path in paths]
        hashes=record.get('image_hashes')
        if hashes is not None and (len(hashes)!=len(infos) or list(hashes)!=[row['hash'] for row in infos]):
            raise ValueError('Original image hash differs from the immutable manifest')
        images_by_record.append(infos)
        for info in infos:
            previous=by_hash.get(info['hash'])
            if previous is None or info['path']<previous['path']:by_hash[info['hash']]=info
    mapping={}
    if variant=='shuffle':
        unique=sorted(by_hash)
        if len(unique)<2:raise ValueError('Shuffle ablation needs at least two unique image hashes')
        random.Random(seed).shuffle(unique)
        mapping={value:unique[(i+1)%len(unique)] for i,value in enumerate(unique)}
        if any(a==b for a,b in mapping.items()):raise AssertionError('Image-group derangement retained an original image')
    blanks={}
    actual_hashes=set()
    for record,infos in zip(transformed,images_by_record):
        original=[info['hash'] for info in infos]
        paths=[];hashes=[];sizes=[]
        if variant=='shuffle':
            for info in infos:
                replacement=by_hash[mapping[info['hash']]]
                paths.append(replacement['path']);hashes.append(replacement['hash']);sizes.append(replacement['size'])
        elif variant=='blank':
            for info in infos:
                size=tuple(info['size'])
                if size not in blanks:blanks[size]=_black_image(size,output_dir,reserve_bytes)
                path,content_hash=blanks[size]
                paths.append(path);hashes.append(content_hash);sizes.append(list(size))
        record['image_paths']=paths;record['image_hashes']=hashes
        # Keep legacy visual aliases consistent; no stale hidden image survives
        # a text-only intervention if a future reader accesses these aliases.
        if 'image' in record:record['image']=paths[0] if len(paths)==1 else (paths if paths else None)
        if 'images' in record:record['images']=list(paths)
        original_phashes=record.pop('image_phashes',None)
        record['image_ablation']={'schema_version':ABLATION_SCHEMA_VERSION,'variant':variant,'seed':seed,
                                  'original_image_hashes':original,'actual_image_hashes':hashes,
                                  'original_rgb_dimensions':[info['size'] for info in infos],
                                  'actual_rgb_dimensions':sizes,
                                  'assignment':'unique_image_hash_cycle_derangement' if variant=='shuffle' else variant}
        if original_phashes is not None:record['image_ablation']['original_image_phashes']=original_phashes
        actual_hashes.update(hashes)
    body=canonical_manifest_bytes(transformed)
    protocol={'schema_version':ABLATION_SCHEMA_VERSION,'variant':variant,'seed':seed,
              'source_manifest_sha256':source_hash,'transformed_manifest_sha256':hashlib.sha256(body).hexdigest(),
              'records':len(records),'unique_original_images':len(by_hash),'unique_actual_images':len(actual_hashes),
              'transformation_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    return transformed,protocol


def write_transformed_manifest(records,path,*,reserve_bytes=120*1024**3):
    """Persist the exact transformed inputs, refusing incompatible existing data."""
    path=Path(path);data=canonical_manifest_bytes(records)
    digest=hashlib.sha256(data).hexdigest()
    if path.is_symlink():raise ValueError('Transformed ablation manifest may not be a symlink')
    if path.exists():
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError('Existing transformed ablation manifest differs; choose a separate output directory')
        return digest
    ensure_disk_space(path.parent,len(data)+65536,reserve_bytes)
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,temporary=tempfile.mkstemp(prefix='.ablation-manifest-',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as handle:
            handle.write(data);handle.flush();os.fsync(handle.fileno())
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)
    return digest
