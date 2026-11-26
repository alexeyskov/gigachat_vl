#!/usr/bin/env python3
"""Prepare pinned CPU-only CIDEr/PTB assets without changing the training env."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path('/workspace/.research_eval_assets/caption_v1')
VENDOR = ROOT / 'third_party/research_caption'
CONFIG = ROOT / 'configs/research/caption_metric_assets.json'
ORIGINAL = '3a9afb2682141a03e1cdc02b0df6770d2c884f6f'
PORT = 'a24f74c408c918f1f4ec34e9514bc8a76ce41ffd'
JRE_URL = ('https://github.com/adoptium/temurin8-binaries/releases/download/jdk8u514-b01/'
           'OpenJDK8U-jre_x64_linux_hotspot_8u514b01.tar.gz')
JRE_SHA = 'b3e38dc60261a548fe83d297bf6faf7880f1b84f7d48e5741f3c6f3c3021cd4d'
JRE_BYTES = 41870390
RESERVE = 120 * 2**30


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(2**20), b''): h.update(part)
    return h.hexdigest()


def fetch(url, limit):
    request = urllib.request.Request(url, headers={'User-Agent': 'research-caption-assets-v1'})
    with urllib.request.urlopen(request, timeout=45) as response:
        data = response.read(limit + 1)
    if len(data) > limit: raise ValueError('Asset exceeds declared download bound')
    return data


def publish(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data: raise ValueError('Existing pinned asset differs: ' + str(path))
        return
    fd, temporary = tempfile.mkstemp(prefix='.'+path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def main():
    if shutil.disk_usage(ROOT).free < RESERVE + 300 * 2**20:
        raise RuntimeError('Preserve 120 GiB disk reserve plus bounded asset staging')
    CACHE.mkdir(parents=True, exist_ok=True); VENDOR.mkdir(parents=True, exist_ok=True)
    sources = {}
    files = {
        'original_cider_scorer_py2.py': ('tylin/coco-caption', ORIGINAL, 'pycocoevalcap/cider/cider_scorer.py'),
        'original_ptbtokenizer_py2.py': ('tylin/coco-caption', ORIGINAL, 'pycocoevalcap/tokenizer/ptbtokenizer.py'),
        'cider_scorer.py': ('salaniz/pycocoevalcap', PORT, 'cider/cider_scorer.py'),
        'ptbtokenizer_reference.py': ('salaniz/pycocoevalcap', PORT, 'tokenizer/ptbtokenizer.py'),
        'LICENSE': ('salaniz/pycocoevalcap', PORT, 'license.txt'),
    }
    downloaded = 0
    for name, (repo, revision, relative) in files.items():
        url = f'https://raw.githubusercontent.com/{repo}/{revision}/{relative}'
        data = fetch(url, 256 * 2**10); downloaded += len(data)
        publish(VENDOR/name, data)
        sources[name] = {'repo': repo, 'revision': revision, 'path': relative,
                         'url': url, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    original = (VENDOR/'original_cider_scorer_py2.py').read_bytes()
    expected_port = original.replace(b'xrange(', b'range(').replace(b'.iteritems()', b'.items()')
    if expected_port.rstrip(b'\r\n') != (VENDOR/'cider_scorer.py').read_bytes().rstrip(b'\r\n'):
        raise ValueError('Pinned Python3 scorer is not the exact declared Python2 syntax port')
    jar_name = 'stanford-corenlp-3.4.1.jar'
    jar_api = ('https://api.github.com/repos/tylin/coco-caption/contents/'
               f'pycocoevalcap/tokenizer/{jar_name}?ref={ORIGINAL}')
    info = json.loads(fetch(jar_api, 256 * 2**10))
    if not 0 < info['size'] <= 10 * 2**20: raise ValueError('Unexpected pinned PTB jar size')
    jar = CACHE/jar_name
    data = jar.read_bytes() if jar.exists() else fetch(info['download_url'], info['size'])
    git_blob = hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
    if len(data) != info['size'] or git_blob != info['sha'] or not data.startswith(b'PK'):
        raise ValueError('PTB jar differs from immutable upstream Git blob')
    if not jar.exists(): downloaded += len(data)
    publish(jar, data)
    jar_meta = {'path': str(jar), 'sha256': sha(jar), 'bytes': jar.stat().st_size,
                'url': info['download_url'], 'git_blob_sha1': git_blob, 'upstream_revision': ORIGINAL}
    archive = CACHE/'OpenJDK8U-jre_x64_linux_hotspot_8u514b01.tar.gz'
    if not archive.exists():
        data = fetch(JRE_URL, JRE_BYTES)
        if len(data) != JRE_BYTES or hashlib.sha256(data).hexdigest() != JRE_SHA:
            raise ValueError('Pinned JRE release checksum/size differs')
        publish(archive, data); downloaded += len(data)
    if archive.stat().st_size != JRE_BYTES or sha(archive) != JRE_SHA:
        raise ValueError('Existing pinned JRE archive changed')
    installation = CACHE/'temurin8u514-b01-jre'
    if not installation.exists():
        temporary = Path(tempfile.mkdtemp(prefix='.jre-', dir=CACHE))
        try:
            with tarfile.open(archive) as tar:
                members = tar.getmembers()
                if len(members) > 3000 or sum(m.size for m in members) > 200*2**20:
                    raise ValueError('JRE extraction bound exceeded')
                tar.extractall(temporary, filter='data')
            children = list(temporary.iterdir())
            if len(children) != 1 or not (children[0]/'bin/java').is_file():
                raise ValueError('Unexpected JRE directory layout')
            os.rename(children[0], installation)
        finally:
            shutil.rmtree(temporary)
    java = installation/'bin/java'
    libjvm = installation/'lib/amd64/server/libjvm.so'
    version = subprocess.run([str(java), '-version'], capture_output=True, text=True, timeout=20, check=True)
    if '1.8.0_514' not in version.stderr: raise ValueError('Runtime is not the pinned JRE8 release')
    config = {'schema_version': 1, 'scope': 'CPU CIDEr and canonical Stanford PTB tokenizer only',
        'training_environment_modified': False, 'original_revision': ORIGINAL, 'port_revision': PORT,
        'scorer_path': str(VENDOR/'cider_scorer.py'), 'scorer_sha256': sources['cider_scorer.py']['sha256'],
        'ptb_jar': jar_meta, 'java': {'path': str(java), 'sha256': sha(java),
            'libjvm_path': str(libjvm), 'libjvm_sha256': sha(libjvm), 'version_stderr': version.stderr,
            'release_archive_sha256': JRE_SHA, 'release_archive_bytes': JRE_BYTES, 'release_url': JRE_URL},
        'sources': sources, 'python2_port_changes': ['xrange to range', 'iteritems to items', 'terminal newline'],
        'algorithm': {'name': 'COCO-caption CIDEr', 'ngrams': 4, 'sigma': 6.,
            'clipping': True, 'gaussian_length_penalty': True, 'raw_scale': '0-10',
            'document_frequency': 'fixed full evaluation reference-image corpus; per-image union of reference ngrams'},
        'tokenizer': {'name': 'Stanford PTBTokenizer CoreNLP3.4.1',
            'flags': ['-preserveLines', '-lowerCase'], 'encoding': 'UTF-8'},
        'declared_max_asset_staging_bytes': 300*2**20}
    publish(CONFIG, (json.dumps(config, sort_keys=True, indent=2)+'\n').encode())
    publish(VENDOR/'provenance.json', (json.dumps(config['sources'], sort_keys=True, indent=2)+'\n').encode())
    print(json.dumps({'status': 'prepared', 'config': str(CONFIG), 'downloaded_bytes_this_call': downloaded,
        'free_gib': shutil.disk_usage(ROOT).free/2**30, 'cuda_used': False, 'global_java_installed': False}))


if __name__ == '__main__': main()
