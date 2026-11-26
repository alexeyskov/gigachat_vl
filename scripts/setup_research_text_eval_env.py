#!/usr/bin/env python3
"""Bounded public downloads and a separate pinned CPU IFEval environment."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request
import zipfile

ROOT=Path(__file__).resolve().parents[1]
CAP=200*2**20
RESERVE=120*2**30
BASE=ROOT/'research_results/text_scoring_environment'
VENDOR=ROOT/'third_party/ifeval_google_87e25a5'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def save(path,value): Path(path).write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')


def assert_window():
    for p in (ROOT/'research_results').glob('parallel_b1_full_boundary_*_w[24].json'):
        if read(p).get('status')=='running_full512':
            raise RuntimeError('DEFERRED: exclusive full512 proof is running; no heavy setup/test work')


def installed():
    return sorted((d.metadata['Name'],d.version) for d in importlib.metadata.distributions())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--venv',default='/tmp/research_text_eval_venv')
    args=parser.parse_args()
    envpath=Path(args.venv).resolve()
    if envpath==Path(sys.prefix).resolve() or envpath.is_relative_to('/venv/main'):
        raise ValueError('Must use a separate auxiliary environment')
    assert_window(); BASE.mkdir(parents=True,exist_ok=True)
    before=installed()
    observation=read(BASE/'official_pypi_observation.json')
    provenance=read(VENDOR/'UPSTREAM_PROVENANCE.json')
    budget=sum(x['metadata_bytes'] for x in observation['packages'].values())+sum(x['bytes'] for x in provenance['files'])
    # Conservative allowance covers pinned release metadata and GitHub JSON,
    # including repeat resolution. Download artifacts are counted separately.
    metadata_allowance=16*2**20
    # First actual setup hit NLTK3.10's directory-permission check after
    # downloading. Preserve a conservative bound for its repeated archive
    # transfer while rebinding the pinned data revision; nothing is hidden.
    retry_transfer_allowance=32*2**20
    artifacts=BASE/'artifacts';artifacts.mkdir(exist_ok=True)
    downloads=[]

    def get(url,target=None,expected=None):
        nonlocal budget
        assert_window()
        if target and target.exists():
            if expected and sha(target)!=expected: raise ValueError('Existing public artifact hash differs')
            receipt=target.with_suffix(target.suffix+'.download.json')
            if expected or (receipt.is_file() and read(receipt)=={'url':url,'sha256':sha(target)}):
                return target.read_bytes()
        with urllib.request.urlopen(url,timeout=45) as response:
            chunks=[];total=0
            while chunk:=response.read(256*1024):
                assert_window();total+=len(chunk)
                if budget+metadata_allowance+retry_transfer_allowance+total>CAP:raise RuntimeError('Additional200MiB download cap would be exceeded')
                if shutil.disk_usage(BASE).free-total<RESERVE:raise RuntimeError('Disk reserve would be exceeded')
                chunks.append(chunk)
        raw=b''.join(chunks);budget+=len(raw)
        if expected and hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('Public source digest differs')
        if target:
            if target.exists() and target.read_bytes()!=raw:raise ValueError('Existing unbound resource differs from pinned download')
            target.write_bytes(raw)
            save(target.with_suffix(target.suffix+'.download.json'),{'url':url,'sha256':hashlib.sha256(raw).hexdigest()})
        return raw

    from pip._vendor.packaging.tags import sys_tags
    from pip._vendor.packaging.utils import parse_wheel_filename
    priorities={tag:i for i,tag in enumerate(sys_tags())}
    wheels=[];sdists=[]
    for package,version in sorted(observation['pins'].items()):
        url=f'https://pypi.org/pypi/{package}/{version}/json'
        raw=get(url);info=json.loads(raw);choices=[]
        for file in info['urls']:
            if file['packagetype']=='bdist_wheel':
                tags=parse_wheel_filename(file['filename'])[3]
                if set(tags)&priorities.keys(): choices.append((min(priorities[t] for t in tags if t in priorities),file))
        chosen=min(choices,key=lambda x:x[0])[1] if choices else next(x for x in info['urls'] if x['packagetype']=='sdist')
        target=artifacts/chosen['filename'];get(chosen['url'],target,chosen['digests']['sha256'])
        if chosen['packagetype']=='bdist_wheel':wheels.append(target)
        else:sdists.append(target)
        downloads.append({'kind':'pypi','name':package,'version':version,'url':chosen['url'],
            'path':str(target),'bytes':target.stat().st_size,'sha256':sha(target),'release_metadata_sha256':hashlib.sha256(raw).hexdigest()})
    # Pin the exact public NLTK data revision before fetching its archives.
    revision_file=BASE/'nltk_data_revision.json'
    if revision_file.is_file():revision=read(revision_file)['sha']
    else:
        head=json.loads(get('https://api.github.com/repos/nltk/nltk_data/commits/gh-pages'))
        revision=head['sha'];save(revision_file,{'sha':revision,'source':'https://api.github.com/repos/nltk/nltk_data/commits/gh-pages'})
    resources=BASE/'nltk_data';resources.mkdir(exist_ok=True)
    for name in ('punkt','punkt_tab'):
        url=f'https://raw.githubusercontent.com/nltk/nltk_data/{revision}/packages/tokenizers/{name}.zip'
        target=artifacts/(name+'.zip');get(url,target)
        archive=zipfile.ZipFile(target)
        for member in archive.infolist():
            dest=(resources/'tokenizers'/member.filename).resolve()
            if not dest.is_relative_to(resources.resolve()) or (member.external_attr>>16)&0o170000==0o120000:
                raise ValueError('Unsafe NLTK archive member')
        assert_window();archive.extractall(resources/'tokenizers')
        downloads.append({'kind':'nltk_resource','name':name,'revision':revision,'url':url,
            'path':str(target),'bytes':target.stat().st_size,'sha256':sha(target)})
    # NLTK3.10 rejects data roots writable by another account. Change only this
    # newly created resource tree, retaining exact upstream content bytes.
    resources.chmod(0o700)
    for path in resources.rglob('*'):
        if path.is_symlink():raise ValueError('NLTK resource tree may not contain symlinks')
        path.chmod(0o700 if path.is_dir() else 0o600)
    total_artifacts=sum(d['bytes'] for d in downloads)
    initial_metadata=sum(x['metadata_bytes'] for x in observation['packages'].values())
    vendor_bytes=sum(x['bytes'] for x in provenance['files'])
    if total_artifacts+initial_metadata+vendor_bytes+metadata_allowance+retry_transfer_allowance>CAP:
        raise RuntimeError('Actual artifacts and metadata exceed additional200MiB cap')
    assert_window()
    if not (envpath/'bin/python').is_file():
        subprocess.run(['/usr/bin/python3','-m','venv',str(envpath)],check=True)
    python=str(envpath/'bin/python')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NLTK_DATA=str(resources))
    with (BASE/'install.log').open('a') as log:
        assert_window()
        subprocess.run([python,'-m','pip','install','--no-index','--no-deps','--no-cache-dir',*[str(p) for p in wheels]],
            check=True,env=env,stdout=log,stderr=subprocess.STDOUT)
        if sdists:
            assert_window()
            subprocess.run([python,'-m','pip','install','--no-index','--no-deps','--no-cache-dir','--no-build-isolation',*[str(p) for p in sdists]],
                check=True,env=env,stdout=log,stderr=subprocess.STDOUT)
        assert_window()
        subprocess.run([python,'-m','pip','check'],check=True,env=env,stdout=log,stderr=subprocess.STDOUT)
    probe="import importlib.metadata,json,sys,nltk; nltk.data.path[:]=[sys.argv[1]]; assert len(nltk.sent_tokenize('One sentence. Another sentence.'))==2; assert len(nltk.word_tokenize('One sentence.'))==3; print(json.dumps({d.metadata['Name']:d.version for d in importlib.metadata.distributions()},sort_keys=True))"
    assert_window()
    versions=json.loads(subprocess.check_output([python,'-c',probe,str(resources)],env=env,text=True))
    for name,version in observation['pins'].items():
        normalized={k.lower().replace('_','-'):v for k,v in versions.items()}
        if normalized.get(name)!=version:raise ValueError('Auxiliary installed package differs from observed pin')
    after=installed()
    if before!=after:raise RuntimeError('Main environment packages changed')
    file_probe="""import hashlib,importlib.metadata,json,pathlib,sys
root=pathlib.Path(sys.prefix).resolve(); result={}
for name in json.loads(sys.argv[1]):
 for file in importlib.metadata.distribution(name).files or []:
  p=pathlib.Path(file.locate()).resolve()
  if p.suffix=='.pyc' or not p.is_file(): continue
  if not p.is_relative_to(root): raise RuntimeError('Dependency file outside auxiliary environment')
  result[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
print(json.dumps(result,sort_keys=True))
"""
    dependency_files=json.loads(subprocess.check_output([python,'-c',file_probe,json.dumps(list(observation['pins']))],env=env,text=True))
    resource_files={str(p.relative_to(resources)):sha(p) for p in resources.rglob('*') if p.is_file()}
    report={'schema_version':1,'status':'passed','scope':'auxiliary CPU official IFEval dependencies/resources only',
        'venv':str(envpath),'python':python,'python_version':subprocess.check_output([python,'--version'],text=True).strip(),
        'installed_packages':versions,'package_pins':observation['pins'],'downloads':downloads,
        'dependency_files_sha256':dependency_files,
        'nltk_data':str(resources),'nltk_resource_files_sha256':resource_files,'nltk_data_revision':revision,
        'download_accounting':{'artifact_bytes':total_artifacts,'initial_observation_metadata_bytes':initial_metadata,
            'vendor_bytes':vendor_bytes,'metadata_additional_allowance_bytes':metadata_allowance,
            'first_failed_setup_repeated_transfer_allowance_bytes':retry_transfer_allowance,'cap_bytes':CAP,
            'upper_bound_bytes':total_artifacts+initial_metadata+vendor_bytes+metadata_allowance+retry_transfer_allowance},
        'vendor_provenance_sha256':sha(VENDOR/'UPSTREAM_PROVENANCE.json'),
        'main_environment_before_sha256':hashlib.sha256(json.dumps(before).encode()).hexdigest(),
        'main_environment_after_sha256':hashlib.sha256(json.dumps(after).encode()).hexdigest(),
        'main_environment_unchanged':True,'GPU_used':False,'source_sha256':sha(__file__)}
    save(BASE/'environment.json',report)
    (BASE/'requirements.lock').write_text(''.join(f'{k}=={v}\n' for k,v in sorted(observation['pins'].items())))
    print(json.dumps({'status':'passed','venv':str(envpath),'download_accounting':report['download_accounting']}))


if __name__=='__main__':main()
