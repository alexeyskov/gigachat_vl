"""Source-verified adapter to the unmodified pinned Google IFEval checker."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[2]
VENDOR=ROOT/'third_party/ifeval_google_87e25a5'
ENVIRONMENT=ROOT/'research_results/text_scoring_environment/environment.json'
COMMIT='87e25a5ba04d2c4cf0108f0cec96d648fe2b945a'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class OfficialIFEval:
    def __init__(self,environment_path=ENVIRONMENT):
        self.environment_path=Path(environment_path)
        self.environment=json.loads(self.environment_path.read_text())
        env=self.environment
        if (env.get('schema_version')!=1 or env.get('status')!='passed' or
                Path(sys.prefix).resolve()!=Path(env['venv']).resolve()):
            raise RuntimeError('Official IFEval requires its pinned auxiliary CPU environment, not /venv/main')
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
            raise RuntimeError('Official CPU scorer requires CUDA_VISIBLE_DEVICES empty')
        provenance=VENDOR/'UPSTREAM_PROVENANCE.json'
        if sha(provenance)!=env['vendor_provenance_sha256']:
            raise ValueError('Pinned Google source provenance was changed')
        self.upstream=json.loads(provenance.read_text())
        if self.upstream.get('commit')!=COMMIT or self.upstream.get('unmodified_upstream_source') is not True:
            raise ValueError('IFEval source revision/modification scope differs')
        for row in self.upstream['files']:
            path=(ROOT/row['path']).resolve(strict=True)
            if not path.is_relative_to(VENDOR.resolve()) or sha(path)!=row['sha256']:
                raise ValueError('Pinned Google checker/license source bytes differ')
        for name,version in env['package_pins'].items():
            if importlib.metadata.version(name)!=version:
                raise RuntimeError('Official IFEval dependency pin differs: '+name)
        dependency_files=env.get('dependency_files_sha256')
        if not isinstance(dependency_files,dict) or not dependency_files:
            raise RuntimeError('Actual auxiliary dependency file hashes are missing')
        prefix=Path(env['venv']).resolve()
        for name,expected in dependency_files.items():
            path=(prefix/name).resolve(strict=True)
            if not path.is_relative_to(prefix) or sha(path)!=expected:
                raise ValueError('Actual auxiliary dependency file content differs')
        self.resources=Path(env['nltk_data']).resolve(strict=True)
        for name,expected in env['nltk_resource_files_sha256'].items():
            path=(self.resources/name).resolve(strict=True)
            if not path.is_relative_to(self.resources) or sha(path)!=expected:
                raise ValueError('Pinned official NLTK resource content differs')
        sys.path.insert(0,str(VENDOR))
        from instruction_following_eval import evaluation_lib,instructions_registry
        import nltk
        import langdetect
        for module in (evaluation_lib,instructions_registry):
            if not Path(module.__file__).resolve().is_relative_to(VENDOR.resolve()):
                raise ValueError('Imported IFEval namespace is shadowed by another implementation')
        nltk.data.path[:]=[str(self.resources)]
        # Exercise both actual resources, rather than accepting installed names.
        if nltk.sent_tokenize('One sentence. Another sentence.')!=['One sentence.','Another sentence.']:
            raise RuntimeError('Pinned NLTK sentence tokenizer resource probe differs')
        if nltk.word_tokenize('One sentence.')!=['One','sentence','.']:
            raise RuntimeError('Pinned NLTK word tokenizer resource probe differs')
        # langdetect documents this reproducibility option. No checker changes.
        langdetect.DetectorFactory.seed=0
        self.library=evaluation_lib
        self.registry=instructions_registry.INSTRUCTION_DICT
        self.provenance={'repo':self.upstream['repo'],'commit':COMMIT,
            'vendor_provenance_sha256':sha(provenance),'environment_sha256':sha(self.environment_path),
            'package_pins':env['package_pins'],'nltk_data_revision':env['nltk_data_revision'],
            'langdetect_seed':0,'random_seed_policy':'SHA256(seed42|record_id), reset for strict/loose; restore global random state',
            'official_code_modified':False,'dependency_instructions_skipped':0}

    def score(self,record,response):
        if not isinstance(response,str):raise ValueError('IFEval raw response must be text')
        reference=record['reference'];ids=reference['instruction_id_list'];kwargs=reference['kwargs']
        if not ids or len(ids)!=len(kwargs) or any(i not in self.registry for i in ids):
            raise ValueError('Unknown/missing official instruction; no instruction may be skipped')
        if record['prompt']!=record['question']:
            raise ValueError('IFEval official prompt must be verbatim, without appended answer instructions')
        original=record['source']['original_id']
        if not original.startswith('key:'):raise ValueError('IFEval released source key missing')
        example=self.library.InputExample(key=int(original[4:]),instruction_id_list=ids,
            prompt=record['prompt'],kwargs=kwargs)
        old=random.getstate();seed=int(hashlib.sha256(('42|'+record['record_id']).encode()).hexdigest(),16)
        result={'record_id':record['record_id'],'task':'ifeval','instruction_ids':ids}
        try:
            for mode,function in [('strict',self.library.test_instruction_following_strict),
                                  ('loose',self.library.test_instruction_following_loose)]:
                random.seed(seed)
                output=function(example,{example.prompt:response})
                flags=output.follow_instruction_list
                if len(flags)!=len(ids) or any(type(flag) is not bool for flag in flags):
                    raise ValueError('Official IFEval did not return every instruction Boolean')
                result[mode]={'follow_all_instructions':output.follow_all_instructions,
                    'follow_instruction_list':list(flags)}
        finally:random.setstate(old)
        return result

    def preflight(self,records):
        selected=[r for r in records if r['task']=='ifeval']
        if not selected:raise ValueError('No IFEval records were supplied for dependency preflight')
        ids={i for r in selected for i in r['reference']['instruction_id_list']}
        if ids-set(self.registry):raise ValueError('Released instruction IDs are absent from pinned checker')
        # A nonempty synthetic response forces every actual instruction checker
        # to execute, including language detection and sentence/word tokenizers.
        for record in selected:self.score(record,'This is an English sentence. Here is another sentence.\nA short final line.')
        return {'status':'passed','records_exercised':len(selected),'distinct_instruction_ids':sorted(ids),
            'instruction_evaluations_per_mode':sum(len(r['reference']['instruction_id_list']) for r in selected),
            'synthetic_response_only':True,'model_generation_performed':False,'provenance':self.provenance}
