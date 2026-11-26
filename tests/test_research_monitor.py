"""Telemetry follows newer validation phases instead of extrapolating stale training."""
import json
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys
from unittest.mock import patch

from scripts import monitor_research as monitor


class TestResearchMonitor(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.root=Path(self.temporary.name)
        self.path=self.root/'progress.json'
        self.path.write_text(json.dumps({'run_id':'test','step':100,'total_steps':170,
            'completed_records':6400,'total_records':10831,'elapsed_seconds':1000}))
        os.utime(self.path,(1000,1000))

    def tearDown(self):self.temporary.cleanup()

    def write_phase(self,name,data,when):
        p=self.root/'runs/research/test'/name;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(data));os.utime(p,(when,when))

    def test_internal_validation_has_own_counts_and_elapsed(self):
        self.write_phase('selection_progress.json',{'event':'internal_selection_record','completed_records':29,
            'total_records':116,'prediction':'hidden','score':1.},1060)
        r=monitor.read_progress(self.path,root=self.root)
        self.assertEqual(r['phase'],'internal_validation');self.assertEqual(r['completed_records'],29)
        self.assertEqual(r['elapsed_seconds'],60);self.assertEqual(r['eta_current_stage_hours'],.05)
        self.assertEqual(r['training_progress']['step'],100);self.assertNotIn('prediction',r)
        os.utime(self.path,(1070,1070))
        self.assertEqual(monitor.read_progress(self.path,root=self.root)['phase'],'train')

    def test_benchmark_progress_and_loading_do_not_use_old_training_eta(self):
        self.write_phase('evaluation/heldout_v3/progress.json',{'event':'generated','completed_records':64,
            'selected_records':512,'elapsed_seconds':128},1100)
        r=monitor.read_progress(self.path,root=self.root)
        self.assertEqual(r['phase'],'benchmark_evaluation');self.assertEqual(r['total_records'],512)
        with patch.object(monitor,'live_queue_job',return_value={'run_id':'next'}):
            r=monitor.read_progress(self.path,root=self.root,queue_state='test.json')
        self.assertEqual(r['phase'],'loading_model');self.assertNotIn('eta_current_stage_hours',r)

    def test_missing_process_is_not_verified_from_queue_state_alone(self):
        state=self.root/'queue.json';state.write_text(json.dumps({'jobs':{'test':{'status':'running','pid':999999999}}}))
        self.assertIsNone(monitor.live_queue_job(state))
        self.assertFalse(monitor.read_progress(self.path,root=self.root,queue_state=state)['activity_verified'])

    def test_grounding_has_separate_live_variant_and_progress(self):
        output=self.root/'runs/research/test/evaluation/grounding_v1/shuffle'
        self.write_phase('evaluation/grounding_v1/shuffle/progress.json',{'event':'generated',
            'completed_records':10,'selected_records':512,'elapsed_seconds':20},1120)
        queue=self.root/'queue.json';queue.write_text(json.dumps({'jobs':{'test':{
            'status':'grounding_evaluating','active_grounding_variant':'shuffle',
            'grounding_evaluations':{'shuffle':{'status':'running','pid':123,'output':str(output)}}}}}))
        with patch.object(Path,'read_bytes',return_value=b'python\0scripts/eval_research_checkpoint.py\0'):
            r=monitor.read_progress(self.path,root=self.root,queue_state=queue)
        self.assertEqual(r['phase'],'grounding_shuffle');self.assertTrue(r['activity_verified'])
        self.assertEqual(r['total_records'],512);self.assertEqual(r['completed_records'],10)

    def test_checkpoint_validation_does_not_display_stale_benchmark_completion(self):
        self.write_phase('evaluation/heldout_v3/progress.json',{'event':'completed','records':512,
            'manifest_records':512},1120)
        queue=self.root/'queue.json';queue.write_text(json.dumps({'jobs':{'test':{
            'status':'checkpoint_validating','checkpoint_validation':{'status':'running','pid':124}}}}))
        with patch.object(Path,'read_bytes',return_value=b'python\0scripts/validate_research_checkpoint.py\0'):
            r=monitor.read_progress(self.path,root=self.root,queue_state=queue)
        self.assertEqual(r['phase'],'checkpoint_roundtrip');self.assertTrue(r['activity_verified'])
        self.assertNotIn('eta_current_stage_hours',r)

    def test_previous_run_grounding_overrides_newer_training_run_progress(self):
        output=self.root/'runs/research/previous/evaluation/grounding_v1/shuffle'
        output.mkdir(parents=True)
        (output/'progress.json').write_text(json.dumps({'event':'generated','completed_records':43,
            'selected_records':512,'elapsed_seconds':86,'prediction':'private','score':0.}))
        active={'run_id':'previous','evaluation_active':True,'grounding_variant':'shuffle',
                'evaluation':{'output':str(output)}}
        with patch.object(monitor,'live_queue_job',return_value=active):
            result=monitor.read_progress(self.path,root=self.root,queue_state='queue.json')
        self.assertEqual(result['run_id'],'previous')
        self.assertEqual(result['phase'],'grounding_shuffle')
        self.assertEqual(result['completed_records'],43)
        self.assertEqual(result['eta_current_stage_hours'],938/3600)
        self.assertNotIn('prediction',result)


class TestContainerMemory(unittest.TestCase):
    def test_v2_hard_limit_takes_precedence_over_host_ram(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);info=root/'meminfo'
            info.write_text('MemTotal: 1000 kB\nMemAvailable: 900 kB\n')
            (root/'memory.max').write_text('600000')
            (root/'memory.current').write_text('400000')
            (root/'memory.stat').write_text('inactive_file 50000\n')
            result=monitor.ram_snapshot(info,root)
        self.assertEqual(result['MemTotal'],1024000)
        self.assertEqual(result['effective_capacity_bytes'],600000)
        self.assertEqual(result['conservative_available_bytes'],200000)
        self.assertEqual(result['cgroup']['inactive_file_bytes'],50000)

    def test_unlimited_or_unavailable_cgroup_preserves_host_counters(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);info=root/'meminfo'
            info.write_text('MemTotal: 1000 kB\nMemAvailable: 900 kB\n')
            result=monitor.ram_snapshot(info,root)
            self.assertNotIn('cgroup',result)
            (root/'memory.max').write_text('max');(root/'memory.current').write_text('400000')
            result=monitor.ram_snapshot(info,root)
        self.assertIsNone(result['cgroup']['limit_bytes'])
        self.assertEqual(result['effective_capacity_bytes'],1024000)
        self.assertEqual(result['conservative_available_bytes'],921600)

    def test_v1_and_over_limit_accounting_clamp_headroom_to_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);info=root/'meminfo';memory=root/'memory';memory.mkdir()
            info.write_text('MemTotal: 1000 kB\nMemAvailable: 900 kB\n')
            (memory/'memory.limit_in_bytes').write_text('600000')
            (memory/'memory.usage_in_bytes').write_text('650000')
            result=monitor.ram_snapshot(info,root)
        self.assertEqual(result['cgroup']['version'],1)
        self.assertEqual(result['conservative_available_bytes'],0)


class TestVerifiedBatch4Probe(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.path=self.root/'train-progress.json'
        self.path.write_text(json.dumps({'run_id':'old','completed_records':10000,
            'total_records':10831,'elapsed_seconds':1000}))
        self.run=self.root/'runs/research/owned-run'
        self.output=self.run/'evaluation/batch4_v1/none';self.output.mkdir(parents=True)
        self.argv=['/venv/main/bin/python','-u',str(self.root/'scripts'/monitor.BATCH4_SCRIPT),
            '--run',str(self.run),'--manifest',str(self.root/'manifest.jsonl'),'--checkpoint','best',
            '--image-ablation','none','--max-new-tokens','64','--ablation-seed','42',
            '--compare-native-cache',str(self.run/'evaluation/heldout_v3')]
        self.record={'owner':monitor.PROBE_OWNER,'status':'running_b4','worker_pid':123,'worker_start_time':456,
            'worker_command':self.argv,'worker_command_sha256':hashlib.sha256(json.dumps(self.argv,separators=(',',':')).encode()).hexdigest(),
            'job_id':self.run.name,'source':{'run':str(self.run),'output':str(self.output),
                'manifest':str(self.root/'manifest.jsonl'),'native_cache':str(self.run/'evaluation/heldout_v3')}}
        self.receipt=self.root/monitor.PROBE_RECEIPT;self.receipt.parent.mkdir(parents=True)
        self.write_receipt()
        self.process={'pid':123,'start_time':456,'argv':list(self.argv)}
        self.patch=patch.object(monitor,'process_identity',side_effect=lambda pid:self.process if pid==123 else None)
        self.patch.start()
    def tearDown(self):self.patch.stop();self.temp.cleanup()
    def write_receipt(self):self.receipt.write_text(json.dumps(self.record))
    def progress(self,**values):
        (self.output/'progress.json').write_text(json.dumps(dict(profile='batch4_v1',**values)))
        return monitor.read_progress(self.path,root=self.root)

    def test_guard_phase_uses32_counts_without_old_training_eta_or_private_fields(self):
        result=self.progress(event='guard_native_b1',variant='shuffle',completed_records=12,total_records=32,
            prediction='private text',answers=['private'],score=1,generated_token_ids=[7,8],nested={'answer':'private'},eta_seconds=1)
        self.assertEqual(result['phase'],'batch4_guard_shuffle')
        self.assertTrue(result['activity_verified']);self.assertEqual(result['activity_source'],'verified_batch4_probe')
        self.assertEqual(result['completed_records'],12);self.assertEqual(result['total_records'],32)
        self.assertNotIn('eta_current_stage_hours',result);self.assertNotIn('elapsed_seconds',result)
        for field in ('prediction','answers','score','generated_token_ids','nested','eta_seconds'):self.assertNotIn(field,result)

    def test_main_generation_uses512_and_only_measured_elapsed_yields_eta(self):
        result=self.progress(event='generated_batch',completed_records=64,manifest_records=512)
        self.assertEqual(result['phase'],'batch4_generation');self.assertEqual(result['total_records'],512)
        self.assertNotIn('eta_current_stage_hours',result)
        result=self.progress(event='generated_batch',completed_records=64,elapsed_seconds=32)
        self.assertEqual(result['eta_current_stage_hours'],224/3600)
        done=self.progress(event='completed',completed_records=512)
        self.assertEqual(done['phase'],'batch4_generation_completed');self.assertEqual(done['total_records'],512)

    def test_live_probe_overrides_even_missing_training_progress_without_fabricating_counts(self):
        self.path.unlink()
        result=monitor.read_progress(self.path,root=self.root)
        self.assertEqual(result['phase'],'batch4_loading_model');self.assertTrue(result['activity_verified'])
        self.assertNotIn('completed_records',result);self.assertNotIn('eta_current_stage_hours',result)

    def test_failed_completed_foreign_or_stale_receipts_never_override_queue_phase(self):
        for change in ({'status':'failed'},{'status':'completed'},{'owner':'foreign'},{'worker_start_time':999}):
            with self.subTest(change=change):
                original=dict(self.record);self.record.update(change);self.write_receipt()
                self.assertIsNone(monitor.live_batch4_probe(self.root))
                result=monitor.read_progress(self.path,root=self.root)
                self.assertEqual(result['phase'],'train');self.record=original

    def test_pid_reuse_or_actual_command_mismatch_is_not_verified(self):
        self.process['start_time']=999
        self.assertIsNone(monitor.live_batch4_probe(self.root))
        self.process['start_time']=456;self.process['argv']=list(self.argv)+['--resume']
        self.assertIsNone(monitor.live_batch4_probe(self.root))
        self.process=None
        self.assertIsNone(monitor.live_batch4_probe(self.root))

    def test_wrong_recorded_script_hash_or_output_is_not_verified(self):
        for change in ({'worker_command_sha256':'wrong'},
                       {'worker_command':['python','other.py']},
                       {'source':dict(self.record['source'],output=str(self.run/'evaluation/heldout_v3'))}):
            with self.subTest(change=change):
                original=dict(self.record);self.record.update(change);self.write_receipt()
                self.assertIsNone(monitor.live_batch4_probe(self.root));self.record=original

    def test_invalid_guard_count_cannot_make512_row_guard_eta(self):
        result=self.progress(event='guard_batch4',variant='none',completed_records=512,total_records=512,elapsed_seconds=1)
        self.assertEqual(result['total_records'],32);self.assertNotIn('completed_records',result)
        self.assertNotIn('eta_current_stage_hours',result)

    def test_future_batch4_live_queue_worker_is_approved_and_maps_guard_counts(self):
        self.receipt.unlink()
        self.progress(event='guard_batch4',variant='blank',completed_records=4,total_records=32)
        queue=self.root/'queue.json';queue.write_text(json.dumps({'jobs':{'owned-run':{
            'status':'evaluating','evaluation':{'pid':123,'output':str(self.output)}}}}))
        with patch.object(Path,'read_bytes',return_value=('\0'.join(self.argv)+'\0').encode()):
            result=monitor.read_progress(self.path,root=self.root,queue_state=queue)
        self.assertEqual(result['phase'],'batch4_guard_blank');self.assertEqual(result['total_records'],32)
        self.assertEqual(result['activity_source'],'verified_live_queue');self.assertTrue(result['activity_verified'])


class TestMonitorProcessIdentity(unittest.TestCase):
    def test_birthtime_parses_comm_with_spaces_and_zombie_is_not_live(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);process=root/'123';process.mkdir()
            fields=['S','1']+['0']*17+['456']
            (process/'stat').write_text('123 (worker with spaces) '+' '.join(fields))
            (process/'cmdline').write_bytes(b'python\0-u\0worker.py\0')
            self.assertEqual(monitor.process_identity(123,root),{'pid':123,'ppid':1,'start_time':456,'argv':['python','-u','worker.py']})
            fields[0]='Z';(process/'stat').write_text('123 (worker) '+' '.join(fields))
            self.assertIsNone(monitor.process_identity(123,root))
            self.assertIsNone(monitor.process_identity(True,root))

    def test_import_does_not_load_torch_or_training_stack(self):
        result=subprocess.run([sys.executable,'-c',
            "import sys; import scripts.monitor_research; assert 'torch' not in sys.modules; assert 'scripts.run_research_stage' not in sys.modules"],
            cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,check=True)
        self.assertEqual(result.returncode,0)


class TestParallelNativeB1Telemetry(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.root=Path(self.temporary.name)
        self.run=self.root/'runs/research/test';self.output=self.run/'evaluation/parallel_b1_v1/workers4/none'
        self.attempt=self.output/'execution/attempt';self.attempt.mkdir(parents=True)
        self.parent={'pid':123,'start_time':456,'argv':[sys.executable,'-u',
            str(self.root/'scripts'/monitor.PARALLEL_SCRIPT),'--run',str(self.run),'--workers','4','--image-ablation','none']}
        self.protocol={'scope':'full512_parallel','workers':4,'manifest_records':512,'checkpoint_sha256':'abc'}
        (self.output/'protocol.json').write_text(json.dumps(self.protocol))
        self.digest=hashlib.sha256(json.dumps(self.protocol,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        self.barrier={'owner':monitor.PARALLEL_OWNER,'status':'released','parent':self.parent,
            'context':{'run':str(self.run),'checkpoint_identity':{'sha256':'abc'}},'output':str(self.output),
            'protocol_path':str(self.output/'protocol.json'),'protocol_sha256':self.digest,'nonce':'nonce',
            'initial_pending':{str(i):list(range(i,512,4)) for i in range(4)},
            'cohort':[{'logical_worker':i} for i in range(4)]}
        self.write_barrier()
        self.patch=patch.object(monitor,'process_identity',return_value=self.parent);self.patch.start()

    def tearDown(self):
        self.patch.stop();self.temporary.cleanup()

    def write_barrier(self):
        path=self.attempt/'allocation_barrier.json';path.write_text(json.dumps(self.barrier))
        self.barrier_sha=hashlib.sha256(path.read_bytes()).hexdigest()
        (self.attempt/'receipt.json').write_text(json.dumps({'owner':monitor.PARALLEL_OWNER,'status':'running',
            'allocation_barrier_sha256':self.barrier_sha,'protocol_sha256':self.digest}))

    def gate(self):
        (self.attempt/'generation_barrier.json').write_text(json.dumps({'allocation_barrier_sha256':self.barrier_sha,
            'nonce':'nonce','released_monotonic':100.}))

    def test_guard_counts_models_and_never_fabricates_generation_eta(self):
        for i in range(2):
            (self.attempt/f'worker{i}.progress.json').write_text(json.dumps({'phase':'guard_completed',
                'completed_records':64,'prediction':'private'}))
        (self.attempt/'worker2.progress.json').write_text(json.dumps({'phase':'native_guard64_generation','completed_records':3}))
        value=monitor.read_progress(self.root/'missing.json',root=self.root)
        self.assertEqual(value['phase'],'parallel_native_b1_guard')
        self.assertEqual(value['guard_models_completed'],2)
        self.assertEqual(value['active_models'],4);self.assertEqual(value['guard_unique_records'],64)
        self.assertEqual(value['guard_model_requests_completed'],131);self.assertEqual(value['guard_model_requests_total'],256)
        self.assertNotIn('completed_records',value);self.assertNotIn('eta_seconds',value)
        self.assertNotIn('private',str(value))

    def test_resumed_rows_do_not_inflate_this_execution_throughput(self):
        self.barrier['initial_pending']={str(i):list(range(256+i,512,4)) for i in range(4)}
        self.write_barrier();self.gate()
        with patch.object(monitor.time,'monotonic',return_value=160.):
            value=monitor.parallel_b1_progress(self.root)
            self.assertEqual(value['completed_records'],256);self.assertNotIn('eta_seconds',value)
            (self.attempt/'worker0.progress.json').write_text(json.dumps({'phase':'native_b1_generation','completed_records':10}))
            value=monitor.parallel_b1_progress(self.root)
        self.assertEqual(value['total_records'],512);self.assertEqual(value['completed_records'],266)
        self.assertEqual(value['elapsed_seconds'],60.);self.assertEqual(value['eta_seconds'],1476.)

    def test_stale_pid_failed_receipt_and_corrupt_bindings_are_not_live(self):
        with patch.object(monitor,'process_identity',return_value=dict(self.parent,start_time=457)):
            self.assertIsNone(monitor.parallel_b1_progress(self.root))
        receipt=self.attempt/'receipt.json';value=json.loads(receipt.read_text());value['status']='failed';receipt.write_text(json.dumps(value))
        self.assertIsNone(monitor.parallel_b1_progress(self.root))
        self.write_barrier();self.barrier['initial_pending']['0'].append(1);self.write_barrier()
        self.assertIsNone(monitor.parallel_b1_progress(self.root))

    def test_invalid_generation_count_or_gate_cannot_override_training(self):
        self.gate()
        (self.attempt/'worker0.progress.json').write_text(json.dumps({'phase':'native_b1_generation','completed_records':129}))
        with patch.object(monitor.time,'monotonic',return_value=160.):
            self.assertIsNone(monitor.parallel_b1_progress(self.root))


class TestSerialReference64Telemetry(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.root=Path(self.temporary.name)
        self.run=self.root/'runs/research/test';self.manifest=self.root/'evaluation.jsonl'
        self.output=self.run/'evaluation/parallel_b1_v1/workers4/native_guard/blank'
        self.attempt=self.output/'execution'/('b'*32);self.attempt.mkdir(parents=True)
        self.nonce='a'*64
        script=str(self.root/'scripts'/monitor.PARALLEL_SCRIPT)
        self.parent={'pid':123,'ppid':99,'start_time':456,'argv':[sys.executable,'-u',script,
            '--run',str(self.run),'--manifest',str(self.manifest),'--workers','4','--image-ablation','blank']}
        self.worker={'pid':124,'ppid':123,'start_time':789,'argv':[sys.executable,'-u',script,'--private-worker',
            '--barrier',str(self.attempt/'allocation_barrier.json'),'--logical-worker','0','--parent-pid','123',
            '--parent-start','456','--nonce',self.nonce]}
        self.worker['command_sha256']=hashlib.sha256(json.dumps(self.worker['argv'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
        self.selected=[i for start in range(0,512,64) for i in range(start,start+8)]
        self.protocol={'owner':'parallel_native_b1_evaluator_v1','namespace':'parallel_b1_v1',
            'scope':'guard64_serial_reference','assignment':'named_serial_guard64','workers':1,
            'manifest_records':512,'evaluation_records':64,'seed':42,'variant':'blank',
            'manifest_sha256':'a'*64,'transformed_manifest_sha256':'b'*64,'checkpoint_sha256':'c'*64,
            'selected_global_indices':self.selected}
        self.barrier={'owner':monitor.PARALLEL_OWNER,'status':'released','parent':self.parent,
            'context':{'run':str(self.run),'manifest':str(self.manifest),'variant':'blank',
                'manifest_sha256':'a'*64,'transformed_manifest_sha256':'b'*64,'checkpoint_identity':{'sha256':'c'*64}},
            'output':str(self.output),'protocol_path':str(self.output/'protocol.json'),'nonce':self.nonce,
            'world_size':1,'guard_required':False,'initial_committed':[],
            'initial_pending':{'0':self.selected},'cohort':[{'logical_worker':0,'identity':self.worker}]}
        self.write_bindings();self.progress(16)
        self.processes={123:self.parent,124:self.worker}
        self.patch=patch.object(monitor,'process_identity',side_effect=lambda pid:self.processes.get(pid));self.patch.start()

    def tearDown(self):self.patch.stop();self.temporary.cleanup()

    def write_bindings(self):
        (self.output/'protocol.json').write_text(json.dumps(self.protocol))
        self.bound=hashlib.sha256(json.dumps(self.protocol,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        self.barrier['protocol_sha256']=self.bound
        path=self.attempt/'allocation_barrier.json';path.write_text(json.dumps(self.barrier))
        self.barrier_sha=hashlib.sha256(path.read_bytes()).hexdigest()
        (self.attempt/'receipt.json').write_text(json.dumps({'owner':monitor.PARALLEL_OWNER,'status':'running',
            'cohort':self.barrier['cohort'],'allocation_barrier_sha256':self.barrier_sha,'protocol_sha256':self.bound}))
        (self.attempt/'generation_barrier.json').write_text(json.dumps({'owner':monitor.PARALLEL_OWNER,
            'status':'released','nonce':self.nonce,'allocation_barrier_sha256':self.barrier_sha,'released_monotonic':100.}))

    def progress(self,count):
        (self.attempt/'worker0.progress.json').write_text(json.dumps({'phase':'native_b1_generation',
            'logical_worker':0,'completed_records':count,'total_records':len(self.barrier['initial_pending']['0']),
            'prediction':'must remain private','score':99}))

    def test_actual_parent_workers4_has_one_diagnostic_worker_and64_counts(self):
        with patch.object(monitor.time,'monotonic',return_value=132.):
            value=monitor.read_progress(self.root/'missing.json',root=self.root)
        self.assertEqual(value['phase'],'parallel_native_b1_serial_reference64')
        self.assertEqual((value['completed_records'],value['total_records']),(16,64))
        self.assertEqual((value['workers'],value['configured_parallel_workers'],value['active_models']),(1,4,1))
        self.assertEqual(value['worker_pid'],124);self.assertEqual(value['parent_pid'],123)
        self.assertTrue(value['diagnostic']);self.assertFalse(value['scientific_full512_progress'])
        self.assertEqual(value['eta_seconds'],96.);self.assertNotIn('private',str(value));self.assertNotIn('score',value)

    def test_loading_reference_has_no_invented_generation_eta(self):
        (self.attempt/'worker0.progress.json').unlink();(self.attempt/'generation_barrier.json').unlink()
        value=monitor.parallel_b1_progress(self.root)
        self.assertEqual((value['completed_records'],value['total_records']),(0,64))
        self.assertNotIn('eta_seconds',value)

    def test_partial_reference_resume_counts_only_this_attempt_for_eta(self):
        self.barrier['initial_committed']=self.selected[:8];self.barrier['initial_pending']={'0':self.selected[8:]}
        self.write_bindings();self.progress(8)
        with patch.object(monitor.time,'monotonic',return_value=116.):value=monitor.parallel_b1_progress(self.root)
        self.assertEqual(value['completed_records'],16);self.assertEqual(value['eta_seconds'],96.)

    def test_exited_foreign_or_reused_worker_never_reports_stale_reference(self):
        for changed in (None,dict(self.worker,start_time=790),dict(self.worker,ppid=999),dict(self.worker,argv=['python','foreign.py'])):
            with self.subTest(changed=changed):
                self.processes[124]=changed
                self.assertIsNone(monitor.parallel_b1_progress(self.root))

    def test_parent_pid_birth_argv_and_ppid_must_match(self):
        for changed in (dict(self.parent,start_time=457),dict(self.parent,ppid=100),dict(self.parent,argv=['python','foreign.py'])):
            with self.subTest(changed=changed):
                self.processes[123]=changed
                self.assertIsNone(monitor.parallel_b1_progress(self.root))

    def test_wrong_scope_path_nonce_or_progress_count_is_refused(self):
        for field,value in [('scope','full512_parallel'),('workers',4),('evaluation_records',512),('variant','text_only')]:
            with self.subTest(field=field):
                original=self.protocol[field];self.protocol[field]=value;self.write_bindings()
                self.assertIsNone(monitor.parallel_b1_progress(self.root));self.protocol[field]=original
        self.write_bindings();self.progress(65);self.assertIsNone(monitor.parallel_b1_progress(self.root))
        self.progress(16);self.barrier['nonce']='f'*64;self.write_bindings()
        self.assertIsNone(monitor.parallel_b1_progress(self.root))

    def test_serial_protocol_at_scientific_output_path_is_refused(self):
        self.barrier['output']=str(self.run/'evaluation/parallel_b1_v1/workers4/blank')
        self.write_bindings();self.assertIsNone(monitor.parallel_b1_progress(self.root))

    def test_symlinked_progress_and_bad_receipt_cohort_are_refused(self):
        progress=self.attempt/'worker0.progress.json';progress.rename(self.attempt/'copied.json');progress.symlink_to(self.attempt/'copied.json')
        self.assertIsNone(monitor.parallel_b1_progress(self.root));progress.unlink();self.progress(16)
        receipt=self.attempt/'receipt.json';value=json.loads(receipt.read_text());value['cohort']=[];receipt.write_text(json.dumps(value))
        self.assertIsNone(monitor.parallel_b1_progress(self.root))
