"""Mocked boundary probe lifecycle; no real controller/GPU signals are sent."""
import copy
import json
from pathlib import Path
import signal
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import probe_research_batch_at_boundary as p


class TestBatchBoundaryProbe(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.spec=json.loads((Path(__file__).resolve().parents[1]/p.SPEC).read_text())
        self.job=next(j for j in self.spec['jobs'] if j['id']==p.TARGET_JOB)
        self.source=next(j for j in self.spec['jobs'] if j['id']==p.SOURCE_JOB)
        self.write(p.SPEC,self.spec)
        self.directory=p.at(self.root,self.job['argv'][self.job['argv'].index('--output')+1])
        self.directory.mkdir(parents=True)
        self.write(self.directory/'resolved_config.json',{'stage':'SFT','recipe':'1b','output':str(self.directory)})
        self.write(self.directory/'.checkpoint_manager.json',{'owner':'owner'})
        target=self.directory/'.checkpoint_objects/object-last';target.mkdir(parents=True)
        self.write(target/'.research_checkpoint.json',{'owner':'owner','step':170})
        (target/'state.pt').write_bytes(b'unchanged target trainer weights')
        for role in ('best','last'):(self.directory/role).symlink_to(target)
        self.training_command=['/venv/main/bin/python','-u',str(self.root/'scripts/run_research_stage.py'),*self.job['argv']]
        self.native={'pid':33707,'start_time':123,'state':'S','ppid':1,'cwd':str(self.root),
            'argv':['/venv/main/bin/python','-u',p.OLD_SCRIPT,'--spec',p.SPEC]}
        self.training={'pid':39254,'start_time':456,'state':'S','ppid':33707,'cwd':str(self.root),'argv':self.training_command}
        self.infos={33707:self.native,39254:None}
        self.context={'source_checkpoint':str(self.root/'q4/best/state.pt'),
            'checkpoint_identity':{'format':'legacy_single_file','sha256':'a'*64,'primary_path':str(self.root/'q4/state.pt'),
                'primary_bytes':100,'primary_mtime_ns':1,'total_bytes':100,'members':[]},
            'manifest':str(self.root/'manifest.jsonl'),'manifest_sha256':'b'*64,
            'record_ids':[str(i) for i in range(32)],'source_sha256':{'source':'hash'},'weighted_readiness_sha256':'c'*64}
        _,self.snapshot=p.spec_snapshot(self.root)
        self.state={'status':'running','spec_sha256':self.snapshot['canonical_spec_sha256'],
            'jobs':{p.TARGET_JOB:{'status':'running','pid':39254,'command':self.training_command}}}
        self.write(Path(p.SPEC).with_suffix('.state.json'),self.state)
        self.signals=[];self.exec_calls=[]
        self.patches=[patch.object(p,'ROOT',self.root),
            patch.object(p,'process_info',side_effect=lambda pid:self.infos.get(pid)),
            patch.object(p,'service_status',return_value='research-pilot RUNNING pid 33707, uptime 0:10:00'),
            patch.object(p,'source_context',return_value=self.context),
            patch.object(p,'host_available_bytes',return_value=200*1024**3),
            patch.object(p,'gpu_pids',side_effect=AssertionError('Unit test must mock GPU listing')),
            patch.object(p.DiskBudget,'ensure_space'),
            patch.object(p.os,'kill',side_effect=self.kill)]
        for patcher in self.patches:patcher.start()
    def tearDown(self):
        for patcher in reversed(self.patches):patcher.stop()
        self.temp.cleanup()
    def write(self,name,value):
        path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
    def complete(self):self.write(self.directory/'completed.json',{'steps':170,'records':10831})
    def kill(self,pid,sig):
        self.signals.append((pid,sig))
        if pid==33707 and sig==signal.SIGSTOP:self.native['state']='T'
        if pid==33707 and sig==signal.SIGCONT:self.native['state']='S'
    def passed_report(self):
        ids=[[7,2] for _ in range(32)]
        return dict(status='passed',batch_sizes=[1,4,8],max_new_tokens=64,
            manifest_sha256=self.context['manifest_sha256'],record_ids=self.context['record_ids'],
            source_sha256=self.context['source_sha256'],checkpoint_identity=self.context['checkpoint_identity'],
            observed_gpu_process_checks_exclusive=True,
            profiles=[dict(batch_size=b,records=32,seconds=float(32/b),records_per_second=float(b),
                           exact_token_parity_with_native_b1=True,generated_token_ids=ids) for b in [1,4,8]])
    def execute(self,root,context,report,receipt,record,timeout):
        self.exec_calls.append((root,context,report,timeout))
        self.write(report,self.passed_report());record['attempted']=True
        return self.passed_report()
    def run_probe(self):
        return p.run(self.root)
    def receipt(self):
        return dict(owner=p.OWNER,status='failed',parent_pause_outstanding=True,controller_pid=33707,
            controller_start_time=123,controller_command_sha256=p.command_sha(self.native['argv']),
            queue_spec_path=p.SPEC,**self.snapshot,attempted=True)

    def test_safe_completed_window_only_pauses_controller_and_restores_weights_and_state(self):
        self.complete();before_state=(self.root/Path(p.SPEC).with_suffix('.state.json')).read_bytes()
        before_weights=(self.directory/'last/state.pt').read_bytes()
        with patch.object(p,'wait_training_and_gpu'),patch.object(p,'execute_probe',side_effect=self.execute):
            result=self.run_probe()
        self.assertEqual(result['status'],'completed')
        self.assertEqual(self.signals,[(33707,signal.SIGSTOP),(33707,signal.SIGCONT)])
        self.assertEqual(before_state,(self.root/Path(p.SPEC).with_suffix('.state.json')).read_bytes())
        self.assertEqual(before_weights,(self.directory/'last/state.pt').read_bytes())
        self.assertFalse(p.read_json(self.root/p.RECEIPT)['parent_pause_outstanding'])

    def test_evaluation_started_before_window_skips_without_any_signal_or_probe(self):
        self.complete();self.state['jobs'][p.TARGET_JOB]['status']='evaluating'
        self.write(Path(p.SPEC).with_suffix('.state.json'),self.state)
        with patch.object(p,'execute_probe') as execute:
            result=self.run_probe()
        self.assertEqual(result['status'],'skipped');execute.assert_not_called();self.assertEqual(self.signals,[])

    def test_phase_advances_between_initial_check_and_pause_unpauses_without_probe(self):
        self.complete();original=self.kill
        def shifted(pid,sig):
            original(pid,sig)
            if sig==signal.SIGSTOP:
                self.state['jobs'][p.TARGET_JOB]['status']='evaluating'
                self.write(Path(p.SPEC).with_suffix('.state.json'),self.state)
        with patch.object(p.os,'kill',side_effect=shifted),patch.object(p,'execute_probe') as execute:
            result=self.run_probe()
        self.assertEqual(result['reason'],'phase_advanced_at_barrier')
        self.assertEqual(self.signals,[(33707,signal.SIGSTOP),(33707,signal.SIGCONT)]);execute.assert_not_called()

    def test_failure_or_parity_mismatch_always_continues_controller(self):
        self.complete()
        with patch.object(p,'wait_training_and_gpu'),patch.object(p,'execute_probe',side_effect=ValueError('token parity failed')):
            with self.assertRaisesRegex(ValueError,'parity'):self.run_probe()
        self.assertEqual(self.signals[-1],(33707,signal.SIGCONT))
        receipt=p.read_json(self.root/p.RECEIPT)
        self.assertEqual(receipt['status'],'failed');self.assertFalse(receipt['parent_pause_outstanding'])

    def test_shared_gpu_or_training_cleanup_timeout_never_signals_training_child(self):
        evidence={'training_pid':39254,'training_start_time':456}
        self.infos[39254]=self.training
        with patch.object(p,'gpu_pids',return_value=[999]):
            with self.assertRaisesRegex(RuntimeError,'Other GPU'):p.wait_training_and_gpu(evidence,timeout=0)
        with patch.object(p,'gpu_pids',return_value=[39254]):
            with self.assertRaisesRegex(RuntimeError,'cleanup exceeded'):p.wait_training_and_gpu(evidence,timeout=0)
        self.assertEqual(self.signals,[])

    def test_training_child_finishes_naturally_then_stale_nvml_pid_clears(self):
        evidence={'training_pid':39254,'training_start_time':456}
        with patch.object(p,'gpu_pids',side_effect=[[39254],[]]),patch.object(p.time,'sleep'):
            p.wait_training_and_gpu(evidence)
        self.assertEqual(self.signals,[])

    def test_startup_restores_only_owned_paused_parent_even_with_failed_report(self):
        self.native['state']='T';self.write(p.RECEIPT,self.receipt())
        self.write(p.REPORT,{'status':'failed','error':'previous parity mismatch'})
        with self.assertRaisesRegex(ValueError,'no automatic duplicate'):self.run_probe()
        self.assertEqual(self.signals,[(33707,signal.SIGCONT)])
        self.assertEqual(self.native['state'],'S')

    def test_recovery_wont_signal_reused_or_foreign_parent_pid(self):
        receipt=self.receipt();receipt['controller_start_time']=999;self.native['state']='T'
        self.assertFalse(p.resume_owned_parent(receipt))
        receipt=self.receipt();receipt['owner']='foreign'
        self.assertFalse(p.resume_owned_parent(receipt))
        self.assertEqual(self.signals,[])

    def test_controller_identity_change_before_pause_skips_without_any_signal(self):
        self.complete()
        def changed():
            self.native['start_time']=999
            return 200*1024**3
        with patch.object(p,'host_available_bytes',side_effect=changed),patch.object(p,'execute_probe') as execute:
            result=self.run_probe()
        self.assertEqual(result['reason'],'controller_changed_before_pause')
        execute.assert_not_called();self.assertEqual(self.signals,[])

    def test_verified_report_is_reused_without_duplicate_probe_or_controller_pause(self):
        self.write(p.REPORT,self.passed_report())
        with patch.object(p,'execute_probe') as execute:
            result=self.run_probe()
        self.assertTrue(result['report_reused']);execute.assert_not_called();self.assertEqual(self.signals,[])
        bad=self.passed_report();bad['profiles'][2]['generated_token_ids']=[[1]]*32
        self.write(p.REPORT,bad)
        with self.assertRaisesRegex(ValueError,'token IDs differ'):self.run_probe()
        self.assertEqual(self.signals,[])

    def test_malformed_or_wrong_spec_state_never_pauses_controller(self):
        self.complete()
        for key,value in [('spec_sha256','wrong'),('jobs',[])]:
            with self.subTest(key=key):
                state=dict(self.state);state[key]=value;self.write(Path(p.SPEC).with_suffix('.state.json'),state)
                with self.assertRaises(ValueError):self.run_probe()
        self.assertEqual(self.signals,[])

    def test_probe_cli_is_fixed32_b1_4_8_with_explicit_gpu_and_completed_q4_source(self):
        command=p.benchmark_command(self.root,self.context,self.root/p.REPORT)
        self.assertEqual(command[command.index('--records')+1],'32')
        self.assertEqual(command[command.index('--batch-sizes')+1:command.index('--max-new-tokens')],['1','4','8'])
        self.assertIn('--allow-gpu',command);self.assertIn(self.context['source_checkpoint'],command)
        self.assertNotIn('run_research_stage.py',' '.join(command))

    def test_owned_probe_timeout_terminates_only_probe_and_retains_original_exception(self):
        fake=SimpleNamespace(pid=777,returncode=None,poll=lambda:None,
            terminate=lambda:self.exec_calls.append('probe terminate'),
            wait=lambda **_:self.exec_calls.append('probe wait'),kill=lambda:self.exec_calls.append('probe kill'))
        record=self.receipt()
        with patch.object(p.subprocess,'Popen',return_value=fake),patch.object(p.time,'monotonic',side_effect=[0,1]):
            with self.assertRaisesRegex(TimeoutError,'600s cap'):
                p.execute_probe(self.root,self.context,self.root/p.REPORT,self.root/p.RECEIPT,record,timeout=.5)
        self.assertEqual(self.exec_calls,['probe terminate','probe wait']);self.assertEqual(self.signals,[])

    def test_parent_death_guard_applies_only_to_new_benchmark_child(self):
        libc=SimpleNamespace(prctl=lambda *args:0)
        with patch.object(p.ctypes,'CDLL',return_value=libc),patch.object(p.os,'getppid',return_value=123):
            p.arm_parent_death(123)
        with patch.object(p.ctypes,'CDLL',return_value=libc),patch.object(p.os,'getppid',return_value=456),patch.object(p.os,'_exit') as exit:
            p.arm_parent_death(123)
        exit.assert_called_once_with(128+signal.SIGTERM);self.assertEqual(self.signals,[])


if __name__=='__main__':unittest.main()

