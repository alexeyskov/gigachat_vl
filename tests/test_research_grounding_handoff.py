"""Controller migration never signals a trainer or rewrites training state."""
import copy
import json
from pathlib import Path
import signal
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import handoff_research_grounding as h


class TestGroundingHandoff(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.spec=json.loads((Path(__file__).resolve().parents[1]/h.SPEC).read_text())
        self.job=next(job for job in self.spec['jobs'] if '--stage' in job['argv'] and job['argv'][job['argv'].index('--stage')+1]=='SFT')
        self.config=self.root/'supervisor/research-pilot.conf';self.config.parent.mkdir()
        self.config.write_text('[program:research-pilot]\nautorestart=false\n'+h.OLD_COMMAND+'\n')
        self.patches=[patch.object(h,'ROOT',self.root),patch.object(h,'CONFIG',self.config)]
        for p in self.patches:p.start()
        self.write(h.SPEC,self.spec)
        sources={}
        for name in h.REQUIRED_SOURCES:
            p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('validated '+name)
            sources[name]=h.digest(p.read_bytes())
        training=self.root/'scripts/run_research_stage.py';training.write_text('validated training')
        self.write('configs/research/weighted_sft_ready.json',{'status':'passed','source_sha256':{'scripts/run_research_stage.py':h.digest(training.read_bytes())}})
        self.write(h.READY,{'status':'passed','tests_passed':45,'test_command':'pytest focused',
            'checks':dict.fromkeys(h.CHECKS,True),'source_sha256':sources})
        self.directory=h.at(self.root,self.job['argv'][self.job['argv'].index('--output')+1])
        self.write(self.directory/'resolved_config.json',{'stage':'SFT','output':str(self.directory)})
        self.write(self.directory/'completed.json',{'steps':170})
        self.write(self.directory/'checkpoint_compaction.json',{'status':'completed','training_completed':True,'resumable_optimizer':False})
        self.write(self.directory/'.checkpoint_manager.json',{'owner':'owner'})
        target=self.directory/'.checkpoint_objects/object-owned';target.mkdir(parents=True)
        (target/'state.pt').write_bytes(b'controlled local checkpoint')
        self.write(target/'.research_checkpoint.json',{'owner':'owner','step':170,'resumable_optimizer':False,'training_completed':True})
        for label in ('best','last'):(self.directory/label).symlink_to(target)
        self.manifest=h.at(self.root,self.job['evaluation']['manifest'])
        self.manifest.parent.mkdir(parents=True,exist_ok=True);self.manifest.write_text('{"record_id":"one"}\n')
        self.output=h.at(self.root,self.job['evaluation']['output'])
        self.write(self.output/'protocol.json',{'image_ablation':{'variant':'none'},'manifest_sha256':h.digest(self.manifest.read_bytes())})
        self.native={'pid':1001,'ppid':1,'start_time':11,'state':'S','cwd':str(self.root),
            'argv':['/venv/main/bin/python','-u',h.OLD_SCRIPT,'--spec',h.SPEC]}
        self.child={'pid':2002,'ppid':1001,'start_time':22,'state':'S','cwd':str(self.root),
            'argv':['/venv/main/bin/python','-u',str(self.root/'scripts/eval_research_checkpoint.py'),
                '--run',str(self.directory),'--manifest',str(self.manifest),'--output',str(self.output),
                '--checkpoint','best','--max-new-tokens','64','--reserve-gb','120']}
        self.replacement=dict(self.native,pid=3003,start_time=33,
            argv=['/venv/main/bin/python','-u',h.NEW_SCRIPT,'--spec',h.SPEC])
        self.infos={1001:self.native,2002:self.child,3003:self.replacement}
        self.proc_patch=patch.object(h,'process_info',side_effect=lambda pid:self.infos.get(pid))
        self.proc_patch.start()
        _,self.snapshot=h.spec_snapshot(self.root)
        self.identity=dict(self.snapshot,**h.ready_proof(self.root))
        self.state={'status':'running','spec_sha256':self.snapshot['canonical_spec_sha256'],
            'jobs':{self.job['id']:{'status':'evaluating','training_completed':True,
                'evaluation':{'status':'running','pid':2002,'output':str(self.output)}}}}
        self.write(Path(h.SPEC).with_suffix('.state.json'),self.state)

    def tearDown(self):
        self.proc_patch.stop()
        for p in reversed(self.patches):p.stop()
        self.temp.cleanup()

    def write(self,name,value):
        p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))
    def boundary(self,state=None):
        return h.safe_evaluation_boundary(self.root,self.spec,state or self.state,self.native)
    def receipt(self,status='stopping'):
        original=self.config.read_text()
        return dict(status=status,**self.identity,**self.boundary(),
            source_supervisor_sha256=h.digest(original.encode()),
            replacement_supervisor_sha256=h.digest(h.configuration(original).encode()))
    def kill(self,pid,sig):
        self.signals.append((pid,sig))
        if sig==signal.SIGSTOP:self.infos[pid]['state']='T'
        if sig==signal.SIGCONT:self.infos[pid]['state']='S'

    def test_live_safe_phase_requires_compaction_none_exact_child_and_same_spec(self):
        self.assertEqual(self.boundary()['evaluation_pid'],2002)
        with self.subTest('not a trainer boundary'):
            state=copy.deepcopy(self.state);state['jobs'][self.job['id']]['status']='running'
            self.assertIsNone(self.boundary(state))
        with self.subTest('foreign parent'):
            self.child['ppid']=999;self.assertIsNone(self.boundary());self.child['ppid']=1001
        with self.subTest('different evaluator output'):
            old=list(self.child['argv']);self.child['argv'][self.child['argv'].index('--output')+1]='/wrong'
            self.assertIsNone(self.boundary());self.child['argv']=old
        with self.subTest('ablation'):
            self.write(self.output/'protocol.json',{'image_ablation':{'variant':'shuffle'}})
            self.assertIsNone(self.boundary())
        with self.subTest('stale spec'):
            state=copy.deepcopy(self.state);state['spec_sha256']='stale'
            with self.assertRaisesRegex(ValueError,'state/specification'):self.boundary(state)

    def test_malformed_state_or_missing_compaction_never_stops_any_process(self):
        with patch.object(h.os,'kill') as kill:
            for state in ({'spec_sha256':self.snapshot['canonical_spec_sha256'],'jobs':[]},
                          {'spec_sha256':self.snapshot['canonical_spec_sha256'],'jobs':{'x':[]}}):
                with self.assertRaisesRegex(ValueError,'Malformed'):self.boundary(state)
            (self.directory/'checkpoint_compaction.json').unlink()
            self.assertIsNone(self.boundary())
        kill.assert_not_called()

    def test_readiness_requires_tests_and_exact_current_source_hashes(self):
        self.assertIsNotNone(h.ready_proof(self.root))
        source=self.root/'scripts/research_grounding_queue.py';source.write_text('untested change')
        with self.assertRaisesRegex(ValueError,'source changed'):h.ready_proof(self.root)

    def test_barrier_signals_controller_only_and_always_unpauses(self):
        boundary=self.boundary();self.signals=[]
        before=(self.root/Path(h.SPEC).with_suffix('.state.json')).read_bytes()
        with patch.object(h.os,'kill',side_effect=self.kill):
            self.assertTrue(h.pause_and_interrupt(self.root,self.spec,self.identity,boundary))
        self.assertEqual(self.signals,[(1001,signal.SIGSTOP),(1001,signal.SIGINT),(1001,signal.SIGCONT)])
        self.assertEqual(before,(self.root/Path(h.SPEC).with_suffix('.state.json')).read_bytes())
        self.assertEqual(h.read_json(self.root/h.RECEIPT)['status'],'stopping')

    def test_phase_shift_after_pause_unpauses_without_pending_interrupt(self):
        boundary=self.boundary();self.signals=[]
        state=copy.deepcopy(self.state);state['jobs'][self.job['id']]['status']='running'
        self.write(Path(h.SPEC).with_suffix('.state.json'),state)
        with patch.object(h.os,'kill',side_effect=self.kill):
            self.assertFalse(h.pause_and_interrupt(self.root,self.spec,self.identity,boundary))
        self.assertEqual(self.signals,[(1001,signal.SIGSTOP),(1001,signal.SIGCONT)])
        self.assertFalse((self.root/h.RECEIPT).exists())

    def test_barrier_exception_or_changed_spec_always_continues_controller(self):
        boundary=self.boundary();self.signals=[]
        changed=copy.deepcopy(self.spec);changed['queue_id']='changed'
        self.write(h.SPEC,changed)
        with patch.object(h.os,'kill',side_effect=self.kill):
            with self.assertRaisesRegex(ValueError,'identity changed'):
                h.pause_and_interrupt(self.root,self.spec,self.identity,boundary)
        self.assertEqual(self.signals,[(1001,signal.SIGSTOP),(1001,signal.SIGCONT)])

    def test_same_pid_reused_for_different_evaluator_never_interrupts_parent(self):
        boundary=self.boundary();self.signals=[];self.child['start_time']=999
        with patch.object(h.os,'kill',side_effect=self.kill):
            self.assertFalse(h.pause_and_interrupt(self.root,self.spec,self.identity,boundary))
        self.assertEqual(self.signals,[(1001,signal.SIGSTOP),(1001,signal.SIGCONT)])

    def test_existing_trainer_process_is_never_paused_or_interrupted(self):
        boundary=self.boundary()
        self.native['argv']=['/venv/main/bin/python','-u','scripts/run_research_stage.py','--stage','SFT']
        with patch.object(h.os,'kill') as kill:
            self.assertFalse(h.pause_and_interrupt(self.root,self.spec,self.identity,boundary))
        kill.assert_not_called()

    def test_stale_nvml_listing_is_polled_without_killing_anything(self):
        results=[SimpleNamespace(stdout='2002\n'),SimpleNamespace(stdout='2002\n'),SimpleNamespace(stdout='')]
        with patch.object(h.subprocess,'run',side_effect=results) as run,patch.object(h.time,'sleep'),patch.object(h.os,'kill') as kill:
            h.wait_gpu_idle()
        self.assertEqual(run.call_count,3);kill.assert_not_called()
        with patch.object(h.subprocess,'run',return_value=SimpleNamespace(stdout='999\n')),patch.object(h.os,'kill') as kill:
            with self.assertRaisesRegex(RuntimeError,'CUDA process remains'):h.wait_gpu_idle(timeout=0)
        kill.assert_not_called()

    def test_stopped_stopping_and_starting_receipts_recover_without_live_evaluator(self):
        for phase in ('stopping','starting'):
            with self.subTest(phase=phase):
                previous=self.receipt(phase);self.write(h.RECEIPT,previous)
                with patch.object(h.DiskBudget,'ensure_space'),patch.object(h,'service_status',return_value='research-pilot STOPPED'),\
                     patch.object(h,'launch_replacement') as launch,patch.object(h,'safe_evaluation_boundary',side_effect=AssertionError('no fresh boundary required')),\
                     patch.object(h.os,'kill') as kill:
                    h.main()
                launch.assert_called_once();kill.assert_not_called()

    def test_running_owned_wrapper_recovers_before_gpu_idle_without_signals(self):
        previous=self.receipt('starting');self.write(h.RECEIPT,previous)
        self.config.write_text(h.configuration(self.config.read_text()))
        with patch.object(h.DiskBudget,'ensure_space'),patch.object(h,'service_status',return_value='research-pilot RUNNING pid 3003, uptime 0:01:00'),\
             patch.object(h,'wait_gpu_idle',side_effect=AssertionError('existing wrapper may own GPU')),\
             patch.object(h.os,'kill') as kill:
            h.main()
        kill.assert_not_called()
        self.assertTrue(h.read_json(self.root/h.RECEIPT)['recovered_after_start'])

    def test_recovery_rejects_changed_proof_or_owned_configuration(self):
        previous=self.receipt('starting');self.write(h.RECEIPT,dict(previous,readiness_sha256='stale'))
        with patch.object(h.DiskBudget,'ensure_space'),patch.object(h,'service_status',return_value='research-pilot STOPPED'),patch.object(h.os,'kill') as kill:
            with self.assertRaisesRegex(ValueError,'identity changed'):h.main()
        kill.assert_not_called()
        self.config.write_text('[program:other]\ncommand=other\nautorestart=false\n')
        with self.assertRaisesRegex(ValueError,'command changed'):h.verify_stopped_recovery(self.root,self.spec,previous)

    def test_launch_uses_same_spec_state_and_only_research_pilot_service(self):
        previous=self.receipt();self.write(h.RECEIPT,previous)
        before_spec=(self.root/h.SPEC).read_bytes()
        before_state=(self.root/Path(h.SPEC).with_suffix('.state.json')).read_bytes()
        with patch.object(h,'wait_gpu_idle'),patch.object(h,'service_status',return_value='research-pilot STOPPED'),\
             patch.object(h,'wait_for_running',return_value='research-pilot RUNNING pid 3003, uptime 0:00:05'),\
             patch.object(h.subprocess,'run') as run,patch.object(h.os,'kill') as kill:
            h.launch_replacement(self.root,self.spec,self.identity,previous)
        commands=[call.args[0] for call in run.call_args_list]
        self.assertEqual(commands,[['supervisorctl','reread'],['supervisorctl','update','research-pilot'],['supervisorctl','start','research-pilot']])
        self.assertFalse(any('stop' in command for command in commands));kill.assert_not_called()
        self.assertIn(h.NEW_COMMAND,self.config.read_text())
        self.assertEqual(before_spec,(self.root/h.SPEC).read_bytes())
        self.assertEqual(before_state,(self.root/Path(h.SPEC).with_suffix('.state.json')).read_bytes())
        self.assertEqual(h.read_json(self.root/h.RECEIPT)['status'],'completed')

    def test_service_becoming_active_before_reload_prevents_any_configuration_mutation(self):
        previous=self.receipt();before=self.config.read_bytes()
        with patch.object(h,'wait_gpu_idle'),patch.object(h,'service_status',return_value='research-pilot RUNNING pid 1001'),\
             patch.object(h.subprocess,'run') as run,patch.object(h.os,'kill') as kill:
            with self.assertRaisesRegex(RuntimeError,'no running task stopped'):
                h.launch_replacement(self.root,self.spec,self.identity,previous)
        self.assertEqual(before,self.config.read_bytes());run.assert_not_called();kill.assert_not_called()


if __name__=='__main__':unittest.main()

