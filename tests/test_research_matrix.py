"""Matrix accounting never promotes pilots, CE-only scores or stale artifacts."""
import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.utils.research_storage import BestLastCheckpointManager, DiskBudget

spec = importlib.util.spec_from_file_location("research_matrix", Path(__file__).parents[1]/"scripts/research_matrix.py")
matrix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(matrix)


class TestResearchMatrix(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root/'research_results').mkdir()
        (self.root/'configs/research').mkdir(parents=True)
        self.inventory = [{'repo':repo,'sha':'a'*40} for _,repo,_ in matrix.MODELS] + [{'repo':repo,'sha':'b'*40} for _,repo in matrix.ENCODERS]
        (self.root/'research_results/model_inventory.json').write_text(json.dumps(self.inventory))
    def tearDown(self):
        self.temp.cleanup()
    def write(self, path, data):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(data))
    def make_run(self, scope='primary', stage='SFT', recipe='4f', completed=True):
        directory = self.root/'runs/research'/f'smallthinker4__siglip2__{recipe}__{stage}_{scope}_s42'
        directory.mkdir(parents=True,exist_ok=True)
        manifest = self.root/'datasets'/f'{scope}_{stage}.jsonl'
        manifest.parent.mkdir(exist_ok=True)
        manifest.write_text(json.dumps({'image_paths':['image.png'],'question':'q','answer':'a'})+'\n')
        stats = matrix.manifest_statistics(manifest,{})
        cfg = {'model':matrix.MODELS[0][1],'vision':matrix.ENCODERS[0][1], 'recipe':recipe,'seed':42,
               'stage':stage,'manifest':str(manifest),'manifest_sha256':stats['sha256'],
               'validation_sha256':'validation-hash','output':str(directory),'epochs':1,
               'model_revision':'a'*40,'encoder_revision':'b'*40}
        if stage=='SFT':
            cfg.update(selection_metric='group_macro',validation=str(manifest),selection_manifest_sha256=stats['sha256'])
        self.write(directory/'resolved_config.json',cfg)
        manager = BestLastCheckpointManager(directory,DiskBudget(self.root,0),mode='max' if stage=='SFT' else 'min')
        manager.save(lambda d:(d/'state.pt').write_bytes(b'model-payload'),step=1,metric=.5,anticipated_bytes=100)
        if stage=='SFT':
            self.write(directory/'selection/best.protocol.json',{'name':'internal_teacher_reference_group_macro',
                'official_benchmark_score':False,'selection_direction':'max','seed':42,
                'selection_manifest_sha256':'selection-content-hash','validation_step':1})
            self.write(directory/'selection/best.metrics.json',{'validation_step':1,
                'groups':{group:{'score':50.} for group in matrix.TASK_GROUPS}})
        if completed:self.write(directory/'completed.json',{'steps':1,'records':1})
        return directory,cfg
    def update_config(self,directory,**changes):
        cfg=json.loads((directory/'resolved_config.json').read_text());cfg.update(changes)
        self.write(directory/'resolved_config.json',cfg)
        return cfg
    def primary_fixture(self,benchmark=True):
        a0,c0=self.make_run(stage='A0')
        a1,c1=self.make_run(stage='A1')
        sft,cfg=self.make_run(stage='SFT')
        self.update_config(a1,initial_checkpoint=str(a0/'best/state.pt'))
        cfg=self.update_config(sft,initial_checkpoint=str(a1/'best/state.pt'))
        for directory,count in ((a0,200000),(a1,10000),(sft,588235)):
            self.write(directory/'completed.json',{'steps':1,'records':count})
        self.write(sft/'scientific_data_audit.json',{'status':'passed','train_eval_overlap_checked':True,
                                                   'manifest_sha256':cfg['manifest_sha256'],'source_revisions_pinned':True})
        evidence=matrix.checkpoint_evidence(sft)['checkpoints']['best']
        receipt={'status':'passed','checkpoint_path':evidence['path'],'checkpoint_owner':evidence['owner'],
                 'checkpoint_step':evidence['step'],'checkpoint_size_bytes':evidence['bytes'],
                 'checkpoint_mtime_ns':evidence['mtime_ns'],'validated_tensor_count':5,
                 'all_tensors_finite':True,'expected_keys_match':True}
        self.write(sft/'checkpoint_roundtrip.json',{'status':'passed','checkpoints':[receipt]})
        report={'status':'completed','evaluation_kind':'official_benchmark','checkpoint_path':evidence['path'],
                'checkpoint_size_bytes':evidence['bytes'],'checkpoint_mtime_ns':evidence['mtime_ns'],
                'manifest_sha256':cfg['manifest_sha256'],'evaluation_manifest_sha256':'evaluation-hash',
                'harness_revision':'harness-commit','train_eval_overlap_checked':True,
                'task_groups':list(matrix.TASK_GROUPS),'metrics':{'general_vqa':51.,'ocr_documents':42.},
                'task_group_scores':{group:50. for group in matrix.TASK_GROUPS}}
        if benchmark:self.write(sft/'benchmark_results.json',report)
        def stats(path,cache):
            value=matrix.manifest_statistics_original(path,cache)
            if 'primary_A0' in str(path):value.update(records=200000,multimodal_records=200000,text_records=0)
            elif 'primary_A1' in str(path):value.update(records=10000,multimodal_records=10000,text_records=0)
            elif 'primary_SFT' in str(path):value.update(records=588235,multimodal_records=500000,text_records=88235)
            return value
        return sft,stats,report
    def primary_row(self,report):
        return next(row for row in report['primary'] if row['model_id']=='smallthinker4' and row['encoder_id']=='siglip2' and row['recipe']=='4f' and row['seed']==42)

    def test_exact_396_unique_primary_trajectories_preserve_order_and_seeds(self):
        report=matrix.build_report(self.root)
        rows=report['primary']
        self.assertEqual(len(rows),396)
        self.assertEqual(len({row['trajectory_id'] for row in rows}),396)
        self.assertEqual(report['model_execution_order'],['smallthinker4','lfm25','giga31','gptoss20'])
        self.assertEqual(set(row['seed'] for row in rows),{42,123,2026})
        self.assertEqual(sum(row['status']=='completed' for row in rows),0)
        self.assertEqual(sum(report['primary_status_counts'].values()),396)

    def test_pilot_smoke_and_intermediate_completion_never_promote_primary(self):
        self.make_run(scope='pilot')
        self.make_run(scope='smoke')
        self.make_run(scope='primary',stage='A0')
        report=matrix.build_report(self.root)
        self.assertEqual(report['primary_status_counts'].get('completed',0),0)
        self.assertEqual(report['primary_status_counts'].get('training_completed',0),0)
        self.assertEqual(sum(row['status']=='training_completed' for row in report['observed_stages']),3)

    def test_primary_name_with_tiny_data_is_protocol_failure_not_trained_primary(self):
        self.make_run()
        report=matrix.build_report(self.root)
        row=self.primary_row(report)
        self.assertEqual(row['status'],'failed')
        self.assertIn('SFT_multimodal_budget_below_500K',row['status_reason'])

    def test_training_complete_without_official_eval_remains_distinct(self):
        matrix.manifest_statistics_original=matrix.manifest_statistics
        _,statistics,_=self.primary_fixture(benchmark=False)
        with patch.object(matrix,'manifest_statistics',side_effect=statistics):
            report=matrix.build_report(self.root)
        row=self.primary_row(report)
        self.assertEqual(row['status'],'training_completed')
        self.assertFalse(row['official_benchmark_passed'])

    def test_scientific_complete_requires_all_budgets_roundtrip_and_bound_benchmarks(self):
        matrix.manifest_statistics_original=matrix.manifest_statistics
        _,statistics,_=self.primary_fixture()
        with patch.object(matrix,'manifest_statistics',side_effect=statistics):
            report=matrix.build_report(self.root)
        row=self.primary_row(report)
        self.assertEqual(row['status'],'completed')
        self.assertTrue(row['official_benchmark_passed'])
        self.assertTrue(row['roundtrip_passed'])
        self.assertEqual(row['manifest_sha256'],matrix.read_json(Path(row['run_directory'])/'resolved_config.json')['manifest_sha256'])

    def test_ce_only_or_stale_evaluation_does_not_complete_science(self):
        matrix.manifest_statistics_original=matrix.manifest_statistics
        directory,statistics,benchmark=self.primary_fixture()
        for changes in ({'evaluation_kind':'heldout_ce'}, {'checkpoint_mtime_ns':0}, {'task_groups':['general_vqa']}, {'metrics':{'bad':float('nan')}}, {'image_ablation':{'variant':'shuffle'}}, {'image_ablation':{'variant':'text_only'}}):
            with self.subTest(changes=changes):
                self.write(directory/'benchmark_results.json',dict(benchmark,**changes))
                with patch.object(matrix,'manifest_statistics',side_effect=statistics):
                    row=self.primary_row(matrix.build_report(self.root))
                self.assertEqual(row['status'],'training_completed')

    def test_ce_selected_primary_or_stale_selection_cannot_complete_science(self):
        matrix.manifest_statistics_original=matrix.manifest_statistics
        directory,statistics,_=self.primary_fixture()
        self.update_config(directory,selection_metric='ce')
        with patch.object(matrix,'manifest_statistics',side_effect=statistics):
            row=self.primary_row(matrix.build_report(self.root))
        self.assertNotEqual(row['status'],'completed')
        self.assertIn('SFT_checkpoint_selection_not_internal_group_macro',row['status_reason'])
        self.update_config(directory,selection_metric='group_macro')
        protocol=matrix.read_json(directory/'selection/best.protocol.json')
        self.write(directory/'selection/best.protocol.json',dict(protocol,validation_step=0))
        with patch.object(matrix,'manifest_statistics',side_effect=statistics):
            row=self.primary_row(matrix.build_report(self.root))
        self.assertNotEqual(row['status'],'completed')
        self.assertIn('SFT_internal_selection_artifacts_missing_or_stale',row['status_reason'])

    def test_verified_disposable_smoke_cleanup_is_separate_from_missing_primary_weights(self):
        directory,_=self.make_run(scope='smoke')
        cfg=self.update_config(directory,max_records=128,max_steps=2)
        self.write(directory/'checkpoint_roundtrip.json',{'status':'passed','historical_receipt':True})
        manager=matrix.read_json(directory/'.checkpoint_manager.json')
        receipt={'status':'pruned','owner':manager['owner'],'run_dir':str(directory),
            'scope':'disposable_technical_smoke_weights_only',
            'config_sha256':hashlib.sha256((directory/'resolved_config.json').read_bytes()).hexdigest(),
            'validation_receipt_sha256':hashlib.sha256((directory/'checkpoint_roundtrip.json').read_bytes()).hexdigest()}
        self.write(directory/'technical_smoke_pruned.json',receipt)
        self.assertTrue(matrix.technical_weights_retired(directory,cfg))
        for label in ('best','last'):(directory/label).unlink()
        row=matrix.scan_runs(self.root,{})[0]
        self.assertEqual(row['status'],'training_completed')
        self.assertFalse(row['checkpoint_evidence']['retained'])
        self.assertEqual(row['scope'],'smoke')
        self.assertFalse(matrix.technical_weights_retired(directory,dict(cfg,manifest='primary-data')))
        self.write(directory/'checkpoint_roundtrip.json',{'status':'changed'})
        self.assertFalse(matrix.technical_weights_retired(directory,cfg))

    def test_post_compaction_stale_roundtrip_receipt_must_be_regenerated(self):
        matrix.manifest_statistics_original=matrix.manifest_statistics
        directory,statistics,_=self.primary_fixture()
        (directory/'last/state.pt').write_bytes(b'different-new-model-object-file')
        with patch.object(matrix,'manifest_statistics',side_effect=statistics):
            row=self.primary_row(matrix.build_report(self.root))
        self.assertNotEqual(row['status'],'completed')
        self.assertFalse(row['roundtrip_passed'])

    def test_compacted_completed_model_remains_scientifically_eligible(self):
        matrix.manifest_statistics_original=matrix.manifest_statistics
        directory,statistics,_=self.primary_fixture()
        marker=directory/'last/.research_checkpoint.json'
        data=json.loads(marker.read_text());data['resumable_optimizer']=False
        marker.write_text(json.dumps(data))
        with patch.object(matrix,'manifest_statistics',side_effect=statistics):
            report=matrix.build_report(self.root)
        row=self.primary_row(report)
        self.assertEqual(row['status'],'completed')
        observed=next(r for r in report['observed_stages'] if r['directory']==str(directory))
        self.assertFalse(observed['checkpoint_evidence']['optimizer_resumable'])

    def test_pending_overlap_audit_cannot_be_scientifically_completed(self):
        matrix.manifest_statistics_original=matrix.manifest_statistics
        directory,statistics,_=self.primary_fixture()
        self.write(directory/'scientific_data_audit.json',{'status':'pending'})
        with patch.object(matrix,'manifest_statistics',side_effect=statistics):
            row=self.primary_row(matrix.build_report(self.root))
        self.assertEqual(row['status'],'failed')
        self.assertIn('scientific_data_audit_missing_or_pending',row['status_reason'])

    def test_failed_oom_job_is_resource_blocked_not_completed(self):
        directory,cfg=self.make_run(scope='pilot',completed=False)
        spec_path=self.root/'configs/research/test.json'
        self.write(spec_path,{'jobs':[{'id':directory.name,'argv':['--output',str(directory),'--stage','SFT','--recipe','4f']}]})
        self.write(spec_path.with_suffix('.state.json'),{'jobs':{directory.name:{'status':'failed','error_type':'OutOfMemoryError','error':'CUDA out of memory'}}})
        report=matrix.build_report(self.root)
        observed=next(r for r in report['observed_stages'] if r['directory']==str(directory))
        self.assertEqual(observed['status'],'resource_blocked')
        self.assertEqual(report['primary_status_counts'].get('completed',0),0)

    def test_resource_forecast_uses_total_parameters_not_active_or_quantized_file_bytes(self):
        report=matrix.build_report(self.root)
        cap=report['capacity']
        self.assertEqual(cap['reserve_gib'],120)
        full=cap['models']['gptoss20']['profiles']['4f']
        self.assertAlmostEqual(full['fp32_training_state_lower_bound_gib'],20_000_000_000*16/2**30,places=2)
        self.assertTrue(full['state_alone_exceeds_current_gpu'])
        row=next(row for row in report['primary'] if row['model_id']=='gptoss20' and row['recipe']=='4f')
        self.assertEqual(row['status'],'resource_blocked')
        self.assertEqual(row['execution_status'],'not_started')
        self.assertFalse(cap['all_final_weights_fit_current_disk'])

    def test_reports_include_every_primary_row_without_rewriting_configs(self):
        directory,_=self.make_run(scope='pilot')
        before=(directory/'resolved_config.json').read_bytes()
        report=matrix.build_report(self.root)
        output=self.root/'research_results'
        matrix.write_reports(report,output)
        import csv
        with (output/'matrix.csv').open() as handle:rows=list(csv.DictReader(handle))
        self.assertEqual(len(rows),396)
        self.assertEqual(before,(directory/'resolved_config.json').read_bytes())
        self.assertLess((output/'matrix.md').stat().st_size,30000)
        self.assertTrue((output/'matrix.json').is_file())


if __name__=='__main__':unittest.main()
