"""Verify separation, failure gates and paired evaluation metrics without Webots."""
import copy
import csv
import io
import contextlib
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import results_pipeline as p

class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.cfg=copy.deepcopy(p.load_config())
        self.cfg['experiment'].update(name='fixture',previous_experiment=None)
        self.world=self.root/'world.wbt';self.world.write_text('fixture world')
        self.cfg['webots']['world_template']=str(self.world)

    def batch(self,status='completed'):
        folder=self.root/'eval_batch_fixture';folder.mkdir()
        runs=[]
        for i,time,lateral,success in [(1,90.,1.2,True),(2,100.,-.1,True),(3,None,0.,False)]:
            child=folder/f'gs{i:06d}';child.mkdir();checkpoint=child/'model.pth';checkpoint.write_bytes(bytes([i]))
            digest=p.file_hash(checkpoint);snapshot=child/'effective_config.json';run_id='test'+str(i)
            p.write(snapshot,dict(config=self.cfg,run_id=run_id,source_world_sha256=p.file_hash(self.world),sources=[dict(sha256=digest)]))
            p.write(child/'evaluation_result.json',dict(eval_run_id=run_id,checkpoint_sha256=digest,episodes=[dict(reached=success,fell=not success,reach_time=time,lateral=lateral,forward=10. if success else .2,end_reason='goal' if success else 'fall')]))
            runs.append(dict(status='completed',snapshot=str(snapshot),checkpoint=str(checkpoint),sha256=digest,global_step=i))
        path=folder/'batch_status.json';p.write(path,dict(status=status,runs=runs));return path

    def test_rank_uses_same_episode_and_excludes_failures(self):
        rows,_=p.load_batch(self.batch(),'fixture');chosen=p.select(rows)
        self.assertEqual(chosen['fastest']['time_s'],90.)
        self.assertEqual(chosen['fastest']['abs_lateral_m'],1.2)
        self.assertEqual(chosen['straightest']['lateral_m'],-.1)
        self.assertEqual(chosen['straightest']['time_s'],100.)
        self.assertIsNone(rows[-1]['time_s'])

    def test_changed_checkpoint_is_rejected(self):
        path=self.batch();run=p.read(path)['runs'][0];Path(run['checkpoint']).write_bytes(b'changed')
        with self.assertRaises(ValueError):p.load_batch(path,'fixture')

    def test_incomplete_batch_is_not_reported_as_complete(self):
        path=self.batch('running')
        with self.assertRaises(ValueError):p.load_batch(path,'fixture')
        with patch.object(p,'experiment_root',return_value=self.root),patch.object(p,'references',return_value=[]),contextlib.redirect_stdout(io.StringIO()):
            out=p.report(self.cfg,path)
        data=p.read(out/'comparison.json')
        self.assertEqual(data['status'],'pending_evaluation');self.assertEqual(data['results'],[])

    def test_train_report_does_not_start_evaluation(self):
        with patch.object(p,'experiment_root',return_value=self.root),patch.object(p,'train',return_value=self.world),patch.object(p,'quick_report',return_value=self.root) as quick,patch.object(p,'prepare_evaluation') as evaluation:
            p.run('train-report',self.cfg)
        quick.assert_called_once();evaluation.assert_not_called()

    def test_training_failure_stops_report_and_evaluation(self):
        with patch.object(p,'experiment_root',return_value=self.root),patch.object(p,'train',side_effect=RuntimeError('failed')),patch.object(p,'quick_report') as quick,patch.object(p,'prepare_evaluation') as evaluation:
            with self.assertRaises(RuntimeError):p.run('train-report',self.cfg)
        quick.assert_not_called();evaluation.assert_not_called()
        status=p.read(next((self.root/'pipelines').glob('*/status.json')))
        self.assertEqual(status['status'],'failed')

    def test_evaluation_failure_stops_full_report(self):
        with patch.object(p,'experiment_root',return_value=self.root),patch.object(p,'latest_batch',return_value=None),patch.object(p,'prepare_evaluation',return_value=self.world),patch.object(p,'run_evaluation_batch',side_effect=RuntimeError('failed')),patch.object(p,'report') as report:
            with self.assertRaises(RuntimeError):p.run('eval-report',self.cfg)
        report.assert_not_called()

    def test_successful_eval_reports_exact_new_batch(self):
        with patch.object(p,'experiment_root',return_value=self.root),patch.object(p,'latest_batch',return_value=None),patch.object(p,'prepare_evaluation',return_value=self.world),patch.object(p,'run_evaluation_batch') as evaluate,patch.object(p,'report',return_value=self.root) as report:
            p.run('eval-report',self.cfg)
        evaluate.assert_called_once_with(self.world);report.assert_called_once_with(self.cfg,self.world)

    def test_complete_report_has_all_paired_rows(self):
        path=self.batch()
        with patch.object(p,'experiment_root',return_value=self.root),patch.object(p,'references',return_value=[]),contextlib.redirect_stdout(io.StringIO()):out=p.report(self.cfg,path)
        with (out/'results.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),3);self.assertEqual(rows[1]['lateral_m'],'-0.1')
        self.assertTrue((out/'report.html').exists())

    def test_wait_requires_completed_batch(self):
        path=self.batch('running')
        def finish(_):
            manifest=p.read(path);manifest['status']='completed';p.write(path,manifest)
        with patch.object(p,'evaluation_process_alive',return_value=True),patch.object(p.time,'sleep',side_effect=finish),contextlib.redirect_stdout(io.StringIO()):p.wait_for_evaluation(path)
        self.assertEqual(p.read(path)['status'],'completed')

    def test_dead_evaluation_is_not_waited_forever(self):
        path=self.batch('running')
        with patch.object(p,'evaluation_process_alive',return_value=False):
            with self.assertRaises(RuntimeError):p.wait_for_evaluation(path)

if __name__=='__main__':unittest.main()
