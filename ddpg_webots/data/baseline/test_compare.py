"""Saved-report selection and metric tests, no training dependencies."""
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('report_comparison',Path(__file__).with_name('compare.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.old=(module.PROJECT,module.HERE);module.PROJECT=self.root;module.HERE=self.root/'baseline'
        module.HERE.mkdir()
        self.settings=dict(experiment='auto',same_experiment=True,window_env_steps=5000,
                           reports=dict(current='latest',previous='previous'))

    def tearDown(self):
        module.PROJECT,module.HERE=self.old;self.temp.cleanup()

    def fixture(self,name,report_name,rate):
        run=self.root/'turn/runs'/name;run.mkdir(parents=True,exist_ok=True)
        (run/'result.json').write_text(json.dumps(dict(status='completed')))
        (run/'effective_config.json').write_text(json.dumps(dict(config=dict(reward={},webots=dict(control_time_step_ms=100)))))
        rows=[dict(step=i,reward=1,fell=False,cycle_end=i%2==0,cycle_complete=i%2==0,
                   yaw_delta=rate*.1,cycle_dt=.1,wz_cmd=.1,motion_phase='turn',sensor_gyro_z=rate) for i in range(1,5)]
        with (run/'training.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        report=self.root/'turn/reports'/report_name;report.mkdir(parents=True,exist_ok=True)
        (report/'index.html').write_text('<html></html>')
        # Historical absolute paths must not be required after downloading.
        (report/'data.json').write_text(json.dumps(dict(run='F:\\removed\\runs\\'+name)))
        return report/'index.html'

    def test_latest_previous_are_distinct_runs(self):
        self.fixture('train_001','report_001',.05)
        self.fixture('train_002','report_002',.08)
        self.fixture('train_002','report_003',.08)
        result=module.build(self.settings)
        self.assertEqual(result['status'],'ready')
        self.assertAlmostEqual(result['current']['metrics']['net_rate'],.08)
        self.assertAlmostEqual(result['delta']['net_rate'],.03)
        self.assertEqual(result['current']['metrics']['full_turn_cycles'],2)

    def test_duplicate_report_is_not_previous_training(self):
        self.fixture('train_001','report_001',.05)
        self.fixture('train_001','report_002',.05)
        result=module.build(self.settings)
        self.assertEqual(result['status'],'pending_previous')
        self.assertNotIn('previous',result)

    def test_manual_reports_override_automatic_selection(self):
        a=self.fixture('train_001','report_001',.05)
        b=self.fixture('train_002','report_002',.08)
        self.settings['reports']=dict(current=str(a),previous=str(b))
        result=module.build(self.settings)
        self.assertAlmostEqual(result['delta']['net_rate'],-.03)


if __name__=='__main__':unittest.main()
