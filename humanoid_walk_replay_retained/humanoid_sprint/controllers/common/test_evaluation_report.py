import unittest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from report_evaluation import fastest,successes,valid_rows


class EvaluationReportTests(unittest.TestCase):
    def row(self,step,time,lateral,success=1):
        return dict(global_step=step,reach_time_mean=time,lateral_mean=lateral,eval_success_rate=success)

    def test_best_uses_same_row_and_absolute_deviation_tiebreak(self):
        a=self.row(1,110,-.9);b=self.row(2,110,.2);c=self.row(3,112,.01)
        self.assertIs(fastest(successes([a,b,c])),b)

    def test_failure_is_not_an_arrival(self):
        self.assertEqual(successes([self.row(1,None,0,0)]),[])
        self.assertIsNone(fastest([]))

    def test_partial_batch_only_uses_completed_candidates(self):
        r=self.row(1,110,.2);r['checkpoint_sha256']='abc'
        b=dict(runs=[dict(global_step=1,status='completed',sha256='abc',result=r),
                     dict(global_step=2,status='running')])
        self.assertEqual(valid_rows(b),[r])
        b['runs'][0]['sha256']='different'
        with self.assertRaises(ValueError): valid_rows(b)
