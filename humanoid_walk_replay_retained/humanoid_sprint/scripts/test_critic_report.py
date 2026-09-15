"""Statistical guards for mixed training logs and invalid MSE entries."""
import unittest
from critic_report import analyze

def update(step,loss,**extra):
    return dict(type='update',env_steps=str(step),critic_loss=loss,
                training_stage='joint',actor_lr='1e-5',critic_lr='1e-6',**extra)

class CriticReportTests(unittest.TestCase):
    def test_only_updates_count_and_zero_is_valid(self):
        data=analyze([update(1,'0'),update(2,'2'),dict(type='progress',env_steps='2',critic_loss='999')])
        self.assertEqual(data['overall']['count'],2)
        self.assertEqual(data['overall']['mean'],1.)
        self.assertEqual(data['peak']['env_steps'],2)

    def test_missing_invalid_values_are_not_zeros(self):
        data=analyze([update(i,v) for i,v in enumerate(['','nan','inf','-1','bad','2'])])
        self.assertEqual(data['missing'],1);self.assertEqual(data['nonfinite'],2)
        self.assertEqual(data['negative'],1);self.assertEqual(data['malformed'],1)
        self.assertEqual(data['overall']['mean'],2.)
        self.assertEqual(data['overall']['count'],1)

    def test_bc_and_stage_boundaries_reset_rolling_and_spike_baseline(self):
        rows=[update(i,'1') for i in range(1,7)]+[update(7,'20')]
        rows+=[dict(type='session_event',event_name='reset',bc_reset_applied='True',env_steps='8'),update(9,'100')]
        next_row=update(10,'200');next_row['actor_lr']='1e-6';rows.append(next_row)
        data=analyze(rows)
        self.assertEqual([r['env_steps'] for r in data['spikes']],[7])
        self.assertEqual(data['updates'][-2]['rolling_mean_20'],100.)
        self.assertEqual(data['updates'][-1]['rolling_mean_20'],200.)
        self.assertEqual(len(data['stages']),2)

    def test_final_windows_do_not_overlap(self):
        data=analyze([update(i,str(i)) for i in range(1,13)])
        trend=data['final_segment_trend']
        self.assertEqual(trend['window'],6)
        self.assertEqual(trend['first_mean'],3.5)
        self.assertEqual(trend['last_mean'],9.5)

    def test_absent_loss_is_reported_without_statistics(self):
        data=analyze([update(1,'')])
        self.assertIsNone(data['last_valid']);self.assertIsNone(data['overall']['mean'])
        self.assertEqual(data['final_segment_trend']['window'],0)

    def test_reset_after_last_update_has_no_new_segment_trend(self):
        rows=[update(i,'1') for i in range(12)]
        rows.append(dict(event_name='reset',type='session_event',bc_reset_applied='True',env_steps='12'))
        self.assertEqual(analyze(rows)['final_segment_trend']['window'],0)

if __name__=='__main__':unittest.main()
