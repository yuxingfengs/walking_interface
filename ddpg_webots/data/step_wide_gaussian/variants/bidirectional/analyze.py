"""Analyze saved online transitions without simulation or learning."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def cycles(frame):
    pending, result = [], []
    for row in frame.to_dict('records'):
        pending.append(row)
        if str(row['cycle_end']).lower() != 'true':
            continue
        if str(row['cycle_complete']).lower() == 'true' and not any(str(r['fell']).lower() == 'true' for r in pending):
            target = pending[0]['wz_cmd']
            assert all(abs(r['wz_cmd'] - target) < 1e-6 for r in pending)
            angle = np.array([r['yaw_delta'] for r in pending])
            duration = sum(r['cycle_dt'] for r in pending)
            rate = angle.sum() / duration
            result.append(dict(start=pending[0]['step'], end=row['step'], target=target,
                               net_rate=rate, error=abs(rate-target),
                               forward_angle=np.maximum(np.sign(target)*angle, 0).sum(),
                               reverse_angle=np.maximum(-np.sign(target)*angle, 0).sum(),
                               gyro=np.mean([r['sensor_gyro_z'] for r in pending]),
                               reward=np.mean([r['reward'] for r in pending])))
        pending = []
    return pd.DataFrame(result)


def main():
    summary = {}
    for label, name in [('bidirectional','train_20261120T013956_af7547'),
                        ('positive_only','train_20261118T032820_6a15b3')]:
        folder = ROOT / 'runs' / name
        frame = pd.read_csv(folder / 'training.csv')
        c = cycles(frame)
        c.to_csv(OUT / (label + '_cycles.csv'), index=False)
        data = dict(result=json.loads((folder/'result.json').read_text()), windows={})
        for lo, hi in [(1,5000),(5001,10000),(10001,15000),(15001,20000),(1,20000)]:
            item = {}
            for sign, key in [(1,'positive'),(-1,'negative')]:
                selected = c[(c.start>=lo)&(c.end<=hi)&(c.target*sign>0)]
                steps = frame[(frame.step>=lo)&(frame.step<=hi)&(frame.wz_cmd*sign>0)]
                if len(selected):
                    item[key] = dict(cycles=len(selected), net_rate=selected.net_rate.mean(),
                        error=selected.error.mean(), reverse_angle=selected.reverse_angle.mean(),
                        forward_angle=selected.forward_angle.mean(), reward=selected.reward.mean(),
                        gyro=selected.gyro.mean(), falls=int(steps.fell.sum()),
                        near_target=int((selected.error<=.01).sum()))
            data['windows'][str(lo)+'-'+str(hi)] = item
        data['fall_steps'] = frame.loc[frame.fell, ['step','wz_cmd']].to_dict('records')
        data['actor_updates'] = int(frame.actor_updated.sum())
        if label == 'bidirectional':
            switch = frame.loc[frame.wz_cmd.diff().abs()>.1,'step'].tolist()
            first = []
            for step in switch:
                matched = c[c.start==step]
                if len(matched): first.append(matched.iloc[0].to_dict())
            data['switch_count'] = len(switch)
            data['switch_first_complete_cycles'] = first
            meta = json.loads((folder/'best_reward_cycle/metadata.json').read_text())
            data['best_metadata'] = meta
        summary[label] = data
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    for label, data in summary.items():
        print(label, json.dumps({k:v for k,v in data.items() if k not in ('best_metadata','switch_first_complete_cycles')},indent=2))
    print('BEST',json.dumps(summary['bidirectional']['best_metadata'],indent=2)[:2400])


if __name__ == '__main__': main()
