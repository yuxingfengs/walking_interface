"""Saved-run analysis; endpoint targets separated from ramp and zero commands."""
import hashlib
import importlib.util
import json
from pathlib import Path
import pandas as pd
import numpy as np

OUT = Path(__file__).resolve().parent
WIDE = OUT.parents[1]
spec = importlib.util.spec_from_file_location('previous_analysis',WIDE/'variants/bidirectional/analyze.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)


def stats(c):
    return dict(cycles=len(c),net_rate=float(c.net_rate.mean()),error=float(c.error.mean()),
                reverse_angle=float(c.reverse_angle.mean()),reward=float(c.reward.mean()),
                near_target=int((c.error<=.01).sum())) if len(c) else None


def main():
    result = {}
    for label,folder in [('gradual',OUT/'runs/train_20261120T024038_d7c199'),
                         ('abrupt',WIDE/'runs/train_20261120T013956_af7547')]:
        frame = pd.read_csv(folder/'training.csv')
        c = previous.cycles(frame)
        zero = c.target.abs()<1e-6
        c.loc[zero,['reverse_angle','forward_angle']] = np.nan
        c.to_csv(OUT/(label+'_cycles.csv'),index=False)
        data = dict(result=json.loads((folder/'result.json').read_text()),
                    cycles=len(c),actor_updates=int(frame.actor_updated.sum()),windows={})
        for lo,hi in [(1,5000),(5001,10000),(10001,15000),(15001,20000),(1,20000)]:
            window = c[(c.start>=lo)&(c.end<=hi)]
            item={}
            for sign,key in [(1,'positive_endpoint'),(-1,'negative_endpoint')]:
                item[key]=stats(window[np.isclose(window.target,sign*.11,atol=1e-6)])
            data['windows'][str(lo)+'-'+str(hi)]=item
        if label=='gradual':
            data['all_targets']={str(round(float(target),3)):stats(group) for target,group in c.groupby('target')}
            # Full recorded attempts have exact 58-attempt cyclic schedule.
            expected=np.r_[np.repeat(.11,20),np.arange(4,-5,-1)*.022,
                           np.repeat(-.11,20),np.arange(-4,5)*.022]
            np.testing.assert_allclose(c.target.to_numpy(),np.resize(expected,len(c)),atol=1e-7)
            grouped=frame.groupby((frame.wz_cmd.diff().abs()>1e-6).cumsum())
            transitions=[]
            previous_target=None
            for _,group in grouped:
                target=float(group.wz_cmd.iloc[0]); start=int(group.step.iloc[0])
                first=c[c.start==start]
                if (len(first) and previous_target is not None and abs(previous_target)<1e-6
                        and np.isclose(abs(target),.022,atol=1e-6)):
                    row=first.iloc[0].to_dict()
                    row['sign_correct']=bool(row['net_rate']*target>0)
                    transitions.append(row)
                previous_target=target
            data['first_small_command']=transitions
            data['cross_zero_first_small_sign_correct']=dict(
                positive=sum(x['sign_correct'] for x in transitions if x['target']>0),
                positive_count=sum(x['target']>0 for x in transitions),
                negative=sum(x['sign_correct'] for x in transitions if x['target']<0),
                negative_count=sum(x['target']<0 for x in transitions))
            meta=json.loads((folder/'best_reward_cycle/metadata.json').read_text())
            assert hashlib.sha256(Path(meta['checkpoint']).read_bytes()).hexdigest()==meta['checkpoint_sha256']
            data['best']=dict(metadata=meta,cycle=c[c.end==meta['end_step']].to_dict('records'))
        result[label]=data
    def finite(value):
        if isinstance(value,dict): return {k:finite(v) for k,v in value.items()}
        if isinstance(value,list): return [finite(v) for v in value]
        return None if isinstance(value,float) and not np.isfinite(value) else value
    (OUT/'summary.json').write_text(json.dumps(finite(result),indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(result['gradual']['windows'],indent=2))
    print('TARGETS',json.dumps(result['gradual']['all_targets'],indent=2))
    print('CROSSZERO',result['gradual']['cross_zero_first_small_sign_correct'])
    print('RESULT',result['gradual']['result'],'cycles',result['gradual']['cycles'],'actor',result['gradual']['actor_updates'])
    print('BEST',result['gradual']['best'])


if __name__=='__main__':main()
