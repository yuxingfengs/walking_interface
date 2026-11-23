"""Compare saved online logs using shared windows; no model updates."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
RUNS={'old':ROOT/'variants/legacy_linear_walk/runs/train_20261122T054807_362389',
      'new':OUT/'runs/train_20261122T063231_ae70f1'}

def load(path):
    f=pd.read_csv(path/'training.csv')
    cycles=[]
    groups=f.cycle_end.shift(1,fill_value=False).astype(int).cumsum()
    for _,g in f.groupby(groups,sort=False):
        if not bool(g.cycle_complete.iloc[-1]) or bool(g.fell.any()):continue
        dt=g.cycle_dt.sum();vx=float((g.actual_vx*g.cycle_dt).sum()/dt)
        cmd=float((g.vx_cmd*g.cycle_dt).sum()/dt)
        cycles.append(dict(start=int(g.step.iloc[0]),end=int(g.step.iloc[-1]),vx=vx,
                           cmd=cmd,error=abs(vx-cmd),duration=float(dt)))
    return f,pd.DataFrame(cycles)

def measure(f,c,lo,hi):
    s=f[f.step.between(lo,hi)];g=c[(c.start>=lo)&(c.end<=hi)]
    return dict(start=lo,end=hi,steps=len(s),falls=int(s.fell.sum()),
        complete_nonfallen_cycles=len(g),cycle_mean_vx=float(g.vx.mean()),
        cycle_mean_abs_target_error=float(g.error.mean()),
        near_target_cycles=int((g.error<=.01).sum()),
        all_step_mean_vx=float(s.actual_vx.mean()),
        critic_updates=int(s.updates.iloc[-1])-int(f[f.step<lo].updates.iloc[-1] if lo>1 else 0),
        actor_updates=int(s.actor_updated.fillna(False).sum()))

def main():
    loaded={k:load(p) for k,p in RUNS.items()};shared=min(int(f.step.max()) for f,c in loaded.values())
    windows=[(1,1000),(1001,5000),(5001,10000),(10001,shared),(shared-4999,shared),(1,shared)]
    result=dict(shared_last_step=shared,windows={},runs={})
    for k,(f,c) in loaded.items():
        result['runs'][k]=dict(run=str(RUNS[k]),last_step=int(f.step.max()),falls=int(f.fell.sum()),
                              complete_nonfallen_cycles=len(c),critic_updates=int(f.updates.iloc[-1]),
                              actor_updates=int(f.actor_updated.fillna(False).sum()))
        result['windows'][k]=[measure(f,c,lo,hi) for lo,hi in windows]
        c.to_csv(OUT/(k+'_complete_cycles.csv'),index=False)
    result['note']='Online exploratory policies, same target, Actor refit/lr/tau changed; not independent Evaluation.'
    f,c=loaded['new']
    if int(f.step.max())>=20000:
        result['new_full_windows']=[measure(f,c,a,b) for a,b in
            [(1,5000),(5001,10000),(10001,15000),(15001,20000),(18001,20000),(19001,20000),(1,20000)]]
    result['new_status']=json.loads((RUNS['new']/'status.json').read_text())
    if (RUNS['new']/'result.json').is_file():
        result['new_result']=json.loads((RUNS['new']/'result.json').read_text())
    result['first_fall_step']=int(f[f.fell].step.iloc[0]) if f.fell.any() else None
    (OUT/'comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
