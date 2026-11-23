"""Analyze saved Walk-only logs; no model or simulation updates."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
RUN=OUT/'runs/train_20261121T074928_17fb51'


def main():
    f=pd.read_csv(RUN/'training.csv')
    episodes=pd.read_csv(RUN/'training_episodes.csv')
    reference=json.loads((RUN/'effective_config.json').read_text())
    cfg=reference.get('config',reference)
    gamma=cfg['training']['gamma']
    returns=np.full(len(f),np.nan)
    for _,g in f.groupby('diagnostic_episode_id',sort=False):
        if not bool(g.fell.iloc[-1]):continue
        value=0.
        for idx in reversed(g.index.tolist()):
            value=float(f.at[idx,'reward'])+gamma*value
            returns[idx]=value
    f['realized_discounted_return_to_fall']=returns
    rows=[]
    for lo,hi in [(1,1000),(1001,5000),(5001,10000),(10001,15000),(15001,20000)]:
        s=f[f.step.between(lo,hi)]
        e=episodes[episodes.end_step.between(lo,hi)&episodes.fell]
        terminal=s[s.fell]
        rows.append(dict(start=lo,end=hi,falls=int(s.fell.sum()),reward=float(s.reward.mean()),
            planar=float(s.reward_tracking_xy.mean()),yaw=float(s.reward_tracking_yaw.mean()),
            tilt=float(s.reward_orientation.mean()),stability=float(s.reward_stability.mean()),
            fall_penalty=float(s.reward_fall.mean()),q=float(s.q_executed.mean()),
            realized_return=float(s.realized_discounted_return_to_fall.mean()),
            vx_mae=float((s.actual_vx-s.vx_cmd).abs().mean()),
            vy_mae=float((s.actual_vy-s.vy_cmd).abs().mean()),
            completed_fall_episode_mean_seconds=float(e.elapsed_s.mean()),
            completed_fall_episode_median_seconds=float(e.elapsed_s.median()),
            terminal_q=float(terminal.q_executed.mean()),terminal_reward=float(terminal.reward.mean())))
    blocks=[]
    targets=cfg['training']['walk_commands']['commands_m_s']
    for index,g in f.groupby((f.step-1)//100):
        np.testing.assert_allclose(g[['vx_cmd','vy_cmd']].mean(),targets[int(index)%16],atol=1e-7)
        blocks.append(dict(block=int(index),start=int(g.step.min()),end=int(g.step.max()),
            vx_cmd=float(g.vx_cmd.iloc[0]),vy_cmd=float(g.vy_cmd.iloc[0]),
            falls=int(g.fell.sum()),full=len(g)==100,reward=float(g.reward.mean()),
            vx=float(g.actual_vx.mean()),vy=float(g.actual_vy.mean()),
            heading_rate=float(g.yaw_delta.sum()/g.cycle_dt.sum())))
    pd.DataFrame(rows).to_csv(OUT/'window_metrics.csv',index=False)
    pd.DataFrame(blocks).to_csv(OUT/'command_blocks.csv',index=False)
    result=dict(run=str(RUN),result=json.loads((RUN/'result.json').read_text()),
        windows=rows,command_blocks=len(blocks),fall_free_blocks=sum(b['full'] and b['falls']==0 for b in blocks),
        fall_free_tail_blocks=sum(b['full'] and b['falls']==0 and b['start']>15000 for b in blocks),
        evaluation='Online exploratory run, not independent Evaluation',
        realized_return_note='Observed discounted return on the recorded changing exploratory policy; not exact Q truth for the final deterministic Actor.')
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
