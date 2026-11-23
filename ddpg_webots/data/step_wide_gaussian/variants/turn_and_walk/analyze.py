"""Offline mixed-command result analysis from saved causal transition logs."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
RUN=OUT/'runs/train_20261120T041310_2de8cb'


def main(out=None, run=None, lateral_vx=0., lateral_levels=None, start_attempt=0, windows=None):
    global OUT, RUN
    if out is not None: OUT=Path(out)
    if run is not None: RUN=Path(run)
    if lateral_levels is None: lateral_levels=[0,.01,.02,.03,.04,.03,.02,.01,0]
    frame=pd.read_csv(RUN/'training.csv')
    yaw=[.022,.044,.066,.088]+[.11]*20+[.088,.066,.044,.022,0,-.022,-.044,-.066,-.088]+[-.11]*20+[-.088,-.066,-.044,-.022,0,.022,.044,.066,.088]+[.066,.044,.022,0]
    schedule=[('turn',0.,0.,v) for v in yaw]
    schedule += [('forward',v,0.,0.) for v in [0,.02,.04,.06,.08,.06,.04,.02,0] for _ in range(4)]
    schedule += [('lateral',lateral_vx,v,0.) for v in lateral_levels for _ in range(4)]
    pending=[];rows=[];attempt=start_attempt
    for row in frame.to_dict('records'):
        pending.append(row)
        if not row['cycle_end']:continue
        label,vx,vy,wz=schedule[attempt%len(schedule)]
        for r in pending:
            np.testing.assert_allclose([r['vx_cmd'],r['vy_cmd'],r['wz_cmd']],[vx,vy,wz],atol=1e-7,
                err_msg=f"step={r['step']} attempt={attempt} stage={label}")
            assert r['motion_phase']==('turn' if label=='turn' else 'walk')
        dt=np.array([r['cycle_dt'] for r in pending]);duration=dt.sum()
        mean=lambda name:float(np.dot([r[name] for r in pending],dt)/duration)
        angle=np.array([r['yaw_delta'] for r in pending])
        actual_wz=angle.sum()/duration
        fell=any(r['fell'] for r in pending)
        rows.append(dict(start=pending[0]['step'],end=row['step'],attempt=attempt,stage=label,
            vx_cmd=vx,vy_cmd=vy,wz_cmd=wz,vx=mean('actual_vx'),vy=mean('actual_vy'),
            net_wz=actual_wz,gyro=mean('sensor_gyro_z'),vx_error=abs(mean('actual_vx')-vx),
            vy_error=abs(mean('actual_vy')-vy),yaw_error=abs(actual_wz-wz),
            reverse=float(np.maximum(-np.sign(wz)*angle,0).sum()) if abs(wz)>1e-6 else None,
            reward=mean('reward'),reward_x=mean('reward_tracking_x'),reward_y=mean('reward_tracking_y'),
            reward_yaw=mean('reward_tracking_yaw'),fell=fell,complete=bool(row['cycle_complete']) and not fell))
        pending=[]
        if fell or row['gait_phase_after'] < row['gait_phase_before']:
            attempt+=1
    cycles=pd.DataFrame(rows);cycles.to_csv(OUT/'cycles.csv',index=False)
    summary=dict(result=json.loads((RUN/'result.json').read_text()),attempts=attempt,attempts_this_run=attempt-start_attempt,
        full_cycles=int(cycles.complete.sum()),actor_updates=int(frame.actor_updated.sum()),windows={})
    def stats(selected):
        good=selected[selected.complete]
        return dict(cycles=len(good),falls=int(selected.fell.sum()),
            **{key:float(good[key].mean()) if len(good) else None for key in
               ['vx','vy','net_wz','gyro','vx_error','vy_error','yaw_error','reverse','reward','reward_x','reward_y','reward_yaw']})
    if windows is None: windows=[(1,5000),(5001,10000),(10001,15000),(15001,20000),(20001,25000),(25001,30000),(1,30000)]
    for lo,hi in windows:
        c=cycles[(cycles.start>=lo)&(cycles.end<=hi)]
        summary['windows'][str(lo)+'-'+str(hi)]={
            'turn_positive_endpoint':stats(c[(c.stage=='turn') & np.isclose(c.wz_cmd,.11)]),
            'turn_negative_endpoint':stats(c[(c.stage=='turn') & np.isclose(c.wz_cmd,-.11)]),
            'forward_maximum':stats(c[(c.stage=='forward') & np.isclose(c.vx_cmd,.08)]),
            'lateral_maximum':stats(c[(c.stage=='lateral') & np.isclose(c.vy_cmd,max(lateral_levels))]),
            'turn_all':stats(c[c.stage=='turn']),
            'forward_all':stats(c[c.stage=='forward']),
            'lateral_all':stats(c[c.stage=='lateral'])}
    summary['by_command']={}
    for (stage,vx,vy,wz),group in cycles.groupby(['stage','vx_cmd','vy_cmd','wz_cmd']):
        summary['by_command'][f'{stage}:{vx},{vy},{wz}']=stats(group)
    summary['falls']=cycles[cycles.fell][['start','end','stage','vx_cmd','vy_cmd','wz_cmd']].to_dict('records')
    meta=json.loads((RUN/'best_reward_cycle/metadata.json').read_text())
    assert hashlib.sha256(Path(meta['checkpoint']).read_bytes()).hexdigest()==meta['checkpoint_sha256']
    summary['best_turn']=dict(metadata=meta,cycle=cycles[cycles.end==meta['end_step']].to_dict('records'))
    def clean(value):
        if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
        if isinstance(value,list):return [clean(v) for v in value]
        return None if isinstance(value,float) and not np.isfinite(value) else value
    (OUT/'summary.json').write_text(json.dumps(clean(summary),indent=2)+'\n',encoding='utf-8')
    print('COUNTS',summary['result'],summary['full_cycles'],summary['attempts'],summary['actor_updates'])
    for key in ['1-5000','10001-15000','15001-20000','25001-30000']:
        if key in summary['windows']: print(key,json.dumps(summary['windows'][key],indent=2))
    print('FALLS',summary['falls'])
    print('BEST',summary['best_turn'])


if __name__=='__main__':main()
