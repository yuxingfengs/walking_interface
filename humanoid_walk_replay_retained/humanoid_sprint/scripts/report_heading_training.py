"""Read-only comparison of sampled training logs; no model evaluation implied."""
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CURRENT = 'walk_bc_strong_heading_70k'
PREVIOUS = 'walk_bc_sincos_rate_heading_70k'
OUT = ROOT/'experiments'/CURRENT/'analysis/training_heading'

def analyze(name):
    path=ROOT/'experiments'/name/'seed_001/metrics/train/metrics.csv'
    if name==PREVIOUS and not path.exists():
        path=OUT/'reference/previous_metrics.csv'
    rows=list(csv.DictReader(path.open(encoding='utf-8')))
    samples=[r for r in rows if r['type']=='progress' and r['heading_error_rad']]
    resets=[r for r in rows if r['event_name']=='reset']
    segments=[];start=0
    for r in resets:
        end=int(r['env_steps'])
        ss=[s for s in samples if start<int(s['env_steps'])<=end]
        angles=[abs(math.degrees(float(s['heading_error_rad']))) for s in ss]
        segments.append(dict(start_step=start,end_step=end,duration_s=(end-start)*.04,
            reason=r['reason'],fell=r['fell']=='True',reached=r['reached_goal']=='True',
            mean_sampled_abs_heading_deg=float(np.mean(angles)) if angles else None,
            terminal_abs_heading_deg=abs(math.degrees(float(r['heading_error_rad']))),
            terminal_world_x_m=float(r['x_value']),bc_reset=r['bc_reset_applied']=='True'))
        start=end
    bins=[]
    for lo,hi in [(0,10000),(10000,20000),(20000,35000),(35000,50000),(50000,60000),(60000,70001)]:
        ss=[s for s in samples if lo<=int(s['env_steps'])<hi]
        angles=[abs(math.degrees(float(s['heading_error_rad']))) for s in ss]
        bins.append(dict(start=lo,end=min(hi,70000),samples=len(ss),mean_abs_deg=float(np.mean(angles)),
                         median_abs_deg=float(np.median(angles))))
    late_goals=[s for s in segments if s['reached'] and s['start_step']>=35000]
    summary=dict(experiment=name,source=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        total_env_steps=int(rows[-1]['env_steps']),completed_segments=len(segments),
        reset_reasons=dict(Counter(s['reason'] for s in segments)),
        falls=sum(s['fell'] for s in segments),goals=sum(s['reached'] for s in segments),
        bc_resets=sum(s['bc_reset'] for s in segments),
        final_partial_segment_steps=int(rows[-1]['env_steps'])-start,
        late_successful_segments=len(late_goals),
        late_success_mean_duration_s=float(np.mean([s['duration_s'] for s in late_goals])),
        late_success_mean_terminal_abs_heading_deg=float(np.mean([s['terminal_abs_heading_deg'] for s in late_goals])),
        late_success_mean_sampled_abs_heading_deg=float(np.mean([s['mean_sampled_abs_heading_deg'] for s in late_goals])),
        bins=bins,segments=segments)
    return summary,samples

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    old,old_samples=analyze(PREVIOUS);new,new_samples=analyze(CURRENT)
    (OUT/'comparison.json').write_text(json.dumps(dict(previous=old,current=new),indent=2),encoding='utf-8')
    for summary in (old,new):
        with (OUT/(summary['experiment']+'_segments.csv')).open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(summary['segments'][0]));writer.writeheader();writer.writerows(summary['segments'])
    fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True,layout='constrained')
    for summary,samples,color,label in [(old,old_samples,'#888888','Previous BC initialization'),(new,new_samples,'#1479b8','Stronger BC initialization')]:
        # Non-overlapping 2000-step bins of logged instantaneous values.
        xx=[];yy=[]
        for lo in range(0,70000,2000):
            vals=[abs(math.degrees(float(r['heading_error_rad']))) for r in samples if lo<int(r['env_steps'])<=lo+2000]
            if vals:xx.append(lo+1000);yy.append(np.mean(vals))
        axes[0].plot(xx,yy,label=label,color=color)
        events=summary['segments']
        for reason,marker in [('goal','o'),('fall','x'),('consecutive_falls','x'),('stall_timeout_steps','|')]:
            selected=[s for s in events if s['reason']==reason]
            axes[1].scatter([s['end_step'] for s in selected],[1 if summary is new else 0]*len(selected),marker=marker,color=color)
    axes[0].set_ylabel('Mean sampled |heading error| (deg)');axes[0].legend();axes[0].grid(alpha=.25)
    axes[0].set_title('Training logs: changing policy + exploration noise; not fixed-checkpoint evaluation')
    axes[1].set_yticks([0,1],['Previous','Stronger']);axes[1].set_ylim(-.5,1.5)
    axes[1].set_title('Reset outcomes: circle = x6 reached; cross = fall; bar = stall')
    axes[1].set_xlabel('Environment steps');axes[1].grid(alpha=.25)
    fig.savefig(OUT/'heading_training.png',dpi=160);plt.close(fig)
    for s in (old,new):print(json.dumps({k:v for k,v in s.items() if k not in ['segments','bins']},indent=2))

if __name__=='__main__':main()
