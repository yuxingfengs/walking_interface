"""Compare fixed-state lateral-command network sensitivity at30k and60k."""
import gzip
import json
from pathlib import Path
import sys
import torch

OUT=Path(__file__).resolve().parent
VARIANT=OUT.parent
WIDE=OUT.parents[3]
sys.path.insert(0,str(WIDE.parent/'runtime/training'))
from basewalk.learning import load_policy
from basewalk.network_inputs import actor_indices


def states(run,lo,hi):
    result=[];previous=None
    with gzip.open(run/'transition_trace.jsonl.gz','rt',encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line);source=row.get('source_observation',previous)
            if lo<=row['step']<=hi and row['step']%20==0 and source is not None:
                result.append(source)
            previous=row.get('next_observation')
    return torch.tensor(result,dtype=torch.float32)


old_run=VARIANT/'runs/train_20261121T052059_f38fce'
new_run=OUT/'runs/train_20261121T060052_2cbf68'
batches=dict(original_30k_states=states(old_run,23323,25993),latest_60k_states=states(new_run,56829,59498))
report=dict(optimizer_updates=0,simulation_started=False,models={})
for label,path in [('30k',old_run/'checkpoints/step_00030000.pth'),('60k',new_run/'checkpoints/step_00060000.pth')]:
    actor,cfg,_=load_policy(path)
    idx=actor_indices(cfg).index(6)
    item=dict(vy_column_norm=float(actor.network[0].weight[:,idx].norm()),batches={})
    with torch.no_grad():
        for key,obs in batches.items():
            a=obs.clone();b=obs.clone();a[:,6]=-.02;b[:,6]=.02
            delta=(actor(a)-actor(b)).abs()
            item['batches'][key]=dict(states=len(obs),max_action_difference=float(delta.max()),
                mean_action_difference=float(delta.mean()),max_joint_target_difference_rad=float(delta.max())*cfg['environment']['action_scale'])
    report['models'][label]=item
(OUT/'policy_sensitivity.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,indent=2))
