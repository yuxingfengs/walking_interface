"""Frozen network-only vy-command sensitivity; no physics or learning."""
import gzip
import json
from pathlib import Path
import sys
import numpy as np
import torch

OUT=Path(__file__).resolve().parent
WIDE=OUT.parents[2]
sys.path.insert(0,str(WIDE.parent/'runtime/training'))
from basewalk.learning import load_policy
from basewalk.network_inputs import actor_indices

run=OUT/'runs/train_20261121T052059_f38fce'
path=run/'checkpoints/step_00030000.pth'
actor,cfg,_=load_policy(path)
states=[]
previous_next=None
with gzip.open(run/'transition_trace.jsonl.gz','rt',encoding='utf-8') as stream:
    for line in stream:
        row=json.loads(line)
        source=row.get('source_observation',previous_next)
        if 23323<=row['step']<=25993 and row['step']%20==0 and source is not None:
            states.append(source)
        previous_next=row.get('next_observation')
assert len(states)>0
obs=torch.tensor(states,dtype=torch.float32)
minus=obs.clone();plus=obs.clone()
minus[:,6]=-.02;plus[:,6]=.02
with torch.no_grad():
    delta=(actor(plus)-actor(minus)).abs()
indices=actor_indices(cfg)
weights=actor.network[0].weight.detach()
result=dict(checkpoint=str(path),states=len(states),fixed_observations_changing_only_vy=True,
    vy_minus=-.02,vy_plus=.02,max_action_change=float(delta.max()),
    mean_absolute_action_change=float(delta.mean()),
    max_joint_target_change_rad=float(delta.max())*cfg['environment']['action_scale'],
    first_layer_column_norm={label:float(weights[:,indices.index(column)].norm())
        for label,column in [('vx_cmd',5),('vy_cmd',6),('wz_cmd',7)]},
    optimizer_updates=0,simulation_started=False,
    interpretation='A controlled network sensitivity probe, not independent physical policy Evaluation.')
(OUT/'policy_sensitivity.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))
