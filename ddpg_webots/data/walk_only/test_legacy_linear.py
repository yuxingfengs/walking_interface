"""Reward and actual runner cadence checks; no physics or optimizer steps."""
import copy
import csv
import json
import math
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
from test_reset_flow import OUT, SyntheticEnv
from project import training_configuration
from basewalk.configuration import training_config
from basewalk import runner
from basewalk.learning import create_algorithm
from basewalk.offline_walk_critic import initialize
import reward


def main():
    cfg=training_config(training_configuration(OUT/'config.json'))
    blocked=False
    try:create_algorithm(cfg)
    except ValueError as exc:
        blocked='not pretrained' in str(exc)
    assert blocked, 'Online startup must require a pretrained Critic'
    cfg.pop('offline_walk_critic')
    for vx in (-.04,0.,.04,.08,.12,.16):
        for vy in (-.05,0.,.025,.05):
            value,terms=reward.motion_reward([.08,0,0],[vx,vy,0],[0,0],[0,0,-1],2,cfg)
            expected=1.25*(1-abs(vx-.08)/.08)+.5*(math.exp(-(vy/.05)**2)-1)
            assert abs(value-expected)<1e-10
            assert abs(terms['tracking_xy']-terms['tracking_x']-terms['tracking_y'])<1e-10
    cfg['environment']['reset_mode']='node'
    cfg['training'].update(total_env_steps=220,learning_starts=1,batch_size=1,checkpoint_every=220,save_best_reward_cycle=False)
    agent=create_algorithm(cfg);calls=[]
    before=copy.deepcopy(agent.actor.state_dict())
    def update(critic_only=False,update_actor=True):
        calls.append(update_actor);agent.updates+=1
        return dict(updates=agent.updates,actor_loss=None,critic_loss=None)
    with tempfile.TemporaryDirectory(prefix='cadence_',dir=OUT/'checks') as temp:
        folder=Path(temp)
        with patch.object(runner,'create_algorithm',return_value=agent),patch.object(runner,'emit'), \
             patch.object(agent,'update',side_effect=update),patch.object(agent.replay,'ready',return_value=True):
            runner.run_train(SyntheticEnv(cfg),cfg,folder,False)
        with (folder/'training.csv').open(newline='') as f:rows=list(csv.DictReader(f))
        attempts=0;expected=[]
        # SyntheticEnv advances a 40ms phase and resets on the two falls and one time limit.
        phase=0.;expected_steps=[]
        for step in range(1,221):
            after=(phase+.04*.6366197723675814)%1
            fell=step in (95,155)
            if after<phase or fell:
                attempts+=1
                if attempts%2==0:expected.append(attempts%4==0);expected_steps.append(step)
            phase=0. if fell or step==160 else after
        actual_steps=[int(row['step']) for row in rows if row['updates'] and
                      int(row['updates'])>(int(rows[int(row['step'])-2]['updates']) if int(row['step'])>1 else 0)]
        assert calls==expected,(calls,expected)
        assert actual_steps==expected_steps,(actual_steps,expected_steps)
    import torch
    assert all(torch.equal(v,agent.actor.state_dict()[k]) for k,v in before.items())
    report=dict(reward_cases=24,critic_update_steps=expected_steps,actor_update_flags=calls,
                attempts=attempts,fall_attempts=2,optimizer_updates=0,physics_simulation=False,
                online_start_blocked_until_pretrained=blocked)
    path=OUT/'variants/legacy_linear_walk/checks.json'
    path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
