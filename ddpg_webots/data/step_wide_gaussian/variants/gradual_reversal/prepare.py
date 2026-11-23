"""Reuse unchanged physical data; derive Reward provenance for the zero-capable scorer."""
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

OUT = Path(__file__).resolve().parent
FOLDER = OUT.parents[1]
PROJECT = FOLDER.parents[1]
sys.path.insert(0, str(FOLDER.parent/'runtime'))
sys.path.insert(0, str(FOLDER.parent/'runtime/training'))
from project import training_configuration
from basewalk.configuration import training_config
from basewalk.reward_dispatch import implementation
from basewalk.gradual_turn import GradualTurnStages, source_motion_stage
from basewalk.fixed_cycle import FixedCycleStages
from basewalk.three_class_online import validate_online_experts
from basewalk.bc_rollout import compatible_actor_identities


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    cfg = training_config(training_configuration(FOLDER/'config.json'))
    scorer = implementation(cfg)
    spec = importlib.util.spec_from_file_location('old_reward', FOLDER/'reward.py')
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    constructor = dict(cycle_limit=20, count_fall_as_attempt=True,
                       turn_only=True, fixed_turn_sign=None)
    stage = GradualTurnStages(cfg, **constructor)
    one = np.r_[np.repeat(.11,20), np.arange(4,-5,-1)*.022,
                np.repeat(-.11,20), np.arange(-4,5)*.022]
    observed = []
    for i in range(len(one)*3):
        command = stage.command(0.,0.)
        observed.append(float(command[2]))
        before = stage.state()
        stage.advance_phase(.1,.2)
        assert stage.state() == before
        # Falls, normal cycle completion and simultaneous fall/wrap count once.
        stage.advance_phase(.9, .95 if i%3 == 0 else .1, fell=i%3 != 1)
        stage = GradualTurnStages(cfg, state=stage.state(), **constructor)
    np.testing.assert_allclose(observed, np.tile(one,3), atol=1e-8)
    assert max(abs(np.diff(observed))) < .022001
    assert source_motion_stage([0,0,0], cfg) == 'turn'
    legacy = copy.deepcopy(cfg)
    legacy['training'].pop('gradual_turn')
    assert source_motion_stage([0,0,0], legacy) == 'stand'
    assert FixedCycleStages(legacy, **constructor).command(0,0)[2] > 0
    for state in (dict(stage='turn',gradual_index=-1,cycles=0),
                  dict(stage='turn',gradual_index=1,cycles=1)):
        try:
            GradualTurnStages(cfg,state=state,**constructor)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid continuation accepted')
    for target in np.arange(-5,6)*.022:
        for velocity in [-.2,0.,.13]:
            value = scorer.motion_reward([0,0,target],[0,0,velocity],[0,0],[0,0,-1],0,cfg)[0]
            assert abs(value - 2*np.exp(-((velocity-target)/.7)**2)) < 1e-12
    expert_source = cfg['three_class_online']['dataset']
    rollout_source = cfg['bc_rollout']['source']['path']
    expert = torch.load(expert_source,map_location='cpu',weights_only=False)
    rollout = torch.load(rollout_source,map_location='cpu',weights_only=False)
    assert len(expert['episodes']) > 0
    maximum = 0.
    commands = []
    for episode in expert['episodes']:
        arrays = episode['arrays']
        before = old.revalue_arrays(arrays,cfg)
        after = scorer.revalue_arrays(arrays,cfg)
        maximum = max(maximum,float(np.max(np.abs(before-after))))
        np.testing.assert_allclose(after,episode['revalued_rewards'],rtol=1e-5,atol=1e-7)
        episode['revalued_rewards'] = after
        commands.extend(np.asarray(arrays['observations'])[np.asarray(arrays['modes'])==0,7].tolist())
    expert['config']['reward_source_sha256'] = cfg['reward_source_sha256']
    expert['derivation_gradual_reversal'] = dict(source=str(expert_source),sha256=digest(expert_source),
        method='Zero-capable Gaussian scorer; no physical transition or source command changed')
    expert_target = PROJECT/'expert/step_wide_gaussian/gradual_reversal/expert.pth'
    expert_target.parent.mkdir(parents=True,exist_ok=True)
    torch.save(expert,expert_target)
    validate_online_experts(expert,cfg)
    before = old.revalue_arrays(rollout['arrays'],cfg)
    after = scorer.revalue_arrays(rollout['arrays'],cfg)
    np.testing.assert_array_equal(before,after)
    # Scores are exactly unchanged on recorded commands, including old fall backfill.
    rollout['config']['reward_source_sha256'] = cfg['reward_source_sha256']
    rollout['reward_source_sha256'] = cfg['reward_source_sha256']
    assert rollout['identity'] == compatible_actor_identities(cfg)[rollout['actor_sha256']]
    rollout['derivation_gradual_reversal'] = dict(source=str(rollout_source),sha256=digest(rollout_source),
        method='Scorer extension verified equivalent on original commands; stored labels/fall residual unchanged')
    rollout_target = OUT/'pretrain/rollout.pth'
    rollout_target.parent.mkdir(parents=True,exist_ok=True)
    torch.save(rollout,rollout_target)
    path = FOLDER/'config.json'
    document = json.loads(path.read_text(encoding='utf-8'))
    active = document['training_config']
    active['three_class_online'].update(dataset='../../expert/step_wide_gaussian/gradual_reversal/expert.pth',
        dataset_sha256=digest(expert_target),critic_sha256='NOT_PRETRAINED')
    active['bc_rollout']['source'] = dict(path='variants/gradual_reversal/pretrain/rollout.pth',sha256=digest(rollout_target))
    document['status'] = 'gradual_reversal_offline_checked'
    path.write_text(json.dumps(document,indent=2)+'\n',encoding='utf-8')
    report = dict(schedule_attempts_checked=len(observed), attempts_per_loop=len(one),
        max_command_delta=.022, endpoint_rate=.110, zero_as_turn=True,
        state_restored_after_each_attempt=True, fall_counts_once=True,
        optimizer_updates=0, physical_data_changed=False, old_reward_max_difference=maximum,
        expert=dict(source=expert_source,source_sha256=digest(expert_source),
            derived=str(expert_target),sha256=digest(expert_target),transitions=20165,
            turn_positive=sum(x>1e-6 for x in commands),turn_negative=sum(x<-1e-6 for x in commands)),
        rollout=dict(source=rollout_source,source_sha256=digest(rollout_source),
                     derived=str(rollout_target),sha256=digest(rollout_target),transitions=20000),
        coverage_note='Old commands retained; exact intermediate/zero-target coverage not assumed. Online collects new staircase transitions.',
        decision='Reuse physical data with scorer provenance derivation; pretrain a new Critic, no collection or BC refit.')
    (OUT/'data_reuse.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__': main()
