"""Read-only command/data checks for forward motion with signed lateral targets."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

OUT=Path(__file__).resolve().parent
WIDE=OUT.parents[2]
sys.path.insert(0,str(WIDE.parent/'runtime'))
sys.path.insert(0,str(WIDE.parent/'runtime/training'))
from project import training_configuration
from basewalk.configuration import training_config
from basewalk.turn_and_walk import TurnAndWalkStages
from basewalk.gradual_turn import source_motion_stage
from basewalk.network_inputs import actor_input_dim,critic_input_dim


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    before=json.loads((OUT/'previous_config.json').read_text(encoding='utf-8'))
    now=json.loads((WIDE/'config.json').read_text(encoding='utf-8'))
    lateral=[0,.005,.010,.015,.020,.015,.010,.005,0,-.005,-.010,-.015,-.020,-.015,-.010,-.005,0]
    expected=copy.deepcopy(before)
    expected['training_config']['training']['turn_and_walk'].update(lateral_vx_m_s=.04,vy_levels_m_s=lateral)
    for key in ('name','status','runs_root','reports_root'):expected[key]=now[key]
    assert expected==now,'Unapproved configuration difference'
    cfg=training_config(training_configuration(WIDE/'config.json'))
    options=dict(count_fall_as_attempt=True,turn_only=False,fixed_turn_sign=None,cycle_limit=20)
    stage=TurnAndWalkStages(cfg,**options)
    yaw=[.022,.044,.066,.088]+[.11]*20+[.088,.066,.044,.022,0,-.022,-.044,-.066,-.088]+[-.11]*20+[-.088,-.066,-.044,-.022,0,.022,.044,.066,.088]+[.066,.044,.022,0]
    commands=[[0,0,v] for v in yaw]
    labels=['turn']*66
    commands += [[v,0,0] for v in [0,.02,.04,.06,.08,.06,.04,.02,0] for _ in range(4)]
    labels+=['forward']*36
    commands += [[.04,v,0] for v in lateral for _ in range(4)]
    labels+=['forward_lateral']*68
    assert len(commands)==170
    restored=[]
    for i in range(510):
        command=stage.command(0,0)
        np.testing.assert_allclose(command,commands[i%170],atol=1e-8)
        target_stage='turn' if labels[i%170]=='turn' else 'walk'
        assert stage.stage==target_stage
        assert source_motion_stage(command,cfg,stage.stage)==target_stage
        state=stage.state();stage.advance_phase(.1,.2);assert state==stage.state()
        # Time limits/reset alone do not advance the attempt; retain state.
        stage=TurnAndWalkStages(cfg,state=stage.state(),**options)
        stage.advance_phase(.9,.95 if i%3==0 else .1,fell=i%3!=1)
        stage=TurnAndWalkStages(cfg,state=stage.state(),**options)
        restored.append(dict(attempt=i,command=command.tolist(),stage=target_stage))
    array=np.array(commands)
    assert abs(array[:,1]).max()==.02
    assert np.abs(np.diff(np.r_[array[:,1],array[0,1]])).max()<.005001
    assert np.abs(np.diff(np.r_[array[:,2],array[0,2]])).max()<.022001
    assert (array[102:,0]==.04).all() and (array[102:,2]==0).all()
    for invalid in (dict(stage='walk',turn_walk_index=-1,cycles=0),
                    dict(stage='turn',turn_walk_index=0,cycles=1)):
        try:TurnAndWalkStages(cfg,state=invalid,**options)
        except ValueError:pass
        else:raise AssertionError('Invalid continuation accepted')
    for key,value in [('lateral_vx_m_s',.2),('vy_levels_m_s',[0,.11,0])]:
        bad=copy.deepcopy(cfg);bad['training']['turn_and_walk'][key]=value
        try:TurnAndWalkStages(bad,**options)
        except ValueError:pass
        else:raise AssertionError('Out-of-range command accepted')
    legacy=copy.deepcopy(cfg)
    legacy['training']['turn_and_walk']=before['training_config']['training']['turn_and_walk']
    old=TurnAndWalkStages(legacy,**options)
    assert sum(old.holds)==138
    assert sum(stage.holds)==170
    unchanged_count=len(old.commands)-len(legacy['training']['turn_and_walk']['vy_levels_m_s'])
    for i in range(unchanged_count):
        np.testing.assert_array_equal(old.commands[i],stage.commands[i])
        assert old.holds[i]==stage.holds[i]
    for i in range(unchanged_count,len(old.commands)):
        assert old.commands[i][0]==0 and old.commands[i][1]>=0
    sources={}
    for name,path in [('actor',cfg['initialization']['bc_checkpoint']),
                      ('critic',cfg['three_class_online']['critic']),
                      ('expert',cfg['three_class_online']['dataset']),
                      ('rollout',cfg['bc_rollout']['source']['path'])]:
        sources[name]=dict(path=path,sha256=sha(path))
    expert=torch.load(sources['expert']['path'],map_location='cpu',weights_only=False)
    observations=np.concatenate([e['arrays']['observations'] for e in expert['episodes']])
    rollout=torch.load(sources['rollout']['path'],map_location='cpu',weights_only=False)
    ro=np.asarray(rollout['arrays']['obs'])
    coverage=dict(expert_nonzero_vy=int((np.abs(observations[:,6])>1e-6).sum()),
                  rollout_nonzero_vy=int((np.abs(ro[:,6])>1e-6).sum()))
    report=dict(optimizer_updates=0,simulation_started=False,pretraining_started=False,
        config_only_lateral_schedule_and_output_metadata_changed=True,
        attempts_per_loop=170,turn_attempts=66,forward_attempts=36,forward_lateral_attempts=68,
        checked_attempts=510,vy_step=.005,vy_bounds=[-.02,.02],lateral_vx=.04,
        fall_and_wrap_count_once=True,reset_and_time_limit_preserve_progress=True,
        old_schedule_138_attempts_unchanged=True,actor_input=actor_input_dim(cfg),
        critic_input=critic_input_dim(cfg,0),total_env_steps=cfg['training']['total_env_steps'],
        sources=sources,coverage=coverage,
        reuse_decision='Same reward/state/action/physics: reuse original initial Actor, pretrained Critic, experts and frozen rollout unchanged. No relabel or pretraining required.',
        coverage_limitation='No nonzero vy commands in original experts/rollout. New paired vx/vy task coverage remains unvalidated; no data collection authorized.')
    assert report['actor_input']==50 and report['critic_input']==61
    (OUT/'offline_check.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    (OUT/'prepared_project_config.json').write_text(json.dumps(now,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
