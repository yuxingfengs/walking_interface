"""Forward variable-speed command ablation, preserving the Walk-only objective."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

OUT=Path(__file__).resolve().parent
WALK=OUT.parents[1]
sys.path.insert(0,str(WALK.parent/'runtime'))
sys.path.insert(0,str(WALK.parent/'runtime/training'))
from project import training_configuration
from basewalk.configuration import training_config
from basewalk.walk_commands import WalkCommands


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    path=WALK/'config.json'
    before=json.loads(path.read_text(encoding='utf-8'))
    if (OUT/'previous_config.json').exists():
        raise ValueError('Variant already prepared; refusing to overwrite its predecessor configuration')
    (OUT/'previous_config.json').write_text(path.read_text(encoding='utf-8'),encoding='utf-8')
    doc=copy.deepcopy(before)
    commands=[[v,0.] for v in [.02,.04,.06,.08,.06,.04]]
    doc['training_config']['training']['walk_commands']=dict(commands_m_s=commands,hold_steps=100)
    doc.update(name='Walk Only - forward variable speed',status='forward_speed_prepared',
        runs_root='variants/forward_speed/runs',reports_root='variants/forward_speed/reports')
    path.write_text(json.dumps(doc,indent=2)+'\n',encoding='utf-8')
    cfg=training_config(training_configuration(path))
    stage=WalkCommands(cfg)
    for step in range(1800):
        np.testing.assert_allclose(stage.command(0,0),[*commands[(step//100)%6],0],atol=1e-8)
        assert stage.command(0,0)[0]>0
        assert (stage.command(0,0)[1:]==0).all()
        stage.advance_phase(.9,.1 if step%2 else .95,fell=step%17==0)
        stage=WalkCommands(cfg,state=stage.state())
    assert stage.step==1800
    old=before['training_config'];new=doc['training_config']
    expected=copy.deepcopy(old)
    expected['training']['walk_commands']=new['training']['walk_commands']
    assert expected==new
    assert new['training']['total_env_steps']==20000
    assert 'checkpoint_start' not in new['training']
    assert new['training']['online_fifo_only'] and 'three_class_online' not in new
    actor=WALK/new['initialization']['bc_checkpoint']
    assert sha(actor)==new['initialization']['bc_sha256']
    report=dict(command_steps_checked=1800,commands=commands,hold_steps=100,
        pure_positive_forward=True,clock_survives_falls_and_reload=True,
        training_config_diff_only_walk_commands=True,
        reward_source_sha256=sha(WALK/'reward.py'),initial_actor_sha256=sha(actor),
        total_env_steps=20000,critic_initialization='same_seed_random',online_replay_initial_count=0,
        expert_replay_count=0,optimizer_updates=0,simulation_started=False,
        reuse_decision='Reuse identical original BC Actor. New online Replay starts empty; no old expert/rollout Replay, collection, relabeling or Critic pretraining required.',
        predecessor='train_20261121T074928_17fb51',predecessor_falls=538,
        reference_source='Existing WalkCommands command-conditioned step-clock design; engineering task simplification ablation.')
    (OUT/'offline_check.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    (OUT/'prepared_project_config.json').write_text(path.read_text(),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
