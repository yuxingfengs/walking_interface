"""Prepare a single positive forward target without changing learning or reward."""
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
    if (OUT/'previous_config.json').exists():raise ValueError('Refusing to overwrite predecessor snapshot')
    before=json.loads(path.read_text(encoding='utf-8'))
    (OUT/'previous_config.json').write_text(path.read_text(),encoding='utf-8')
    doc=copy.deepcopy(before)
    doc['training_config']['training']['walk_commands']=dict(commands_m_s=[[.08,0.]],hold_steps=100)
    doc.update(name='Walk Only - fixed forward 0.08',status='fixed_forward_prepared',
        runs_root='variants/forward_fixed_008/runs',reports_root='variants/forward_fixed_008/reports')
    path.write_text(json.dumps(doc,indent=2)+'\n',encoding='utf-8')
    expected=copy.deepcopy(before['training_config'])
    expected['training']['walk_commands']=doc['training_config']['training']['walk_commands']
    assert expected==doc['training_config']
    cfg=training_config(training_configuration(path))
    assert cfg['training']['total_env_steps']==20000 and 'checkpoint_start' not in cfg['training']
    assert cfg['training']['online_fifo_only'] and 'three_class_online' not in cfg
    clock=WalkCommands(cfg)
    for step in range(600):
        np.testing.assert_allclose(clock.command(0,0),[.08,0,0],atol=1e-8)
        clock.advance_phase(.9,.1 if step%2 else .95,fell=step%13==0)
        clock=WalkCommands(cfg,state=clock.state())
    assert clock.step==600
    source=WALK/'variants/forward_speed/runs/train_20261121T230726_72d388'
    report=dict(command_steps_checked=600,fixed_command=[.08,0,0],
        training_config_diff_only_walk_commands=True,optimizer_updates=0,simulation_started=False,
        initial_actor_sha256=sha(cfg['initialization']['bc_checkpoint']),reward_source_sha256=sha(WALK/'reward.py'),
        initial_actor_same_as_previous=True,critic_initialization='same_seed_random',
        online_replay_initial_count=0,pretraining_required=False,total_env_steps=20000,
        predecessor_run=str(source),predecessor_status=json.loads((source/'status.json').read_text()),
        reuse_decision='Same original BC and unchanged physical/Reward interface. Fresh online FIFO; no collection, relabel or offline Critic pretraining.')
    (OUT/'offline_check.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    (OUT/'prepared_project_config.json').write_text(path.read_text(),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
