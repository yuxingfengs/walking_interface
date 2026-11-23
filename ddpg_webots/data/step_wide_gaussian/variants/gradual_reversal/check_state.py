"""Verify zero-rate Turn reads the same causal shared EMA as its Reward mode."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

OUT = Path(__file__).resolve().parent
sys.path.insert(0,str(OUT.parents[1].parent/'runtime'))
sys.path.insert(0,str(OUT.parents[1].parent/'runtime/training'))
from project import training_configuration
from basewalk.configuration import training_config
from basewalk.environment import WebotsBaseWalk

cfg = training_config(training_configuration(OUT.parents[1]/'config.json'))
command = np.zeros(3,dtype=np.float32)
sample = SimpleNamespace(value=np.array([0.,0.,.031]),key=(0,tuple(command.tolist())))
env = SimpleNamespace(config=cfg,cfg=cfg['environment'],_command=lambda:command,
    gravity=np.array([0.,0.,-1.]),gyro_value=np.zeros(3),phase=.2,q=np.zeros(12),
    qvel=np.zeros(12),last_action=np.zeros(12),velocity=np.zeros(3),
    heading_reference=0.,yaw=0.,support_feedback=SimpleNamespace(value=0.),
    actor_body_gyro=True,actor_gyro_z=.04,reward_velocity_filter=sample,turn_yaw_filter=sample)
with patch('basewalk.environment.observation',return_value=np.zeros(50)):
    state = WebotsBaseWalk._observation(env)
assert len(state)==55
assert abs(state[51]-.031*cfg['environment']['yaw_rate_scale']) < 1e-8
assert state[52] == 1.
(OUT/'state_check.json').write_text(json.dumps(dict(zero_turn_ema_preserved=True,
    shared_state_width=55,optimizer_updates=0),indent=2)+'\n',encoding='utf-8')
print('Zero-rate Turn shared State EMA check passed; no simulation or optimizer updates.')
