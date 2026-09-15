"""Matched physical validation of the current 27D BC Actor."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch

def simulate(supervisor,cfg,out,checkpoint):
    from configs.runtime import runtime_settings,task_settings,create_actor
    from tasks import load_task
    from checkpointing import load_actor_weights
    from envs.nao_locomotion_env import NaoLocomotionEnv
    from webots_platform.webots_runtime import WebotsRuntime
    actor=create_actor(cfg);load_actor_weights(actor,checkpoint,torch.device('cpu'));actor.eval()
    task=load_task(cfg['environment']['task_module'],cfg['environment']['task_class'],task_settings(cfg,'eval'))
    env=NaoLocomotionEnv(WebotsRuntime(supervisor,**runtime_settings(cfg,'eval')),task)
    env.initialize();obs=env.reset(cfg['environment']['settle_steps']);initial=np.array(env.last_raw_state.position)
    states=[];positions=[initial];arrival=None;fell=False
    for i in range(3000):
        states.append(obs.values)
        with torch.no_grad(): chosen=actor(obs.values)[0].numpy()
        step=env.step(chosen,i);obs=step.next_observation;positions.append(np.array(env.last_raw_state.position))
        if arrival is None and positions[-1][0]>=5.:
            net=positions[-1]-initial
            arrival=dict(time_s=(i+1)*.04,forward_m=float(net[0]),lateral_m=float(net[1]),abs_lateral_m=float(abs(net[1])))
        fell=bool(step.info['fell'])
        if fell:break
    states.append(obs.values);a=np.array(states);net=positions[-1]-initial
    heading=-np.arctan2(a[:,24],a[:,25]);cycles=np.floor(np.unwrap(a[:,0]*2*math.pi)/(2*math.pi)).astype(int)
    means=[heading[cycles==c].mean() for c in np.unique(cycles)[1:-1]]
    result=dict(checkpoint=checkpoint,checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                steps=len(positions)-1,fell=fell,reached=arrival is not None,arrival_x5=arrival,
                forward_m=float(net[0]),lateral_m=float(net[1]),abs_lateral_m=float(abs(net[1])),
                first_cycles_deg=float(np.degrees(np.mean(means[:3]))),last_cycles_deg=float(np.degrees(np.mean(means[-3:]))),
                initial_yaw_rad=cfg['webots']['initial_rotation'][3])
    np.savez_compressed(Path(out)/'trajectory.npz',states=a,positions=np.array(positions))
    (Path(out)/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')

