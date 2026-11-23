"""Exercise the actual runner across synthetic falls and reloads, with zero updates."""
import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import numpy as np

OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT.parent/'runtime'))
sys.path.insert(0,str(OUT.parent/'runtime/training'))
from project import training_configuration
from basewalk.configuration import training_config
from basewalk import runner
from basewalk.world_reload import WorldReloadRequested


class SyntheticEnv:
    def __init__(self,cfg,start=0):
        self.config=cfg;self.cfg=copy.deepcopy(cfg['environment']);self.steps=start
        self.dt=.04;self.phase=0.;self.position=np.zeros(3);self.yaw=0.
        self.gravity=np.array([0.,0.,-1.]);self.gyro_value=np.zeros(2)
        self.last=np.zeros(12,np.float32);self.resets=0

    def obs(self):
        o=np.zeros(55,np.float32);o[:3]=self.gravity;o[5:8]=self.stages.command(0,0)
        o[8:10]=[np.cos(2*np.pi*self.phase),np.sin(2*np.pi*self.phase)]
        o[34:46]=self.last
        return o

    def reset(self,command=None,randomize=False):
        self.resets+=1;self.phase=0.;self.position=np.zeros(3);self.start_position=self.position.copy()
        self.last=np.zeros(12,np.float32);self.stages=self.persistent_fixed_cycle_stages
        return self.obs()

    def step(self,action,randomize=False):
        command=self.stages.command(0,0);self.steps+=1
        fell=self.steps in (95,155);truncated=self.steps==160
        before=self.phase;self.phase=(self.phase+.04*.6366197723675814)%1
        self.last=np.asarray(action).copy();self.stages.advance_phase(before,self.phase,fell=fell)
        terms=dict(fall=-5. if fell else 0.,tracking_x=.3,tracking_y=.3,tracking_xy=.3,
            tracking_yaw=0.,orientation=0.,stability=0.,turn_weight=0.,walk_weight=1.,
            turn_weighted=0.,walk_weighted=.3,walk_total=.3)
        info=dict(executed_action=self.last,actual_velocity=np.zeros(3),velocity=np.zeros(3),
            reward_velocity=np.zeros(3),reward_command=command,command=command,
            position=self.position.copy(),yaw=0.,heading_error=0.,motion_phase='walk',
            replay_mode=2,replay_bucket='walk',reward_terms=terms,cycle_dt=.04,
            cycle_actual=[0.,0.,0.],cycle_target=(command*.04).tolist(),report_diagnostics={},next_stage='walk')
        return self.obs(),.3+terms['fall'],fell,truncated,info


def main(result_path=None):
    base=training_config(training_configuration(OUT.parents[1]/'experiments/walk_only/config.json'))
    # Synthetic reset tests do not load or create a trained Critic; startup is tested separately.
    base.pop('offline_walk_critic', None)
    checks=OUT/'checks';checks.mkdir(exist_ok=True)
    reports={}
    for mode in ('node','world_reload'):
        cfg=copy.deepcopy(base);cfg['environment']['reset_mode']=mode
        cfg['training'].update(total_env_steps=220,checkpoint_every=220,save_best_reward_cycle=False)
        saved={};restarts=0;attempts_at_reload=[]
        def capture(folder,config,agent,state, *, filename='world_reload_state.pth'):
            if filename.startswith('continuation_'):
                assert state['next_step'] == 221
                return
            assert state['sampler'] is None
            assert agent.updates==0
            attempts_at_reload.append(state.get('joint_attempts',0))
            saved.update(agent=agent,state=state)
            (Path(folder)/'world_reload_state.pth').write_bytes(b'synthetic-test-placeholder')
        with tempfile.TemporaryDirectory(prefix='reset_',dir=checks) as temp:
            folder=Path(temp).resolve();folder.relative_to(checks.resolve())
            with patch('basewalk.failure_retry.FailureRetry',side_effect=AssertionError('Walk Only used legacy angle retries')), \
                 patch.object(runner,'emit'), \
                 patch.object(runner,'save_continuation',side_effect=capture), \
                 patch.object(runner,'load_continuation',side_effect=lambda *_: saved or None):
                start=0
                while True:
                    env=SyntheticEnv(cfg,start)
                    try:
                        result=runner.run_train(env,cfg,folder,False)
                        break
                    except WorldReloadRequested:
                        restarts+=1
                        start=saved['state']['next_step']-1
                        assert saved['state']['fixed_cycle_state']['step']==start
                        assert restarts<=3
            with (folder/'training.csv').open(newline='') as f:rows=list(csv.DictReader(f))
            assert len(rows)==220 and result['falls']==2 and result['updates']==0
            commands=cfg['training']['walk_commands']['commands_m_s']
            for i,row in enumerate(rows):
                np.testing.assert_allclose([float(row['vx_cmd']),float(row['vy_cmd'])],commands[(i//100)%len(commands)],atol=1e-8)
            np.testing.assert_allclose([float(rows[95]['vx_cmd']),float(rows[95]['vy_cmd'])],commands[0],atol=1e-8)
            np.testing.assert_allclose([float(rows[100]['vx_cmd']),float(rows[100]['vy_cmd'])],commands[1%len(commands)],atol=1e-8)
            reports[mode]=dict(control_steps=220,synthetic_falls=[95,155],synthetic_time_limit=160,
                reloads=restarts,legacy_sampler_not_called=True,updates=0,command_clock_verified=True,
                attempts_at_reload=attempts_at_reload)
            if mode=='world_reload' and cfg['training'].get('joint_update_every_turn_attempts'):
                assert attempts_at_reload==[3,5,5],attempts_at_reload
    destination=Path(result_path) if result_path is not None else OUT/'reset_flow_check.json'
    destination.write_text(json.dumps(dict(tests=reports,physics_simulation=False,
        optimizer_updates=0,synthetic_adapter_only=True),indent=2)+'\n',encoding='utf-8')
    print(json.dumps(reports,indent=2))


if __name__=='__main__':main()
