"""Stronger differential-stride demonstrations and matched BC validation."""
import copy
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'controllers'), str(ROOT/'controllers/common'), str(ROOT/'scripts')]
OUT = ROOT/'experiments/walk_bc_strong_heading_70k/design'

def simulate(supervisor, cfg, out, gain, cap, steps):
    from configs.runtime import runtime_settings, task_settings
    from tasks import load_task
    from experts.strong_heading import StrongHeadingExpertPolicy
    from envs.nao_locomotion_env import NaoLocomotionEnv
    from webots_platform.webots_runtime import WebotsRuntime
    task = load_task(cfg['environment']['task_module'], cfg['environment']['task_class'], task_settings(cfg, 'eval'))
    env = NaoLocomotionEnv(WebotsRuntime(supervisor, **runtime_settings(cfg, 'eval')), task)
    env.initialize()
    obs = env.reset(cfg['environment']['settle_steps'])
    initial = np.array(env.last_raw_state.position)
    states, actions, positions, controls = [], [], [initial], []
    policy = StrongHeadingExpertPolicy(gain=gain, cap=cap)
    arrival = None
    fell = False
    for i in range(steps):
        a = policy.act(obs)
        states.append(obs.values); controls.append(policy.adjustment)
        step = env.step(a, i)
        actions.append(step.action.values)
        obs = step.next_observation
        positions.append(np.array(env.last_raw_state.position))
        if arrival is None and positions[-1][0] >= 5.:
            net = positions[-1]-initial
            arrival = dict(time_s=(i+1)*.04, forward_m=float(net[0]), lateral_m=float(net[1]), abs_lateral_m=float(abs(net[1])))
        fell = bool(step.info['fell'])
        if fell: break
    x = np.array(states, dtype=np.float32)
    heading = -np.arctan2(x[:,24], x[:,25])
    cycles = np.floor(np.unwrap(x[:,0]*2*math.pi)/(2*math.pi)).astype(int)
    means = [heading[cycles==c].mean() for c in np.unique(cycles)[1:-1]]
    result = dict(gain=gain, cap=cap, slew=.0005, initial_yaw_rad=cfg['webots']['initial_rotation'][3],
                  steps=len(states), fell=fell, reached=arrival is not None, arrival_x5=arrival,
                  first_cycles_deg=float(np.degrees(np.mean(means[:3]))), last_cycles_deg=float(np.degrees(np.mean(means[-3:]))),
                  forward_m=float(positions[-1][0]-initial[0]))
    np.savez_compressed(Path(out)/'trajectory.npz', states=x, actions=np.array(actions,dtype=np.float32), positions=positions, controls=controls)
    (Path(out)/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')

def trial(cfg, name, yaw, gain=.6, cap=.04, steps=3000, checkpoint=None):
    from webots_trials import prepare_driver, execute
    out = OUT/'trials'/name
    if (out/'result.json').exists(): return json.loads((out/'result.json').read_text())
    out.mkdir(parents=True, exist_ok=True)
    cfg = copy.deepcopy(cfg)
    cfg['experiment'].update(name=name, output_root=str(OUT/'trial_runs'))
    cfg['initialization'].update(use_bc=False,resume='never',reset_to_bc_on_failure=False)
    cfg['webots']['initial_rotation'] = [0,0,1,yaw]
    call = f'simulate(supervisor,{cfg!r},{str(out)!r},{gain!r},{cap!r},{steps!r})'
    module = 'build_strong_heading_bc'
    if checkpoint:
        module = 'evaluate_bc'
        call = f'simulate(supervisor,{cfg!r},{str(out)!r},{str(checkpoint)!r})'
    driver = f'''from controller import Supervisor
sys.path.insert(0,{str(ROOT/'scripts')!r})
from {module} import simulate
supervisor=Supervisor()
try:
    {call}
except BaseException:
    import traceback
    status.write_text(json.dumps(dict(status='failed',error=traceback.format_exc())),encoding='utf-8')
    supervisor.simulationQuit(1)
    raise
else:
    status.write_text(json.dumps(dict(status='completed')),encoding='utf-8')
    supervisor.simulationQuit(0)
'''
    execute(*prepare_driver(cfg,'train',driver))
    result = json.loads((out/'result.json').read_text())
    print(name, json.dumps(result), flush=True)
    return result

def fit():
    from models import BoundedActor
    from contracts import contract_metadata
    torch.set_num_threads(1); torch.manual_seed(1)
    train=np.load(OUT/'demonstrations27.npz'); valid=np.load(OUT/'validation27.npz')
    x,y=torch.from_numpy(train['states']),torch.from_numpy(train['actions'])
    vx,vy=torch.from_numpy(valid['states']),torch.from_numpy(valid['actions'])
    actor=BoundedActor(27,256,10)
    actor.load_state_dict(torch.load(OUT/'source_bc27.pth',map_location='cpu',weights_only=False)['actor_state_dict'])
    optimizer=torch.optim.Adam(actor.parameters(),lr=1e-4)
    loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x,y),batch_size=256,shuffle=True)
    best=float('inf'); history=[]
    for epoch in range(1,501):
        actor.train(); total=0.
        for bx,by in loader:
            loss=torch.nn.functional.mse_loss(actor(bx),by)
            optimizer.zero_grad();loss.backward();optimizer.step();total+=loss.item()*len(bx)
        actor.eval()
        with torch.no_grad(): mse=torch.nn.functional.mse_loss(actor(vx),vy).item()
        history.append(dict(epoch=epoch,train_mse=total/len(x),validation_mse=mse))
        if mse<best:
            best=mse
            meta=dict(**contract_metadata(state_dim=27),artifact_type='bc_actor',epochs=epoch,seed=1,samples=len(x),validation_mse=mse,learning_rate=1e-4,initialization='previous_bc27_full_finetune',dataset_sha256=hashlib.sha256((OUT/'demonstrations27.npz').read_bytes()).hexdigest())
            torch.save(dict(actor_state_dict=actor.state_dict(),meta=meta),OUT/'actor_bc_heading27.pth')
        if epoch%50==0: print(history[-1],flush=True)
    (OUT/'bc_loss.json').write_text(json.dumps(history,indent=2))

def main():
    from configs.runtime import load_config
    OUT.mkdir(parents=True,exist_ok=True)
    snapshot=OUT/'previous_config.json'
    if not snapshot.exists():
        cfg=load_config();snapshot.write_text(json.dumps(cfg,indent=2))
        shutil.copy2(ROOT/'configs/project_config.py',OUT/'previous_project_config.py')
        shutil.copy2(ROOT/cfg['initialization']['bc_checkpoint'],OUT/'source_bc27.pth')
    cfg=json.loads(snapshot.read_text())
    # Preserve the original experiment source snapshot; record the cleaned entry separately.
    shutil.copy2(__file__,OUT/'build_strong_heading_bc_current.py')
    strong=trial(cfg,'verify_strong',.0837)
    assert strong['reached'] and not strong['fell'], 'Strong expert did not reach safely'
    rng=np.random.default_rng(1); xs=[];ys=[];groups=[]
    for group,yaws in [('straight',[0]),('left',[3,6]),('right',[-3,-6])]:
        data=[]
        for yaw in yaws:
            name=f'train_{group}_{yaw}'
            result=trial(cfg,name,math.radians(yaw),gain=0. if group=='straight' else .6)
            assert not result['fell'] and result['steps']==3000
            data.append(np.load(OUT/'trials'/name/'trajectory.npz'))
        x=np.concatenate([d['states'] for d in data]);y=np.concatenate([d['actions'] for d in data])
        mandatory=np.concatenate([np.arange(i*3000,i*3000+80) for i in range(len(data))])
        rest=np.setdiff1d(np.arange(len(x)),mandatory)
        selected=np.r_[mandatory,rng.choice(rest,2960-len(mandatory),replace=False)]
        xs.append(x[selected]);ys.append(y[selected]);groups.extend([group]*2960)
    np.savez_compressed(OUT/'demonstrations27.npz',states=np.concatenate(xs),actions=np.concatenate(ys),groups=groups)
    xs=[];ys=[]
    for group,yaw in [('straight',.3),('left',4.5),('right',-4.5)]:
        name=f'valid_{group}'
        result=trial(cfg,name,math.radians(yaw),gain=0. if group=='straight' else .6,steps=1500)
        assert not result['fell'] and result['steps']==1500
        d=np.load(OUT/'trials'/name/'trajectory.npz');xs.append(d['states']);ys.append(d['actions'])
    np.savez_compressed(OUT/'validation27.npz',states=np.concatenate(xs),actions=np.concatenate(ys))
    fit()
    result=trial(cfg,'bc_baseline',.0837,checkpoint=OUT/'actor_bc_heading27.pth')
    assert result['reached'] and not result['fell'], 'BC physical validation failed'
    hashes={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file() and p.suffix in ('.npz','.pth','.py')}
    (OUT/'artifact_hashes.json').write_text(json.dumps(hashes,indent=2))
    print('READY_FOR_TRAINING',flush=True)

if __name__=='__main__': main()
