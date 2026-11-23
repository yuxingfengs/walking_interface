"""Fit a fresh 50-input Actor to recorded expert Actions; no physics or RL."""
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import uuid
import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'runtime'))
sys.path.insert(0,str(ROOT.parent/'runtime/training'))
from project import training_configuration
from basewalk.configuration import training_config
from basewalk.learning import create_actor, load_policy
from basewalk.core import CONTRACT
from basewalk.network_inputs import actor_indices, ACTION_KEEP, actor_input


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    config_path=ROOT/'config.json';config_hash=digest(config_path)
    cfg=training_config(training_configuration(config_path))
    source=ROOT.parents[1]/'expert/step_wide_gaussian/turn_and_walk/expert.pth'
    data=torch.load(source,map_location='cpu',weights_only=False)
    if data['contract']!=CONTRACT:raise ValueError('Expert State contract mismatch')
    for key in ('physical_gait','support_observation','actor_observation','network_inputs'):
        if data['config'].get(key)!=cfg.get(key):raise ValueError('Expert contract mismatch: '+key)
    for key in ('action_scale','joint_margin','joint_velocity_scale','yaw_rate_scale','motor_velocity_scale'):
        if data['config']['environment'][key]!=cfg['environment'][key]:
            raise ValueError('Expert Action/sensor mismatch: '+key)
    episodes=data['episodes'];held_out=[1,7]
    # Whole recorded trajectories are held out, never adjacent random transitions.
    train=[];valid=[];manifest=[]
    for index,episode in enumerate(episodes):
        a=episode['arrays'];obs=np.asarray(a['observations'],dtype=np.float32)
        actions=np.asarray(a['executed_actions'],dtype=np.float32)
        modes=np.asarray(a['modes'])
        if (obs.shape!=(len(actions),55) or actions.shape[1:]!=(12,) or
                not np.isfinite(obs).all() or not np.isfinite(actions).all() or np.any(a['terminals'])):
            raise ValueError('Invalid or fallen expert episode')
        if not np.allclose(actions[:,0],actions[:,6],atol=1e-7):raise ValueError('Uncoupled expert Actions')
        pair=(torch.from_numpy(obs.copy()),torch.from_numpy(actions.copy()))
        (valid if index in held_out else train).append(pair)
        manifest.append(dict(index=index,split='validation' if index in held_out else 'train',
                             rows=len(obs),raw_sha256=episode['raw_sha256'],category=episode['category'],
                             modes=np.unique(modes).tolist(),commands=np.unique(obs[:,5:8],axis=0).tolist()))
    if {m['raw_sha256'] for m in manifest if m['split']=='train'} & {m['raw_sha256'] for m in manifest if m['split']=='validation'}:
        raise ValueError('Raw trajectory leaked between training and validation')
    x,y=(torch.cat([p[i] for p in train]) for i in (0,1))
    vx,vy=(torch.cat([p[i] for p in valid]) for i in (0,1))
    torch.set_num_threads(1);torch.manual_seed(cfg['seed'])
    old_path=Path(cfg['initialization']['bc_checkpoint'])
    if digest(old_path)!=cfg['initialization']['bc_sha256']:raise ValueError('Previous Actor hash mismatch')
    old,_,_=load_policy(old_path)
    model=create_actor(cfg)
    column=actor_indices(cfg).index(6)
    # No nonzero vy demonstrations exist; zero sensitivity is explicit, not arbitrary random weights.
    with torch.no_grad():model.network[0].weight[:,column].zero_()
    actor_input(x,cfg);actor_input(vx,cfg)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    loss_fn=nn.MSELoss();keep=list(ACTION_KEEP)
    def score(policy,inputs,labels):
        with torch.no_grad():
            total=0.
            for start in range(0,len(inputs),512):
                pred=policy(inputs[start:start+512])[:,keep]
                total+=float(((pred-labels[start:start+512,keep])**2).sum())
            return total/(len(inputs)*len(keep))
    baseline=score(old,vx,vy);initial=score(model,vx,vy)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:6]
    folder=ROOT/'variants/legacy_linear_bc50/bc'/('fit_'+stamp);folder.mkdir(parents=True)
    source_info=dict(path=str(source),sha256=digest(source),generation_policy=data.get('generation_policy'),
                     source_actor_sha256=data.get('actor_sha256'),episodes=manifest,commands_relabelled=False)
    plan=dict(source=source_info,seed=cfg['seed'],fresh_network=True,batch_size=256,lr=1e-4,
              max_epochs=200,patience=30,validation_episode_indices=held_out,
              actor_input_dim=len(actor_indices(cfg)),loss='MSE on 11 independent executed Actions',
              note='Supervised BC learning rate is separate from online Actor lr5e-7; no nonzero vy coverage.')
    (folder/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    (folder/'effective_config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    best=float('inf');best_state=None;best_epoch=0;updates=0
    generator=torch.Generator().manual_seed(cfg['seed']+1)
    with (folder/'metrics.jsonl').open('w',encoding='utf-8') as stream:
        for epoch in range(1,201):
            model.train();order=torch.randperm(len(x),generator=generator)
            for start in range(0,len(x),256):
                idx=order[start:start+256];optimizer.zero_grad()
                loss=loss_fn(model(x[idx])[:,keep],y[idx][:,keep])
                if not torch.isfinite(loss):raise ValueError('Nonfinite BC loss')
                loss.backward();optimizer.step();updates+=1
            model.eval();validation=score(model,vx,vy)
            if validation<best:
                best=validation;best_state=copy.deepcopy(model.state_dict());best_epoch=epoch
            row=dict(epoch=epoch,updates=updates,validation_mse=validation,best_validation_mse=best)
            stream.write(json.dumps(row)+'\n');stream.flush()
            if epoch==1 or epoch%10==0:print(json.dumps(row),flush=True)
            if epoch-best_epoch>=30:break
    model.load_state_dict(best_state);model.eval()
    final_train=score(model,x,y);final_val=score(model,vx,vy)
    report=dict(plan=plan,train_rows=len(x),validation_rows=len(vx),epochs_completed=epoch,
                best_epoch=best_epoch,bc_optimizer_updates=updates,initial_validation_mse=initial,
                previous_actor_validation_mse=baseline,validation_mse=final_val,train_mse=final_train,
                validation_action_rmse=float(np.sqrt(final_val)),
                validation_joint_rmse_rad=float(np.sqrt(final_val)*cfg['environment']['action_scale']),
                vy_column_norm=float(model.network[0].weight[:,column].norm()),
                physics_evaluation=False,critic_pretraining_started=False,online_training_started=False)
    checkpoint=folder/'actor.pth'
    torch.save(dict(contract=CONTRACT,config=cfg,actor=model.state_dict(),env_steps=0,
                    checkpoint_kind='expert_bc_retrained_50d_v1',training_source=source_info,
                    bc_report=report,actor_contract=dict(input_dim=50,observation_indices=list(actor_indices(cfg)),
                                                       actor_observation=cfg['actor_observation'])),checkpoint)
    policy,_,_=load_policy(checkpoint)
    assert torch.equal(policy(vx[:16]),model(vx[:16]))
    report.update(checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint))
    (folder/'result.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    if digest(config_path)!=config_hash:raise ValueError('Config changed during BC; checkpoint preserved, not registered')
    if final_val>baseline:
        raise ValueError('Fresh BC held-out error worse than previous Actor; weights kept but not registered')
    document=json.loads(config_path.read_text(encoding='utf-8-sig'))
    init=document['training_config']['initialization']
    init.update(bc_checkpoint=str(checkpoint),bc_sha256=digest(checkpoint),
                validation_status='bc_held_out_checked_physics_evaluation_pending')
    document['origin']['actor_source']=str(checkpoint);document['origin']['actor_sha256']=digest(checkpoint)
    document['status']='bc_retrained_not_critic_pretrained'
    temp=config_path.with_suffix('.tmp');temp.write_text(json.dumps(document,indent=2),encoding='utf-8');temp.replace(config_path)
    (ROOT/'variants/legacy_linear_bc50/bc_result.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('BC completed:',checkpoint,'validation MSE',final_val,'previous',baseline,flush=True)


if __name__=='__main__':main()
