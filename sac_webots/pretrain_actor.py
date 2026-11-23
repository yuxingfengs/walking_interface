"""Fit SAC deterministic actions to expert demonstrations, grouped by episode."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
CODE = Path(__file__).resolve().parent
from project import ROOT, training_configuration, digest
sys.path.insert(0, str(CODE/'training'))
from basewalk.sac import SquashedGaussianActor
from basewalk.network_inputs import ACTION_KEEP, observation_dim


def fit(source, walk_only=False):
    if walk_only:
        sys.path.insert(0, str(CODE.parent/'sac_isaac_lab/runner'))
        from configuration import load_configuration
        cfg = load_configuration(experiment='walk_only', for_bc_fit=True)
    else:
        cfg = training_configuration_for_fit()
    data = torch.load(source, map_location='cpu', weights_only=False)
    if data['config']['actor_observation'] != cfg['actor_observation']:
        raise ValueError('Observation projection mismatch')
    groups = []
    if not walk_only:
        target = cfg['training']['online_fixed_turn_sign'] * cfg['training']['online_turn_rate_rad_s']
    for idx, ep in enumerate(data['episodes']):
        a = ep['arrays']
        obs = np.asarray(a['observations'])
        modes = np.asarray(a['modes']).reshape(-1)
        if walk_only:
            mask = (modes == 2) & (obs[:,5] > 0) & (obs[:,6] == 0) & (obs[:,7] == 0)
        else:
            mask = (modes == 0) & np.isclose(obs[:,7], target)
        if mask.any():
            x, y = obs[mask], np.asarray(a['executed_actions'])[mask]
            if (x.shape[1] != observation_dim(cfg) or y.shape[1] != 12 or
                    not np.isfinite(x).all() or not np.isfinite(y).all() or
                    not np.allclose(y[:,0],y[:,6]) or abs(y).max()>1.00001):
                raise ValueError('Invalid demonstration contract')
            groups.append((idx,x,y,np.asarray(a['next_observations'])[mask,50]))
    if len(groups)<2:
        raise ValueError('Need separate training and validation episodes')
    # Hold out one long episode so validation includes a complete gait cycle.
    holdout=max(groups,key=lambda group:len(group[1]))[0]
    def tensors(selected):
        return tuple(torch.tensor(np.concatenate([g[i] for g in selected]),dtype=torch.float32) for i in (1,2))
    x,y=tensors([g for g in groups if g[0]!=holdout])
    vx,vy=tensors([g for g in groups if g[0]==holdout])
    torch.manual_seed(cfg['seed'])
    torch.set_num_threads(1)
    actor=SquashedGaussianActor(cfg,cfg['sac'])
    if walk_only:
        from basewalk.network_inputs import actor_indices
        with torch.no_grad():
            actor.backbone[0].weight[:, actor_indices(cfg).index(6)].zero_()
    torch.nn.init.zeros_(actor.log_std.weight)
    torch.nn.init.constant_(actor.log_std.bias,np.log(.05))
    def loss(o,a):
        return (actor(o)[:,ACTION_KEEP]-a[:,ACTION_KEEP]).square().mean()
    optimizer=torch.optim.Adam(list(actor.backbone.parameters())+list(actor.mean.parameters()),lr=.0005)
    with torch.no_grad(): initial=loss(vx,vy).sqrt().item()
    best=initial
    out=Path(cfg['project_layout']['experiment_folder'])/'bc'
    out.mkdir(exist_ok=True)
    for epoch in range(120):
        for ix in torch.randperm(len(x)).split(64):
            optimizer.zero_grad()
            loss(x[ix],y[ix]).backward()
            optimizer.step()
        with torch.no_grad(): current=loss(vx,vy).sqrt().item()
        if current<best:
            best=current
            best_epoch=epoch+1
            state={k:v.detach().clone() for k,v in actor.state_dict().items()}
    if best>=initial:
        raise RuntimeError('No held-out improvement')
    tmp=out/'actor.tmp'
    torch.save(state,tmp)
    checksum=digest(tmp)
    name='actor_'+checksum[:16]+'.pth'
    tmp.replace(out/name)
    all_obs=np.concatenate([g[1] for g in groups])
    phase=np.mod(np.arctan2(all_obs[:,9],all_obs[:,8])/(2*np.pi),1.)
    record=dict(checkpoint=name,checkpoint_sha256=checksum,dataset=str(source.resolve()),dataset_sha256=digest(source),
        source_samples=len(all_obs),episode_counts={str(g[0]):len(g[1]) for g in groups},validation_episode=holdout,
        training_samples=len(x),validation_samples=len(vx),phase_bins_occupied=int(np.count_nonzero(np.histogram(phase, bins=10,range=(0,1))[0])),
        mean_gyro_z=float(np.mean(np.concatenate([g[3] for g in groups]))),initial_validation_rmse=initial,
        best_validation_rmse=best,selected_epoch=best_epoch,pre_tanh_std=.05,seed=cfg['seed'])
    if walk_only:
        record.update(network_inputs=cfg['network_inputs'], actor_input_dim=actor.backbone[0].in_features,
                      commands=np.unique(all_obs[:,5:8],axis=0).tolist(), task='forward_walk')
    tmp=out/'manifest.tmp'
    tmp.write_text(json.dumps(record,indent=2),encoding='utf-8')
    tmp.replace(out/'manifest.json')
    print(json.dumps(record,indent=2))


def training_configuration_for_fit():
    return training_configuration(load_bc=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('source',type=Path)
    parser.add_argument('--walk-only', action='store_true')
    args = parser.parse_args()
    fit(args.source, args.walk_only)
