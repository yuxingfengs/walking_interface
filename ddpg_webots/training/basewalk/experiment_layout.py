"""Parameter-scoped experiment folders and immutable pretraining runs."""
import copy
import hashlib
import json
from pathlib import Path

from . import TRAINING_ROOT as ROOT


def configure(cfg):
    if not cfg.get('experiment_scoped_assets'):
        return cfg
    base = cfg.get('experiment_base', cfg['experiment'])
    if cfg['experiment'] != base and not cfg['experiment'].startswith(base + '__'):
        base = cfg['experiment']
    identity = copy.deepcopy(cfg)
    for key in ('experiment', 'experiment_base', 'parameter_id', 'previous_experiment',
                'previous_training_run', 'monitoring', 'notifications'):
        identity.pop(key, None)
    for key in ('dataset', 'critic_checkpoint', 'mode'):
        identity['expert_pipeline'].pop(key, None)
    token = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    cfg.update(experiment_base=base, parameter_id=token, experiment=base+'__'+token)
    folder = ROOT/'experiments'/cfg['experiment']
    pipeline = cfg['expert_pipeline']
    pipeline['dataset'] = str(folder/'expert_data'/'successful.pth')
    pipeline['critic_checkpoint'] = str(folder/'pretrain'/'NOT_PRETRAINED.pth')
    pointer = folder/'pretrain'/'latest.json'
    if pointer.exists():
        record = json.loads(pointer.read_text(encoding='utf-8'))
        target = (pointer.parent/record['checkpoint']).resolve()
        try:
            target.relative_to(pointer.parent.resolve())
        except ValueError:
            raise ValueError('Pretraining pointer escapes experiment directory')
        if record['parameter_id'] != token:
            raise ValueError('Invalid experiment pretraining pointer')
        pipeline['critic_checkpoint'] = str(target)
    return cfg


def publish(cfg, output):
    folder = ROOT/'experiments'/cfg['experiment']/'pretrain'
    pointer = folder/'latest.json'
    data = dict(parameter_id=cfg['parameter_id'],
                checkpoint=str(output.resolve().relative_to(folder.resolve())))
    temporary = pointer.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2), encoding='utf-8')
    temporary.replace(pointer)
