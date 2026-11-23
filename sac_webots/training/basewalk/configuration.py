"""Resolve named BC without restoring old learning rates or training budgets."""
import copy
from pathlib import Path
import re
from .core import validate
from .learning import load_policy
from . import bc as bc_registry


def training_config(config):
    cfg = copy.deepcopy(config)
    if type(cfg.get('direction_training', {}).get('promote_success_to_bc', False)) is not bool:
        raise ValueError('direction_training.promote_success_to_bc must be boolean')
    if cfg['environment'].get('reset_mode', 'node') not in ('node', 'world_reload'):
        raise ValueError('environment.reset_mode must be node or world_reload')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,99}', cfg['experiment']):
        raise ValueError('Experiment name must contain only letters, digits, underscores or hyphens')
    init = cfg['initialization']
    enabled = init.get('use_bc', True)
    if type(enabled) is not bool:
        raise ValueError('initialization.use_bc must be boolean')
    name, candidate = init.get('bc_name'), init.get('bc_candidate')
    if not enabled:
        for key in ('bc_name','bc_candidate','candidate_name','bc_checkpoint','bc_sha256',
                    'bc_manifest_sha256','bc_evaluation_file','validation_status'):
            init.pop(key,None)
        init.update(use_bc=False,bc_checkpoint=None,reset_to_bc_on_failure=False)
    elif init.get('explicit_bc_artifact', False):
        import hashlib
        checkpoint = Path(init['bc_checkpoint'])
        if not checkpoint.is_file():
            raise FileNotFoundError('Missing 50D BC. Run 02 Train physical BC.cmd first: '+str(checkpoint))
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != init['bc_sha256']:
            raise ValueError('Explicit BC hash mismatch')
        _, saved, _ = load_policy(checkpoint)
        if not bc_registry.compatible_contract(saved, cfg, False):
            raise ValueError('Explicit BC configuration mismatch')
        init['validation_status'] = '50d_bc_independent_evaluation_pending'
    elif candidate:
        if name: raise ValueError('Choose bc_name OR bc_candidate')
        from .bc import load_bc_config, bundle_path
        bundle = bundle_path(candidate)
        saved = load_bc_config(candidate)  # Identity/hash verification; no training parameters copied.
        allow_yaw_change=bool(init.get('allow_yaw_source_change',False))
        if not bc_registry.compatible_contract(saved,cfg,allow_yaw_change):
            raise ValueError('BC candidate incompatible with requested State/Action/Actor configuration')
        init.update(use_bc=True,bc_checkpoint=saved['initialization']['bc_checkpoint'],
                    bc_sha256=saved['initialization']['bc_sha256'],
                    bc_manifest_sha256=bc_registry.digest(bundle/'manifest.json'),
                    bc_evaluation_file=str(bundle/('migration_report.json' if candidate=='film_rate_error_47d_v1' else 'evaluation.json')),
                    validation_status=saved['initialization']['validation_status'])
        if allow_yaw_change:
            init['yaw_source_migration']='BC evaluated with integrated_yaw_rate; supervisor_truth evaluation pending'
    else:
        raise ValueError('Select a registered bc_candidate or set use_bc=false')
    cfg = validate(cfg)
    monitor = cfg['monitoring']
    import math
    pause = monitor.get('direction_pause_seconds', 0.)
    if not isinstance(pause, (int, float)) or isinstance(pause, bool) or not math.isfinite(pause) or not 0 <= pause <= 60:
        raise ValueError('monitoring.direction_pause_seconds must be between 0 and 60')
    for key in ('refresh_seconds', 'stale_seconds', 'loss_spike_factor'):
        if not math.isfinite(monitor[key]) or monitor[key] <= 0:
            raise ValueError('Invalid monitoring.'+key)
    from .experiment_layout import configure
    return configure(cfg)
