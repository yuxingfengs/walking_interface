"""Local configuration resolver for the standalone SAC Turn project."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def training_configuration(load_bc=True, experiment=None, *, descriptor_folder=None):
    project_path = ROOT / 'config.json'
    project = read_json(project_path)
    experiment = str(experiment or project['current_experiment'])
    if not experiment.isdigit() or len(experiment) != 3:
        raise ValueError('current_experiment must be a three-digit ID')
    folder = (ROOT / project.get('experiments_root', 'experiments') / experiment).resolve()
    if ROOT not in folder.parents:
        raise ValueError('Experiment folder escapes the SAC project')
    bc_folder = folder / 'bc'
    if descriptor_folder is not None:
        folder = Path(descriptor_folder).resolve()
    descriptor_path = folder / 'config.json'
    descriptor = read_json(descriptor_path)
    if descriptor.get('experiment_id') != experiment:
        raise ValueError('Experiment ID/config mismatch')

    cfg = read_json(folder / descriptor.get('training_config', 'training_config.json'))
    init = cfg.get('initialization', {})
    if load_bc and init.get('use_bc'):
        manifest = read_json(bc_folder/'manifest.json')
        init.update(actor_path=str(bc_folder/manifest['checkpoint']), sha256=manifest['checkpoint_sha256'])
    cfg['webots']['world_template'] = str((ROOT / cfg['webots']['world_template']).resolve())
    cfg['experiment'] = 'sac_turn_' + experiment
    cfg['experiment_name'] = descriptor['name']
    cfg['experiment_scoped_assets'] = False
    cfg['reward_module_path'] = str((folder / descriptor.get('reward_file', 'reward.py')).resolve())
    cfg['reward_source_sha256'] = digest(cfg['reward_module_path'])
    budget = project.get('backend_overrides', {}).get('webots', {}).get(
        'total_env_steps', project['training']['total_env_steps'])
    budget = descriptor.get('webots_total_env_steps', budget)
    if type(budget) is not int or budget <= 0:
        raise ValueError('SAC total_env_steps must be a positive integer')
    cfg['training']['total_env_steps'] = budget
    cfg['webots']['python'] = sys.executable
    cfg['project_layout'] = dict(
        project_root=str(ROOT),
        config_path=str(project_path),
        config_sha256=digest(project_path),
        experiment_folder=str(folder),
        descriptor_sha256=digest(descriptor_path),
        runs_root=str(folder / descriptor.get('runs_root', 'runs')),
        reports_root=str(folder / descriptor.get('reports_root', 'reports')),
    )
    cfg['diagnostics'] = dict(
        schema='report_log_v1', units='SI_radians',
        velocity_truth='Supervisor position difference rotated into current body frame',
        angular_velocity_truth='SO(3) world rotation difference, expressed in current body frame',
        gyro_timing='global angular velocity is control-period mean; sensor is post-action sample',
    )
    return cfg

