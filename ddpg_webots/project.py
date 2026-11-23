"""Shared live configuration boundary for training and offline reports."""
import copy
import hashlib
import json
import re
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent
ROOT = CODE / 'data/runtime'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_entry_config(config_path, selected):
    if config_path == ROOT/'config.json' and selected and Path(selected).suffix.lower() == '.json':
        return Path(selected).resolve(), None
    return config_path, selected


def identifier(value):
    value = str(value)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,99}', value):
        raise ValueError('Invalid experiment ID')
    return value


def load_experiment(config_path=None, selected=None):
    path = Path(config_path or ROOT/'config.json').resolve()
    document = read_json(path)
    if document.get('current_training_backend') == 'isaac_lab_physx' or document.get('backend') == 'isaac_lab_physx':
        raise ValueError('Isaac SAC uses the shared Start Training / Check Training entries and sac_isaac_lab outputs. Select an explicit Webots project config for legacy Webots tools.')
    if document.get('experiment_id') and 'training_config' in document:
        experiment_id = identifier(document['experiment_id'])
        if selected is not None and selected != experiment_id:
            raise ValueError('This config belongs to experiment '+experiment_id)
        return document, path.parent, document, path
    if path == ROOT/'config.json' and selected is None and document.get('current_experiment_project'):
        project_path = path.parent / document['current_experiment_project']
        project, folder, descriptor, resolved = load_experiment(project_path, document['current_experiment'])
        shared = document['training_config']['training']
        override = project['experiment_overrides'][descriptor['experiment_id']]['training']
        override.setdefault('total_env_steps', shared['total_env_steps'])
        override['online_count_fall_as_attempt'] = shared['online_count_fall_as_attempt']
        return project, folder, descriptor, resolved
    selected = identifier(selected or document['current_experiment'])
    folder = (path.parent/document.get('experiments_root', 'experiments')/selected).resolve()
    descriptor = read_json(folder/'config.json')
    if descriptor['experiment_id'] != selected:
        raise ValueError('Experiment ID/config mismatch')
    return document, folder, descriptor, path


def training_configuration(config_path=None, selected=None):
    document, folder, descriptor, path = load_experiment(config_path, selected)
    cfg = copy.deepcopy(document['training_config'])
    def merge(target, patch):
        for key,value in patch.items():
            if isinstance(value,dict) and isinstance(target.get(key),dict):
                merge(target[key],value)
            else:
                target[key]=copy.deepcopy(value)
    common_path = CODE/'training_config.json'
    portable = folder.parent == CODE/'experiments'
    if portable:
        shared = read_json(common_path)
        merge(shared, cfg)
        cfg = shared
    merge(cfg,document.get('experiment_overrides',{}).get(descriptor['experiment_id'],{}))
    path_keys = {'path','actor','critic','dataset','bc_checkpoint','source','source_config',
                 'state_migration_report','manifest','world_template'}
    def local_assets(value, key=''):
        if isinstance(value, dict):
            if key == 'asset_relocations':
                return {k:str((path.parent/Path(v)).resolve()) if not Path(v).is_absolute() else v
                        for k,v in value.items()}
            return {k:local_assets(v,k) for k,v in value.items()}
        if isinstance(value,list):return [local_assets(v,key) for v in value]
        if isinstance(value,str) and key in path_keys and not Path(value).is_absolute():
            return str((path.parent/value).resolve())
        return value
    cfg = local_assets(cfg)
    if cfg['webots']['python'] == 'python':cfg['webots']['python'] = sys.executable
    reward = (folder/descriptor.get('reward_file', 'reward.py')).resolve()
    if folder not in reward.parents or not reward.is_file():
        raise ValueError('Experiment Reward must be a file inside the experiment folder')
    cfg['experiment'] = descriptor['experiment_id']
    cfg['experiment_name'] = descriptor['name']
    cfg['experiment_scoped_assets'] = False
    cfg['reward_module_path'] = str(reward)
    cfg['reward_source_sha256'] = digest(reward)
    if cfg.get('reward', {}).get('biped_tracking', {}).get('velocity_source') == 'causal_mean':
        cfg['reward_history_sha256'] = digest(CODE/'training/basewalk/reward_history.py')
    if cfg.get('reward',{}).get('biped_tracking',{}).get('revision') == 'velocity_widths_variants_v1':
        cfg['reward_velocity_source_sha256'] = digest(CODE/'training/basewalk/reward_velocity.py')
    cfg['project_layout'] = dict(project_root=str(ROOT), config_path=str(path),
        config_sha256=digest(path), experiment_folder=str(folder),
        descriptor_sha256=digest(folder/'config.json'), runs_root=str((folder/descriptor.get('runs_root', 'runs')).resolve()),
        reports_root=str((folder/descriptor.get('reports_root', 'reports')).resolve()))
    cfg['diagnostics'] = dict(schema='report_log_v1', units='SI_radians',
        velocity_truth='Supervisor position difference rotated into current body frame',
        angular_velocity_truth='SO(3) world rotation difference, expressed in current body frame',
        gyro_timing='global angular velocity is control-period mean; sensor is post-action sample',
        imu_mount_roll_rad=1.5708, coordinate_frame='NAO torso axes; world gravity=(0,0,-1)',
        forward_lateral_reference='fixed reset task heading_reference',
        control_period_s=cfg['webots']['control_time_step_ms']/1000.)
    if portable:
        cfg['project_layout'].update(training_config_path=str(common_path), training_config_sha256=digest(common_path))
    cfg['diagnostics'].update(joint_truth='Supervisor JointParameters.position; independent causal difference for velocity',
        joint_units='raw radians and radians/s; State offsets/scales are removed for comparison',
        foot_truth='contact points and sensor-body height; includes any contacting object, not force',
        support_force_truth='unavailable_without_physics_engine_force_provider')
    return cfg
