"""Isaac-only task configuration; importing this module never starts a simulator."""
import copy
import hashlib
import importlib.util
import json
import math
import os
import runpy
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
HERE = CODE
EXPERIMENTS = CODE / 'experiments'
SIM = CODE.parent / 'environments/isaac_lab'
if not Path(__file__).with_name('runtime_versions.py').is_file():
    sys.path.append(str(SIM))
SAC = CODE/'source_tasks'
ROOT = CODE.parent
PROJECT_CONFIG = SAC / 'config.json'


def experiment_directory(name, root=EXPERIMENTS):
    if name in ('omni', 'omni_step'):
        return root / 'poor_performance' / name
    if name in ('004', 'walk_only', 'walk_variable', 'walk_lateral', 'walk_lateral_random_vx'):
        return root / 'training_trials' / name
    raise KeyError(name)


def relocated_experiment_path(value, project_root):
    path = Path(value)
    for old_assets in (project_root/'assets', project_root.parent/'sac_turn/isaac_lab/assets'):
        if path.is_relative_to(old_assets):
            return project_root.parent/'environments/isaac_lab/assets'/path.relative_to(old_assets)
    active_root = project_root/'experiments'
    for previous_root in (project_root.parent/'sac_turn/isaac_lab/experiments', active_root):
        try:
            name, *rest = path.relative_to(previous_root).parts
            return experiment_directory(name, active_root).joinpath(*rest)
        except (ValueError, KeyError):
            continue
    legacy = project_root.parent / 'sac_turn'
    if path in (legacy/'isaac_lab/config.json', project_root/'simulator_config.json'):
        return project_root.parent/'environments/isaac_lab/setup/simulator_config.json'
    for old, new in ((legacy/'isaac_lab', project_root), (legacy, project_root/'source_tasks')):
        try:
            return new / path.relative_to(old)
        except ValueError:
            continue
    return path


def checkpoint_path(value, project_root):
    path = Path(value).resolve()
    if not path.exists():
        path = relocated_experiment_path(path, project_root)
    return path.resolve(strict=True)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def launch_configuration(experiment=None, options=None):
    if options is None:
        options = read_json(CODE/'config.json')
    experiment = experiment or options['current_experiment']
    cfg = load_configuration(experiment=experiment, options=options)
    settings = cfg['isaac']
    sim = options.get('sim_device', settings['device'])
    learner = options.get('learner_device', sim)
    if any(device not in ('cpu', 'cuda:0') for device in (sim, learner)):
        raise ValueError('sim_device and learner_device must be cpu or cuda:0')
    headless = options.get('headless', settings['headless'])
    if type(headless) is not bool:
        raise ValueError('headless must be boolean')
    settings.update(device=sim, learner_device=learner, headless=headless)
    return cfg


def simulator_settings(settings_path=None):
    settings_path = Path(settings_path or SIM / 'setup/simulator_config.json').resolve()
    settings = read_json(settings_path)
    for key in ('asset','urdf_source','nao_meshes_path','isaac_python','isaac_sim_path','isaac_lab_path'):
        value = settings.get(key)
        if value and not Path(value).is_absolute():
            relative = Path(value)
            if relative.parts[0] == 'runtime':
                runtime = Path(read_json(CODE/'config.json')['runtime_root'])
                settings[key] = str((CODE/runtime/Path(*relative.parts[1:])).resolve())
            elif relative.parts[0] == 'assets':
                settings[key] = str((SIM/relative).resolve())
            else:
                settings[key] = str((HERE/relative).resolve())
    return settings


def load_configuration(settings_path=None, experiment=None, *, for_bc_fit=False, options=None):
    settings_path = Path(settings_path or SIM / 'setup/simulator_config.json').resolve()
    settings = simulator_settings(settings_path)
    if settings.get('backend') != 'isaac_lab_physx' or settings.get('num_envs') != 1:
        raise ValueError('This backend requires exactly one Isaac PhysX environment')
    for key in ('headless',):
        if type(settings[key]) is not bool:
            raise ValueError(key + ' must be boolean')
    for key in ('decimation', 'check_steps', 'check_resets'):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(key + ' must be a positive integer')
    for key in ('physics_dt', 'spawn_height', 'leg_stiffness', 'leg_damping',
                'other_stiffness', 'other_damping', 'friction', 'hip_tolerance_rad'):
        if not math.isfinite(settings[key]) or settings[key] <= 0:
            raise ValueError(key + ' must be finite and positive')
    upper_pose = settings.get('upper_joint_positions')
    if not isinstance(upper_pose, dict) or set(upper_pose) != {'LShoulderPitch','RShoulderPitch','LElbowRoll','RElbowRoll'}:
        raise ValueError('Expected the four approved upper-body initial joint positions')
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in upper_pose.values()):
        raise ValueError('Upper-body initial joint positions must be finite radians')
    spec = importlib.util.spec_from_file_location('_isaac_sac_project', SAC / 'project.py')
    project = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(project)
    project_settings = read_json(PROJECT_CONFIG)
    backend = project_settings.get('backend_overrides', {}).get(settings['backend'], {})
    experiment = experiment or backend.get('current_experiment') or project_settings['current_experiment']
    if experiment in ('omni_step', 'walk_only', 'walk_variable', 'walk_lateral', 'walk_lateral_random_vx'):
        settings['experiment_variant'] = experiment
        experiment = 'omni'
    settings['experiment_id'] = experiment
    omni = read_json(experiment_directory('omni')/'config.json') if experiment == 'omni' else None
    source_id = omni['source_experiment'] if omni else experiment
    descriptor_folder = experiment_directory('004') if experiment == '004' else None
    cfg = copy.deepcopy(project.training_configuration(experiment=source_id, descriptor_folder=descriptor_folder))
    hyperparameter_path = CODE/'training_config.json'
    hyperparameters = read_json(hyperparameter_path)
    if (cfg.get('algorithm') != 'sac' or not cfg['initialization'].get('use_bc') or
            cfg['actor_observation'] != 'shared_state_with_yaw_ema_only_v1' or
            cfg['critic_observation'] != 'actor'):
        raise ValueError('Isaac requires the shared-state BC-initialized SAC contract')
    if abs(settings['physics_dt'] * settings['decimation'] -
           cfg['webots']['control_time_step_ms'] / 1000.) > 1e-9:
        raise ValueError('Isaac and source SAC control intervals differ')
    shared = project_settings['training']
    if type(shared['total_env_steps']) is not int or shared['total_env_steps'] <= 0:
        raise ValueError('Invalid SAC project online budget')
    source_budget = cfg['training']['total_env_steps']
    output_root = HERE
    budget_path = PROJECT_CONFIG
    budget_field = 'training.total_env_steps'
    if experiment in ('004', 'omni'):
        from standing_check import configure_standing_pose
        configure_standing_pose(settings, 'ddpg_settled', SIM/'assets')
        output_root = experiment_directory(experiment)
        budget_path = Path(cfg['project_layout']['experiment_folder']) / 'config.json'
        budget_field = 'webots_total_env_steps'
    else:
        cfg['training']['total_env_steps'] = shared['total_env_steps']
    if omni:
        budget_path = output_root/'config.json'
        budget_field = 'total_env_steps'
        cfg['training'].pop('online_fixed_cycle_limit')
        cfg['training'].update(total_env_steps=omni[budget_field], omni_curriculum={
            key: omni[key] for key in ('block_steps', 'walk_speed_steps', 'turn_ramp_steps', 'turn_peak_rad_s', 'walk_stages')})
        cfg.update(experiment='omni', experiment_name=omni['name'],
                   reward_module_path=str(output_root/'reward.py'),
                   reward_source_sha256=sha(output_root/'reward.py'))
        if settings.get('experiment_variant') in ('walk_only', 'walk_variable', 'walk_lateral', 'walk_lateral_random_vx'):
            output_root = experiment_directory('walk_only')
            walk_root = output_root
            budget_path = output_root/'config.json'
            walk = read_json(budget_path)
            if settings['experiment_variant'] != 'walk_only':
                output_root = experiment_directory(settings['experiment_variant'])
                if settings['experiment_variant'] == 'walk_lateral_random_vx':
                    recipe = runpy.run_path(str(output_root/'training.py'))
                    budget_path = hyperparameter_path
                    budget = hyperparameters['total_env_steps']
                    if type(budget) is not int or budget < 1:
                        raise ValueError('total_env_steps must be a positive integer')
                    if options is None:
                        options = read_json(CODE/'config.json')
                    walk.update(commands_m_s=recipe['commands_m_s'], hold_steps=recipe['hold_steps'],
                                vx_range_m_s=recipe['vx_range_m_s'], total_env_steps=budget)
                else:
                    budget_path = output_root/'config.json'
                    walk.update(read_json(budget_path))
                if 'vx_range_m_s' in walk:
                    cfg['environment']['command_ranges'][0] = walk['vx_range_m_s']
            if settings['experiment_variant'] == 'walk_lateral':
                count = walk.get('num_envs', 1)
                if type(count) is not int or count < 1:
                    raise ValueError('walk_lateral num_envs must be a positive integer')
                settings['num_envs'] = count
            steps_per_action = walk['physics_steps_per_action']
            if type(steps_per_action) is not int or steps_per_action < 1:
                raise ValueError('Walk physics_steps_per_action must be a positive integer')
            settings['physics_dt'] = cfg['webots']['control_time_step_ms'] / (1000. * steps_per_action)
            settings['decimation'] = steps_per_action
            settings['webots_leg_servo'] = walk['webots_leg_servo']
            if not for_bc_fit:
                manifest_path = walk_root / walk['bc_manifest']
                manifest = read_json(manifest_path)
                if manifest['network_inputs'] != cfg['network_inputs']:
                    raise ValueError('Walk BC Actor input layout mismatch')
                cfg['initialization'].update(actor_path=str(manifest_path.parent / manifest['checkpoint']),
                                             sha256=manifest['checkpoint_sha256'],
                                             source_network_inputs=manifest['network_inputs'])
            del cfg['training']['omni_curriculum']
            cfg['training'].update(total_env_steps=walk['total_env_steps'], walk_commands={
                key: walk[key] for key in ('commands_m_s', 'hold_steps')})
            if 'repeat_from_index' in walk:
                cfg['training']['walk_commands']['repeat_from_index'] = walk['repeat_from_index']
            scale = walk['tracking_scale_m_s']
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError('Walk tracking scale must be finite and positive')
            cfg['reward']['biped_tracking'].update(walk_reward_revision='body_xy_linear_fixed_scale_v5',
                                                  tracking_scale_m_s=scale)
            cfg.update(experiment=settings['experiment_variant'], experiment_name=settings['experiment_variant'],
                       reward_module_path=str(walk_root/'reward.py'), reward_source_sha256=sha(walk_root/'reward.py'))
        elif 'experiment_variant' in settings:
            output_root = experiment_directory(settings['experiment_variant'])
            turn = read_json(output_root/'turn.json')
            curriculum = cfg['training']['omni_curriculum']
            del curriculum['turn_ramp_steps']
            curriculum.update(turn)
            cfg['experiment_name'] = settings['experiment_variant']
    if options is not None:
        name = settings.get('experiment_variant', experiment)
        count = options.get('num_envs', settings['num_envs'])
        if isinstance(count, dict):
            count = count.get(name, settings['num_envs'])
        if type(count) is not int or count < 1:
            raise ValueError('num_envs must be a positive integer')
        if count > 1 and name not in ('walk_lateral', 'walk_lateral_random_vx'):
            raise ValueError(f'{name} uses the single-robot trainer; set num_envs to 1')
        settings['num_envs'] = count
    online = hyperparameters['training']
    if set(online) != {'batch_size', 'replay_size', 'checkpoint_every'}:
        raise ValueError(f'Expected batch_size, replay_size and checkpoint_every in {CODE}/training_config.json')
    cfg['seed'] = hyperparameters['seed']
    cfg['sac'] = hyperparameters['sac']
    cfg['training'].update(online)
    cfg['source_webots'] = cfg.pop('webots')
    cfg['environment'].update(reset_mode='isaac_state', floor_z=0., yaw_source='isaac_truth')
    cfg['backend'] = 'isaac_lab_physx'
    cfg['experiment'] = 'isaac_' + cfg['experiment']
    asset = Path(os.environ.get('NAO_ISAAC_ASSET', settings['asset']))
    if not asset.is_absolute():
        asset = HERE / asset
    asset = relocated_experiment_path(asset, HERE) if not asset.exists() else asset
    cfg['isaac'] = dict(settings, asset=str(asset.resolve()))
    cfg['project_layout'] = dict(project_root=str(HERE), experiment_folder=str(output_root),
        config_path=str(settings_path), config_sha256=sha(settings_path),
        runs_root=str(output_root / 'runs'), reports_root=str(output_root / 'reports'))
    cfg['budget_source'] = dict(path=str(budget_path), sha256=sha(budget_path),
        field=budget_field, source_webots_budget=source_budget,
        effective_budget=cfg['training']['total_env_steps'])
    cfg['diagnostics'] = dict(units='SI_radians', coordinate_frame='NAO torso; world Z up',
        velocity_truth='root-link position difference rotated into current root-link frame',
        angular_velocity_truth='wrapped world heading difference over control interval',
        gyro_timing='post-action PhysX body angular velocity; ideal sensor',
        support_force_truth='world-Z ankle net contact force',
        control_period_s=settings['physics_dt'] * settings['decimation'])
    return cfg
