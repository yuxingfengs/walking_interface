"""Local, explicitly migrated FiLM BC registry."""
import hashlib
import json
from pathlib import Path
from . import DATA_ROOT as ROOT
BUNDLE = ROOT/'bc_candidates/film_rate_error_47d_v1'
FORWARD_NAME = 'forward_gait_bc_v1'
TURN90_NAME = 'film_turn90_47d_v1'
TURN_MULTI_NAME = 'film_turn_multi_47d_v1'
CONCAT_TURN_MULTI_NAME = 'concat_turn_multi_47d_v1'
CONCAT_THREE_STAGE_NAME = 'concat_three_stage_47d_v1'
CONCAT_THREE_STAGE_NOISY_NAME = 'concat_three_stage_noisy_47d_v1'
CONCAT_THREE_STAGE_NOISY_48D_NAME = 'concat_three_stage_noisy_48d_v1'

def bundle_path(name='film_rate_error_47d_v1'):
    if name == 'film_rate_error_47d_v1':
        return BUNDLE
    if name == FORWARD_NAME:
        return ROOT/'bc_candidates'/FORWARD_NAME
    if name == TURN90_NAME:
        return ROOT/'bc_candidates'/TURN90_NAME
    if name == TURN_MULTI_NAME:
        return ROOT/'bc_candidates'/TURN_MULTI_NAME
    if name == CONCAT_TURN_MULTI_NAME:
        return ROOT/'bc_candidates'/CONCAT_TURN_MULTI_NAME
    if name == CONCAT_THREE_STAGE_NAME:
        return ROOT/'bc_candidates'/CONCAT_THREE_STAGE_NAME
    if name == CONCAT_THREE_STAGE_NOISY_NAME:
        return ROOT/'bc_candidates'/CONCAT_THREE_STAGE_NOISY_NAME
    if name == CONCAT_THREE_STAGE_NOISY_48D_NAME:
        return ROOT/'bc_candidates'/'concat_three_stage_noisy_48d_v1'
    raise ValueError('Unknown experimental BC candidate')

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_bc_config(name='film_rate_error_47d_v1'):
    from .learning import load_policy
    bundle = bundle_path(name)
    if not (bundle/'manifest.json').is_file():
        raise ValueError('BC candidate has not been registered: '+name)
    manifest = json.loads((bundle/'manifest.json').read_text(encoding='utf-8'))
    checkpoint = bundle/'actor.pth'
    actual = digest(checkpoint)
    if manifest['name'] != name or actual != manifest['checkpoint_sha256']:
        raise ValueError('FiLM BC identity/hash mismatch')
    if name in (FORWARD_NAME, TURN90_NAME, TURN_MULTI_NAME, CONCAT_TURN_MULTI_NAME,
                CONCAT_THREE_STAGE_NAME, CONCAT_THREE_STAGE_NOISY_NAME,
                CONCAT_THREE_STAGE_NOISY_48D_NAME):
        evidence = bundle/'evaluation.json'
        if not evidence.is_file() or digest(evidence)!=manifest.get('evaluation_sha256'):
            raise ValueError('BC evaluation evidence hash mismatch')
        evaluation=json.loads(evidence.read_text(encoding='utf-8'))
        if evaluation.get('checkpoint_sha256')!=actual or not evaluation.get('episodes'):
            raise ValueError('BC evaluation checkpoint identity mismatch')
    _, config, _ = load_policy(checkpoint)
    config['initialization'] = dict(bc_checkpoint=str(checkpoint), bc_sha256=actual,
        candidate_name=manifest['name'], validation_status=manifest['status'])
    return config

def model_contract(config):
    from .core import CONTRACT
    e = config['environment']
    result=dict(interface=CONTRACT, actor_type=config['actor_type'],
        actor_hidden=config['actor_hidden'],
        preprocessing={k:e[k] for k in ['action_scale', 'joint_margin', 'joint_velocity_scale',
            'gait_frequency_hz', 'yaw_source', 'yaw_rate_scale', 'command_slew_per_second']},
        control_time_step_ms=config['webots']['control_time_step_ms'])
    result['preprocessing']['phase_runs_at_zero_command'] = bool(
        e.get('phase_runs_at_zero_command', False))
    result['preprocessing']['support_observation'] = config['support_observation']
    result['preprocessing']['imu_full_axes'] = config['physical_gait']['imu_full_axes']
    if config.get('network_inputs'):
        result['network_inputs'] = config['network_inputs']
    if config.get('actor_observation'):
        result['actor_observation'] = config['actor_observation']
    if config['actor_type']=='film_gait_correction':
        result.update(correction_hidden=config['correction_hidden'],
                      correction_scale=config['correction_scale'])
    return result


def compatible_contract(saved,requested,allow_yaw_source_change=False):
    """Compare Actor contracts; optionally migrate only the outer yaw provider."""
    left=model_contract(saved);right=model_contract(requested)
    if allow_yaw_source_change:
        left['preprocessing']=dict(left['preprocessing'])
        right['preprocessing']=dict(right['preprocessing'])
        left['preprocessing'].pop('yaw_source',None)
        right['preprocessing'].pop('yaw_source',None)
    return left==right
