"""Validation, immutable run input, and controller configuration adapters."""

import copy
import hashlib
import json
import math
import os
from pathlib import Path

from configs.project_config import CONFIG

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_ENV = "WALKING_RUN_CONFIG"


def project_path(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def validate(config):
    # Historical snapshots predate exploration; preserve their no-noise behavior.
    config = copy.deepcopy(config)
    config["environment"].setdefault("target_heading_rad",0.)
    config["training"].setdefault("residual_limit_rad",.03)
    config["initialization"].setdefault("stall_min_forward_progress_m", .02)
    config["initialization"].setdefault("stall_max_abs_x_progress_m", .02)
    if isinstance(config.get('evaluation'), dict):
        # Old frozen evaluations keep their original stopping conditions.
        for key, value in dict(stall_cycle_limit=0, stall_window_steps=35,
                               stall_max_abs_x_progress_m=.02).items():
            config['evaluation'].setdefault(key, value)
    if isinstance(config.get('environment'), dict):
        config['environment'].setdefault('heading_reward_weight', 0.0)
        config['environment'].setdefault('foot_contact_threshold_n', 1.0)
    if isinstance(config.get("initialization"), dict):
        config['initialization'].setdefault('training_reset_mode', 'legacy')
        config['initialization'].setdefault('clear_replay_on_bc_reset', True)
        config["initialization"].setdefault("bc_reset_max_rollouts", 0)
        for key, value in dict(bc_reset_max_env_steps=0, consecutive_falls=15,
                               stall_window_steps=35, stall_timeout_steps=1400,
                               consecutive_fall_rollouts=15, stall_rollouts=40).items():
            config['initialization'].setdefault(key, value)
    if isinstance(config.get("training"), dict):
        config["training"].setdefault("action_noise_std_rad", 0.0)
        config['training'].setdefault('noise_switch_env_steps', 0)
        config['training'].setdefault('action_noise_std_after_switch', config['training']['action_noise_std_rad'])
        config["training"].setdefault("critic_warmup_updates", 0)
        config['training'].setdefault('manage_critic_batchnorm', False)
        config['training'].setdefault('sync_bn_running_stats', False)
        config['training'].setdefault('lr_switch_env_steps', 0)
        config['training'].setdefault('actor_lr_after_switch', config['training'].get('actor_lr'))
        config['training'].setdefault('critic_lr_after_switch', config['training'].get('critic_lr'))
        # Legacy fields are normalized only for historical adapters.
        for key, value in dict(sampling_mode='legacy_rollouts', total_env_steps=70000,
                               update_every_steps=350, log_every_steps=35,
                               rollouts=2000, rollout_steps=35, train_every_rollouts=10).items():
            config['training'].setdefault(key, value)
        config['training'].setdefault('critic_warmup_update_every_steps', config['training']['update_every_steps'])
    if set(config) != set(SCHEMA):
        raise ValueError("Unknown or missing configuration group")
    for group in SCHEMA:
        # Keys are checked against a schema independent of the editable values.
        if set(config[group]) != set(SCHEMA[group]):
            raise ValueError(f"Unknown or missing parameter in {group}: {set(config[group]) ^ set(SCHEMA[group])}")
        for key, expected in SCHEMA[group].items():
            value = config[group][key]
            if expected == "optional_int":
                valid = value is None or type(value) is int
            elif expected == "optional_str":
                valid = value is None or isinstance(value, str)
            elif expected == "number":
                valid = type(value) in (int, float) and math.isfinite(value)
            else:
                valid = type(value).__name__ == expected
            if not valid:
                raise ValueError(f"Invalid type/value: {group}.{key}={value!r}")
    w, t, e, i, v = (config[k] for k in ("webots", "training", "environment", "initialization", "evaluation"))
    for group, keys in POSITIVE.items():
        for key in keys:
            if config[group][key] <= 0:
                raise ValueError(f"{group}.{key} must be positive")
    if w["control_time_step_ms"] % w["basic_time_step_ms"]:
        raise ValueError("control_time_step_ms must be a multiple of basic_time_step_ms")
    if t["buffer_size"] <= t["batch_size"] or t["min_buffer_for_train"] > t["buffer_size"]:
        raise ValueError("buffer_size must exceed batch_size and cover min_buffer_for_train")
    if not 0 <= t["gamma"] <= 1 or not 0 < t["tau"] <= 1:
        raise ValueError("Invalid gamma/tau")
    if not 0 < e["motor_velocity_scale"] <= 1 or not 0 <= e["phase_reset_value"] < 1:
        raise ValueError("Invalid motor scale/phase reset")
    if e["phase_mode"] not in ("continuous_time", "sequence") or i["resume"] not in ("auto", "never", "required"):
        raise ValueError("Invalid phase_mode/resume")
    if t['sampling_mode'] not in ('continuous', 'legacy_rollouts'):
        raise ValueError('Invalid sampling_mode')
    residual = t['algorithm_module'] == 'algorithms.ddpg_residual'
    heading = e['task_module'] in ('tasks.heading_correction', 'tasks.heading_rate')
    if (residual and not heading) or (heading and t['sampling_mode'] != 'continuous'):
        raise ValueError('Residual Actor requires heading Task; heading Task requires continuous sampling')
    if residual and not 0 < t['residual_limit_rad'] <= .1:
        raise ValueError('residual_limit_rad must be in (0,.1]')
    if t['sync_bn_running_stats'] and not t['manage_critic_batchnorm']:
        raise ValueError('BN statistics synchronization requires managed Critic modes')
    if t['noise_switch_env_steps'] and t['sampling_mode'] != 'continuous':
        raise ValueError('Noise schedule requires continuous sampling')
    if t['lr_switch_env_steps'] and t['sampling_mode'] != 'continuous':
        raise ValueError('Learning-rate stages require continuous sampling')
    if i['training_reset_mode'] not in ('legacy', 'world_reload'):
        raise ValueError('Invalid training_reset_mode')
    if i['training_reset_mode'] == 'world_reload' and t['sampling_mode'] != 'continuous':
        raise ValueError('world_reload training reset requires continuous sampling')
    if t['sampling_mode'] == 'continuous' and e['phase_mode'] != 'continuous_time':
        raise ValueError('Continuous sampling requires continuous_time phase')
    if not 0 < v["first_episode_step_fraction"] <= 1:
        raise ValueError("first_episode_step_fraction must be in (0, 1]")
    for key, size in (("initial_translation", 3), ("initial_rotation", 4)):
        if len(w[key]) != size or not all(type(x) in (int, float) and math.isfinite(x) for x in w[key]):
            raise ValueError(f"Invalid {key}")
    if sum(x*x for x in w["initial_rotation"][:3]) == 0:
        raise ValueError("Rotation axis cannot be zero")
    for group, keys in NONNEGATIVE.items():
        if any(config[group][key] < 0 for key in keys):
            raise ValueError(f"Negative setting in {group}")
    if v["selected_global_step"] is not None and v["selected_global_step"] < 0:
        raise ValueError("Negative selected_global_step")
    for name in (config["experiment"]["name"], v["metrics_filename"]):
        if not name or name in (".", "..") or any(c in name for c in '/\\:'):
            raise ValueError("Experiment/metrics name must be a single filename component")
    return config


SCHEMA = {
    "experiment": dict(name="str", seed="int", output_root="str", previous_experiment="optional_str", baseline_id="str"),
    "webots": dict(executable="str", python="str", world_template="str", basic_time_step_ms="int", control_time_step_ms="int", initial_translation="list", initial_rotation="list"),
    "environment": dict(task_module="str", task_class="str", settle_steps="int", sensor_warmup_steps="int", shoulder_position_rad="number", motor_velocity_scale="number", joint_limit_margin_rad="number", fall_height_m="number", phase_mode="str", phase_coefficient="number", phase_reset_value="number"),
    "training": dict(algorithm_module="str", algorithm_class="str", actor_lr="number", critic_lr="number", action_noise_std_rad="number", gamma="number", tau="number", batch_size="int", buffer_size="int", min_buffer_for_train="int", critic_warmup_updates="int", rollouts="int", rollout_steps="int", train_every_rollouts="int", stop_x_m="number", checkpoint_every_updates="int"),
    "initialization": dict(use_bc="bool", bc_checkpoint="str", resume="str", resume_actor_lr="number", resume_critic_lr="number", bc_on_resume="bool", reset_to_bc_on_failure="bool", bc_reset_max_rollouts="int", consecutive_fall_rollouts="int", stall_rollouts="int", stall_max_abs_x_progress_m="number"),
    "evaluation": dict(episodes="int", stop_x_m="number", max_steps="int", first_episode_step_fraction="number", stop_after_first_failure="bool", start_global_step="int", selected_global_step="optional_int", only_latest="bool", metrics_filename="str", inherit_training_config="bool", allow_legacy_without_snapshot="bool"),
}
SCHEMA['training'].update(sampling_mode='str', total_env_steps='int', update_every_steps='int', log_every_steps='int')
SCHEMA['training'].update(lr_switch_env_steps='int', actor_lr_after_switch='number', critic_lr_after_switch='number')
SCHEMA['training']['manage_critic_batchnorm'] = 'bool'
SCHEMA['training']['sync_bn_running_stats'] = 'bool'
SCHEMA['environment']['heading_reward_weight'] = 'number'
SCHEMA['environment']['target_heading_rad'] = 'number'
SCHEMA['training']['residual_limit_rad'] = 'number'
SCHEMA['environment']['foot_contact_threshold_n'] = 'number'
SCHEMA['training'].update(noise_switch_env_steps='int', action_noise_std_after_switch='number')
SCHEMA['training']['critic_warmup_update_every_steps'] = 'int'
SCHEMA['evaluation'].update(stall_cycle_limit='int', stall_window_steps='int', stall_max_abs_x_progress_m='number')
SCHEMA['initialization'].update(bc_reset_max_env_steps='int', consecutive_falls='int', stall_window_steps='int', stall_timeout_steps='int')
SCHEMA['initialization']['training_reset_mode'] = 'str'
SCHEMA['initialization']['clear_replay_on_bc_reset'] = 'bool'
SCHEMA['initialization']['stall_min_forward_progress_m'] = 'number'
POSITIVE = {
    "webots": ("basic_time_step_ms", "control_time_step_ms"),
    "training": ("actor_lr", "critic_lr", "batch_size", "buffer_size", "min_buffer_for_train", "rollouts", "rollout_steps", "train_every_rollouts"),
    "initialization": ("resume_actor_lr", "resume_critic_lr", "consecutive_fall_rollouts", "stall_rollouts"),
    "environment": ("fall_height_m", "phase_coefficient"),
    "evaluation": ("episodes", "max_steps"),
}
NONNEGATIVE = {"experiment": ("seed",), "environment": ("settle_steps", "sensor_warmup_steps", "joint_limit_margin_rad"), "training": ("checkpoint_every_updates", "action_noise_std_rad", "critic_warmup_updates"), "initialization": ("stall_max_abs_x_progress_m", "bc_reset_max_rollouts"), "evaluation": ("start_global_step",)}


POSITIVE['training'] += ('total_env_steps', 'update_every_steps', 'log_every_steps')
POSITIVE['training'] += ('critic_warmup_update_every_steps',)
POSITIVE['training'] += ('actor_lr_after_switch', 'critic_lr_after_switch')
NONNEGATIVE['training'] += ('lr_switch_env_steps',)
NONNEGATIVE['environment'] += ('heading_reward_weight',)
POSITIVE['environment'] += ('foot_contact_threshold_n',)
NONNEGATIVE['training'] += ('noise_switch_env_steps', 'action_noise_std_after_switch')
POSITIVE['evaluation'] += ('stall_window_steps',)
NONNEGATIVE['evaluation'] += ('stall_cycle_limit', 'stall_max_abs_x_progress_m')
POSITIVE['initialization'] += ('consecutive_falls', 'stall_window_steps', 'stall_timeout_steps', 'stall_min_forward_progress_m')
NONNEGATIVE['initialization'] += ('bc_reset_max_env_steps',)


def load_config(mode=None):
    snapshot = os.environ.get(SNAPSHOT_ENV)
    if snapshot:
        payload = json.loads(Path(snapshot).read_text(encoding="utf-8"))
        if mode and payload["mode"] != mode:
            raise ValueError("Run snapshot mode does not match Controller")
        return validate(copy.deepcopy(payload["config"]))
    return validate(copy.deepcopy(CONFIG))


def require_launch():
    if not os.environ.get(SNAPSHOT_ENV):
        raise RuntimeError("Please start with launch.py train / eval so world settings and the run snapshot are applied.")
    payload = json.loads(Path(os.environ[SNAPSHOT_ENV]).read_text(encoding="utf-8"))
    for source in payload["sources"]:
        if file_hash(source["checkpoint"]) != source["sha256"]:
            raise RuntimeError(f"Model changed since preparation; prepare again: {source['checkpoint']}")


def evaluation_checkpoint_paths():
    snapshot = os.environ.get(SNAPSHOT_ENV)
    if not snapshot:
        return None
    payload = json.loads(Path(snapshot).read_text(encoding="utf-8"))
    return {str(Path(s["checkpoint"]).resolve()) for s in payload["sources"]}


def task_settings(config, mode):
    e = config["environment"]
    settings = dict(sequence_length=config["training"].get("rollout_steps", 35), phase_mode=e["phase_mode"],
                phase_frequency_hz=e["phase_coefficient"], phase_reset_value=e["phase_reset_value"],
                fall_height=e["fall_height_m"], stop_x=config["training" if mode == "train" else "evaluation"]["stop_x_m"],
                heading_reward_weight=e.get('heading_reward_weight', 0.0))
    if e['task_module'] in ('tasks.heading_correction', 'tasks.heading_rate'):
        settings['target_heading_rad'] = e['target_heading_rad']
    return settings


def algorithm_settings(config, device, task_id):
    t, i = config["training"], config["initialization"]
    settings = dict(state_dim=observation_dim(config), action_dim=10, hidden_dim=256, device=device, task_id=task_id, critic_warmup_updates=t["critic_warmup_updates"],
                manage_critic_batchnorm=t.get('manage_critic_batchnorm', False),
                sync_bn_running_stats=t.get('sync_bn_running_stats', False),
                **{k: t[k] for k in ("actor_lr", "critic_lr", "gamma", "tau", "batch_size", "buffer_size")},
                **{k: i[k] for k in ("resume_actor_lr", "resume_critic_lr")})
    if t["algorithm_module"] == "algorithms.ddpg_residual":
        settings["residual_limit_rad"] = t["residual_limit_rad"]
    return settings


def runtime_settings(config, mode=None):
    e = config["environment"]
    return dict(time_step_ms=config["webots"]["control_time_step_ms"],
                reset_mode=config['initialization'].get('training_reset_mode', 'legacy') if mode == 'train' else 'legacy',
                sensor_warmup_steps=e["sensor_warmup_steps"], shoulder_position=e["shoulder_position_rad"],
                motor_velocity_scale=e["motor_velocity_scale"], joint_limit_margin=e["joint_limit_margin_rad"],
                foot_contact_threshold_n=e.get('foot_contact_threshold_n', 1.0))


def observation_dim(config):
    if config['environment']['task_module']=='tasks.heading_rate':
        return 27
    return 25 if config['environment']['task_module']=='tasks.heading_correction' else 24


def create_actor(config):
    if config['training']['algorithm_module']=='algorithms.ddpg_residual':
        from residual_actor import ResidualActor
        return ResidualActor(observation_dim(config),256,10,config['training']['residual_limit_rad'])
    from models import BoundedActor
    return BoundedActor(observation_dim(config),256,10)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
