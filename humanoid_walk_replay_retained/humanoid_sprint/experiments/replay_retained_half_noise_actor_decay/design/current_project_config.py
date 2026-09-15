"""统一参数入口。修改后通过 launch.py train / eval 启动，新运行才生效。

路径相对于 humanoid_sprint；不要在旧 controllers/*/run_config.py 中修改参数。
"""

CONFIG = {
    "experiment": {
        "name": "replay_retained_half_noise_actor_decay",
        "seed": 1,
        "output_root": "experiments",
        "previous_experiment": "improve_head_reward_1_noise2_critic1e6_no_norm",
        "baseline_id": "bc_strong_heading_20260915_v1",
    },
    "webots": {
        "executable": "C:/Program Files/Webots/msys64/mingw64/bin/webots.exe",
        "python": "C:/Users/fengy/anaconda3/envs/webot/python.exe",
        "world_template": "worlds/humanoid_sprint.wbt",
        "basic_time_step_ms": 20,
        "control_time_step_ms": 40,
        "initial_translation": [-5.02099, 1.0, 0.696172],
        "initial_rotation": [0.0, 0.0, 1.0, 0.0837],
    },
    "environment": {
        "task_module": "tasks.heading_rate",
        "task_class": "HeadingRateTask",
        "heading_reward_weight": 1,  # Absolute heading penalty; yaw rate is observation only.
        "target_heading_rad": 0.0,  # World +X direction; error = wrap(target - world heading).
        "foot_contact_threshold_n": 1.0,  # Per-foot local normal force threshold.
        "settle_steps": 64,
        "sensor_warmup_steps": 5,
        "shoulder_position_rad": 1.1,
        "motor_velocity_scale": 0.8,
        "joint_limit_margin_rad": 0.001,
        "fall_height_m": 0.6,
        "phase_mode": "continuous_time",
        # 保留旧公式：phase = elapsed * coefficient / (2*pi)，不是每秒4个周期。
        "phase_coefficient": 4.0,
        "phase_reset_value": 0.0,
    },
    "training": {
        "algorithm_module": "algorithms.ddpg_no_norm",
        "algorithm_class": "DDPGNoNormAlgorithm",
        "actor_lr": 1e-5,
        "critic_lr": 1e-6,
        "manage_critic_batchnorm": True,
        "sync_bn_running_stats": False,
        "lr_switch_env_steps": 100000,  # Second half starts at transition 100001.
        "actor_lr_after_switch": 1e-6,
        "critic_lr_after_switch": 1e-6,
        "action_noise_std_rad": 0.002,  # Training-only Gaussian exploration; evaluation has no noise.
        "noise_switch_env_steps": 100000,
        "action_noise_std_after_switch": 0.0,
        "gamma": 0.98,
        "tau": 1e-4,
        "batch_size": 64,
        "buffer_size": 2000,
        "min_buffer_for_train": 320,
        "critic_warmup_updates": 50,  # Freeze Actor for the first 50 Critic update rounds.
        "critic_warmup_update_every_steps": 35,  # Faster Critic-only updates; joint updates keep their own interval.
        "sampling_mode": "continuous",
        "total_env_steps": 200000,  # Two equal 100000-step phases.
        "update_every_steps": 350,  # Joint update every 10 nominal 35-step rollouts.
        "log_every_steps": 35,  # Logging only; does not change phase, rewards or updates.
        "stop_x_m": 6.0,  # Training continues 1 m past the evaluation finish.
        "checkpoint_every_updates": 10,
    },
    "initialization": {
        "training_reset_mode": "world_reload",
        "use_bc": True,
        "bc_checkpoint": "experiments/walk_bc_strong_heading_70k/design/actor_bc_heading27.pth",
        "resume": "never",  # 本轮从BC开始；新实验避免覆盖历史模型或恢复旧optimizer学习率
        "resume_actor_lr": 1e-6,
        "resume_critic_lr": 1e-6,
        "bc_on_resume": True,  # 保留当前“恢复后仍加载BC Actor”的行为
        "reset_to_bc_on_failure": True,
        "clear_replay_on_bc_reset": False,  # Keep failure experience for Critic learning.
        "bc_reset_max_env_steps": 0,  # Allow failure-triggered BC recovery throughout training.
        "consecutive_falls": 5,
        "stall_window_steps": 35,
        "stall_timeout_steps": 700,  # 28 s of consecutive low progress at 40 ms per step.
        "stall_min_forward_progress_m": 0.02,  # New signed odometry progress required to restart timer.
    },
    "evaluation": {
        "episodes": 1,
        "stop_x_m": 5,
        "max_steps": 6000,
        "first_episode_step_fraction": 0.5,  # 保留原首回合最多3000步的行为
        "stop_after_first_failure": True,
        "stall_cycle_limit": 40,  # Stop after MORE than 40 consecutive low-progress windows; 0 disables.
        "stall_window_steps": 35,
        "stall_max_abs_x_progress_m": 0.02,
        "start_global_step": 0,
        "selected_global_step": None,
        "only_latest": False,
        "metrics_filename": "metrics_fixed_phase_one_episode.csv",
        "inherit_training_config": True,
        "allow_legacy_without_snapshot": True,  # 旧模型无快照时明确警告并使用当前配置
    },
}
