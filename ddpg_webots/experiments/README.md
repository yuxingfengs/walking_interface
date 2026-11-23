# DDPG-Webots 实验索引

在模块的 `config.json` 中用目录名指定 `current_experiment`。每个实验的实际输出位置见其 `runs_root` 和 `reports_root`。

| 配置 | 任务 |
|---|---|
| [cycle_mean_gaussian](cycle_mean_gaussian/config.json) | Cycle mean gyro Gaussian |
| [no_reward_pool](no_reward_pool/config.json) | Cycle mean gyro Gaussian |
| [step_narrow_gaussian](step_narrow_gaussian/config.json) | Step narrow gyro EMA Gaussian alpha0.40 |
| [ema_alpha_040](ema_alpha_040/config.json) | Step narrow gyro EMA Gaussian alpha0.40 |
| [step_wide_gaussian](step_wide_gaussian/config.json) | Turn and Walk - forward lateral continued 30k to 60k |
| [bidirectional](bidirectional/config.json) | Step gyro Gaussian without heading penalty - alternating turns |
| [gradual_reversal](gradual_reversal/config.json) | Step wide Gaussian - gradual reversal |
| [turn_and_walk](turn_and_walk/config.json) | Turn and Walk - step wide Gaussian |
| [forward_lateral](forward_lateral/config.json) | Turn and Walk - forward lateral curriculum |
| [fresh_50k](fresh_50k/config.json) | Turn and Walk - forward lateral 50k |
| [resume_30k_to_60k](resume_30k_to_60k/config.json) | Turn and Walk - forward lateral continued 30k to 60k |
| [walk_only](walk_only/config.json) | Walk Only - legacy BC50 fixed forward 0.07 |
| [forward_fixed_008](forward_fixed_008/config.json) | Walk Only - fixed forward 0.08 |
| [forward_speed](forward_speed/config.json) | Walk Only - forward variable speed |
| [legacy_linear_bc50](legacy_linear_bc50/config.json) | Walk Only - legacy rates and retrained BC50 |
| [legacy_linear_bc50_v070](legacy_linear_bc50_v070/config.json) | Walk Only - legacy BC50 fixed forward 0.07 |
| [legacy_linear_walk](legacy_linear_walk/config.json) | Walk Only - legacy linear expert speed |
| [turn_fixed_positive](turn_fixed_positive/config.json) | Step gyro Gaussian without heading penalty |
| [cycle_mean_with_pool](cycle_mean_with_pool/config.json) | Cycle mean gyro Gaussian |
| [turn_ema010](turn_ema010/config.json) | Step narrow gyro EMA Gaussian |
| [walk_initial](walk_initial/config.json) | Walk Only |
