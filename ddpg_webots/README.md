# DDPG-Webots

本模块的 `config.json` 选择实验（`current_experiment`）及续训来源（`resume_from`）。根入口为 `Start Training.cmd`、`Continue Training.cmd` 和 `Check Training.cmd`。各后端接受的存档类型见[根说明](../README.md)。

每个子目录的 `Start Training.cmd`、`Check Training.cmd`、`Pretrain Critic.cmd` 和 `Report.cmd` 均使用对应的 `ddpg_webots/experiments/<实验名>/config.json`。先双击 `Check Training.cmd` 可核对所选配置、预训练 Critic、Observation 和输出路径。

| 入口目录 | 任务 |
|---|---|
| `cycle_mean_gaussian/` | gait cycle 平均角速度的 Gaussian Reward |
| `step_wide_gaussian/` | 逐步宽 Gaussian Reward，具体变体由当前配置指定 |
| `step_narrow_gaussian/` | 逐步窄 Gaussian Reward 及配置选定的 EMA |
| `walk_only/` | 平面 Walk 速度跟踪 |

`launch.ps1` 根据入口选择 `train.py`、`pretrain.py` 或 `report.py`。`training/basewalk/` 管理 DDPG、Replay、Reward 调用和 reset 后的训练状态。`environments/webots/` 提供设备初始化、传感器与 Supervisor 数据采样、诊断和场景渲染。生成 controller 时这些依赖一起复制到 run 的 `source/basewalk/`。

在 `ddpg_webots/experiments/<实验名>/` 编辑配置并查看 `DESIGN.md`。实际输出目录以 `Check Training.cmd` 显示的 `Runs` 为准，有些变体写入 `variants/`。原有 world reload 继续使用同一 run 的 `world_reload_state.pth`，包括 Replay、优化器、随机状态和阶段计数。

从已完成的 run 追加训练时，在所选实验配置中提高生效的 `total_env_steps`，再将模块 `config.json` 的 `resume_from` 指向最后一个 `checkpoints/step_*.pth`，双击 `Continue Training.cmd`。选择 `step_00020000.pth` 这样的数字步数文件。新 run 跑满时会在旁边生成 `continuation_00020000.pth`，续训自动读取其中的 Replay、随机状态和任务阶段。

纯旋转、纯 Walk、Turn/Walk 与 forward/lateral 等历史选择项及对应 run 配置见[实验索引](experiments/README.md)。每个目录保留对应的 Reward。
