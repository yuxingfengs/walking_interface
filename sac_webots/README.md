# SAC-Webots

本模块的 `config.json` 选择实验（`current_experiment`）及续训来源（`resume_from`）。根入口为 `Start Training.cmd`、`Continue Training.cmd` 和 `Check Training.cmd`。各后端接受的存档类型见[根说明](../README.md)。

双击 `Start Training.cmd` 启动 Webots，双击 `Check Training.cmd` 只加载模型、渲染 world 并检查零更新状态。本入口始终选择 Webots。

`launch.ps1` → `train_sac.py` → `training/scripts/training.py` → 每个 run 的冻结 controller。训练使用本模块的 SAC、Replay、Reward 调用和任务阶段逻辑。motor、IMU、FSR、Supervisor 反馈及场景模板逻辑来自 `environments/webots/`。

模块入口从同目录的 `config.json` 选择实验。每个源实验的 `config.json`、`training_config.json`、`reward.py` 和 BC manifest 位于 `sac_webots/data/experiments/<id>/`。004 自身指定 100,000 步，优先于项目的 Webots 默认预算。

在模块 `config.json` 的 `resume_from` 填入原 run 目录，再双击 `Continue Training.cmd`。中断后会恢复到已保存的 episode 边界。新 run 跑满后会另存最后一步的完整状态。例如 004 已完成 100,000 步，要再跑 20,000 步，就将 `data/experiments/004/config.json` 的 `webots_total_env_steps` 改为 `120000`。续训从第 100,001 步开始，原预算的结果保存在同目录的 `result_100000_*.json`。

恢复会校验有效配置、原 world 的 SHA-256 以及训练日志与 checkpoint 的步数。保留该 run 及 `sac_webots/data/training/.webots_runs/` 中对应的 world。每步 SAC 指标和最终报告分别位于 run 的 `training.csv` 和 `report/`。
