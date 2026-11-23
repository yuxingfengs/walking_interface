# SAC-Isaac Lab

## 文件位置

| 用途 | 位置 |
|---|---|
| 安装 Isaac 依赖、配置 Warp、安装 NAO 网格、准备 URDF、迁移运行时 | [环境安装工具](../environments/isaac_lab/setup/) |
| 配置仿真步长、关节驱动及 Isaac 路径 | [simulator_config.json](../environments/isaac_lab/setup/simulator_config.json) |
| 检查 Isaac 运行时和初始站姿 | [运行时检查](../environments/isaac_lab/diagnostics/) 和 [站姿检查](../environments/isaac_lab/check_standing.ps1) |
| 选择实验、续训 checkpoint 和设备 | [config.json](config.json) |
| 设置 SAC 超参数 | [training_config.json](training_config.json) |
| 训练启动实现、并行训练和离线检查 | [runner](runner/) |
| 回归测试、Slurm 提交与实验监控 | [Isaac environment tests](../environments/isaac_lab/tests/), [SAC tests](tests/) 和 [scripts](scripts/) |

从仓库根目录安装依赖时运行 `& '.\environments\isaac_lab\setup\Install Isaac Dependencies.cmd'`。日常训练仍使用本目录的 `Start Training.cmd`、`Continue Training.cmd` 和 `Check Training.cmd`。

本模块的 `config.json` 选择实验（`current_experiment`）和 Continue 的来源（`continue_training.run`、`continue_training.checkpoint`）。默认值 `latest` 和 `latest.pth` 对应当前实验导出的最新模型。指定历史 run 时，填写相对于该实验 `runs/` 的目录名，例如 `train_20261123T024754Z_1f09e7`。入口为 `Start Training.cmd`、`Continue Training.cmd` 和 `Check Training.cmd`。

本机主实验是 [walk_lateral_random_vx](experiments/training_trials/walk_lateral_random_vx/)。[根 README](../README.md) 提供新开 run、续训以及最近已完成的英文报告入口。若需显式指定实验，可在仓库根目录使用以下 PowerShell 命令，分别进行离线检查、固定策略 PhysX 检查和正式训练。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\sac_isaac_lab\launch.ps1 -Experiment walk_lateral_random_vx -Mode offline
powershell -NoProfile -ExecutionPolicy Bypass -File .\sac_isaac_lab\launch.ps1 -Experiment walk_lateral_random_vx -Mode check -Headless
powershell -NoProfile -ExecutionPolicy Bypass -File .\sac_isaac_lab\launch.ps1 -Experiment walk_lateral_random_vx -Mode train -Headless
```

`runner/configuration.py` 从 `environments/isaac_lab/setup/simulator_config.json`、SAC 源实验和变体配置构造有效配置。`omni` 使用源实验 002 的 BC 和本实验的 Reward 与课程。`runner/main.py` 冻结本模块的 `training/`、实验 Reward、模型和 `environments/isaac_lab/` 中的物理环境，再运行固定策略检查和所选模式。仿真资产位于仓库根目录下的 `environments/isaac_lab/assets/`，Kit 状态位于 `runtime/`，实验配置和新 run 位于本模块的 `experiments/`。

## 本机机器人数量和设备

在本目录的 [config.json](config.json) 中，`num_envs.walk_lateral_random_vx` 和 `num_envs.walk_lateral` 分别设置两个横移实验的并行机器人数量，默认各为 4。其他实验的本机训练入口使用 1 个机器人。`sim_device` 设置 PhysX 设备，`learner_device` 设置 SAC 网络和优化器的设备，两项各自可填 `cuda:0` 或 `cpu`。`headless` 控制窗口显示。Check Training 会显示下次启动的设备与机器人数量，离线检查本身在 CPU 上运行。Slurm 多卡入口单独通过 `--gpus` 选择 collector 数量。

### walk_lateral_random_vx 的超参数

| 文件 | 生效字段 |
|---|---|
| [全方位行走训练文件](experiments/training_trials/walk_lateral_random_vx/training.py) | 指令保持步数、随机前后速度范围和横向速度序列。启动时自动定位前置实验生成的 50,000 步 SAC 模型。 |
| [Walk 基础配置](experiments/training_trials/walk_only/config.json) | 速度跟踪尺度、每个 Action 的物理子步数、关节伺服和 BC manifest |
| [Isaac 公共训练超参数](training_config.json) | `seed`、`sac` 下的学习率、网络、gamma、tau 和熵参数，以及 `training` 下的 Batch Size、Replay Buffer 容量和 checkpoint 间隔 |

七个 Isaac 实验共用模块根目录的 `training_config.json`。其中 `sac.actor_lr` 设置共享 Actor 学习率，顶层 `total_env_steps` 设置 `walk_lateral_random_vx` 的总训练预算。每个 run 的快照在 `source/isaac_training_config.json` 中保留当时的公共参数。

每次启动的实际取值记录在对应模型或 session 的 `effective_config.json`。总环境步数按所有机器人产生的 transition 合计。恢复 checkpoint 时会校验原训练配置，包括机器人数量。

## 训练报告

七个实验共用 [tracking_report.py](training/scripts/tracking_report.py)。本机和多 GPU 的 run 完成后自动写入 `<run>/report/index.html`，Continue 跑完后根据完整日志更新。报告可离线打开。要单独为已完成的 run 生成报告，可从仓库根目录运行 `python sac_isaac_lab/training/scripts/tracking_report.py "<run目录>"`。

速度图可选择机器人和训练步数范围，逐条对比目标速度与该指令段的实际平均速度。每 1,000 个全局环境步汇总 vx、vy 的逐步绝对误差和平面速度误差，Turn 另列 wz 误差。报告同时给出窗口样本数、摔倒次数、平均 Reward 和指令覆盖。

报告的首末比较使用前 25,000 步与最后 10,000 步，列出误差均值和相对下降率。满 35,000 步时两个区间互不重叠，初期误差为零的指标只显示均值。

## Checkpoint 恢复

同一套训练配置共用 `runs/config_<hash>/`。每点一次 `Start Training.cmd`，就在其中创建一个独立的 `train_<id>/` 模型目录。使用 `Continue Training.cmd` 时，同配置的模型继续写原目录的 `training.csv` 和 `checkpoints/`。本次启动的诊断、快照和恢复信息保存在模型目录或其 `sessions/` 下。

恢复时会核对 checkpoint 的步数和更新次数。若 CSV 超前，超出的行保存在本次 session 的 `training_after_checkpoint.rows`；若回到较早步骤，后续模型会归档到 `previous_checkpoints/`。完整恢复文件包含 Replay 和随机状态。仅提高总预算并 Continue 时，继续写入所选模型目录。新 run 跑满后会把最后一步的 Replay、优化器和随机状态保存到 `checkpoints/recovery_latest.pth`，下次从该步之后开始。首次从旧布局续训时会导入所选模型的历史，旧路径仍可读取。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\sac_isaac_lab\launch.ps1 -Experiment walk_lateral_random_vx -Mode train -Headless -Resume '.\sac_isaac_lab\experiments\training_trials\walk_lateral_random_vx\runs\config_<hash>\train_<id>\checkpoints\recovery_latest.pth'
```

本机主实验的单卡输出位于 `sac_isaac_lab/experiments/training_trials/walk_lateral_random_vx/runs/`。保留原 run 的 `training.csv` 供恢复验证。常规 step checkpoint 的恢复沿用原有空 Replay 规则。

## GPU 并行与故障诊断

Linux/Slurm 从仓库根目录提交，预先配置现有的 `ISAACLAB_PATH` 和 Isaac Python 环境。显式指定 `--experiment omni`，多卡入口省略此参数时默认选择 `omni_step`。

```bash
sbatch --gpus=2 sac_isaac_lab/scripts/train_multigpu.sbatch --experiment omni --gpus 2
```

每个可见 GPU 上运行一个独立 headless worker，所有 worker 向一个 learner 提供 transition。多卡恢复向同一入口传入 `--resume <multi_gpu_run>/checkpoints/recovery_latest.pth`。保持原 collector 数量与任务配置。单卡和多卡各使用自己的 checkpoint 协议。

`runner/walk_batch.py` 管理同一设备上的 SAC 采样与更新。`environments/isaac_lab/batched_environment.py` 管理并行机器人、异常 lane 的物理状态记录与 reset 重试。`runner/multigpu_worker.py` 保留分 GPU 的检查和错误回传。诊断时先看 run 的 `status.json`，再看对应模式生成的 reset 记录、worker 目录和 Kit 日志。多卡输出位于相应实验的 `multi_gpu_runs/`。

## 可选实验

设置本目录 `config.json` 的 `current_experiment`。

| 实验 | 分类目录 | 训练内容 |
|---|---|---|
| [`004`](experiments/training_trials/004/) | `training_trials` | 连续正反转 yaw，保留站姿 reset |
| [`omni`](experiments/poor_performance/omni/) | `poor_performance` | 继承 SAC 002 的 BC，分阶段加入斜行、侧行和后退，Walk 和 Turn 交替 |
| [`omni_step`](experiments/poor_performance/omni_step/) | `poor_performance` | omni 系列，阶梯式 yaw 指令 |
| [`walk_only`](experiments/training_trials/walk_only/) | `training_trials` | 固定前向速度 |
| [`walk_variable`](experiments/training_trials/walk_variable/) | `training_trials` | 分段改变前后速度 |
| [`walk_lateral`](experiments/training_trials/walk_lateral/) | `training_trials` | 固定前进速度，改变侧向速度 |
| [`walk_lateral_random_vx`](experiments/training_trials/walk_lateral_random_vx/) | `training_trials` | 每周期随机前后速度，叠加侧向速度序列 |


## 本机资产与安装

在 `environments/isaac_lab/setup/simulator_config.json` 设置 Isaac Python、Isaac Sim、Isaac Lab 与 mesh 路径。同一 `setup/` 目录中的 `Install Isaac Dependencies.cmd` 安装依赖，`Install NAO Meshes.cmd` 安装几何资源，`Prepare NAO Asset.cmd` 准备 URDF。`environments/isaac_lab/diagnostics/Check Isaac Runtime.cmd` 运行空场景检查，`environments/isaac_lab/Check Isaac Standing.cmd` 检查站姿。

在 PowerShell 运行 `& '.\environments\isaac_lab\setup\Move Isaac Runtime.cmd'` 可将软件安装到 `F:\issac\nao-runtime`。脚本使用现有依赖离线重建 Python 环境，校验完成后更新 `config.json` 的 `runtime_root`。可用 `--destination` 指定其他项目外目录。Kit 用户脚本与设置单独由 `kit_state_root` 定位。
