# NAO 行走与转向训练

本项目包含 [DDPG-Webots](ddpg_webots/README.md)、[SAC-Webots](sac_webots/README.md) 和 [SAC-Isaac Lab](sac_isaac_lab/README.md) 三个独立训练模块。以下从一台新的 Windows 电脑开始，运行当前主实验 [walk_lateral_random_vx](sac_isaac_lab/experiments/training_trials/walk_lateral_random_vx/)。

该实验训练 NAO 全方位平面行走。每台机器人每 800 个本地控制步重新抽取一次前后速度 `vx`（−0.11 至 +0.11 m/s），每 100 步切换一次侧向速度 `vy`（0、+0.04、+0.08、+0.04、0、−0.04、−0.08、−0.04 m/s）。两者组合产生斜向行走指令，目标转速 `wz = 0`。

## 1. 克隆仓库并取得初始模型

先安装 Git 和 Git LFS，然后在 Windows PowerShell 中运行。后面的命令都以克隆目录为工作目录。

```powershell
git lfs install
git clone --branch nao-gym-interface https://github.com/yuxingfengs/walking_interface.git walking2_interface
Set-Location .\walking2_interface
git lfs pull
Test-Path '.\sac_isaac_lab\Start Training.cmd'
```

最后一行应返回 `True`。如果返回 `False`，说明当前检出的版本没有这套训练入口，请向维护者取得包含重构目录的分支、提交或项目包，再继续。

**新开 run 时，训练入口自动查找并校验前置实验 `walk_lateral` 训练生成的 50,000 步 SAC Checkpoint，用它初始化模型。** 该文件随 Git LFS 提供。Walk BC 目录内还需有 `manifest.json` 和其指定的 Actor。

```text
sac_isaac_lab/experiments/training_trials/walk_lateral/runs/train_20261122T100335Z_b14a36/checkpoints/step_00050000.pth
sac_isaac_lab/experiments/training_trials/walk_only/bc/webots_forward/
```

50,000 步文件的 SHA-256 是 `fb069874b45978a661db73d091abc36521e0448b0d231491c02d09a1bfb76ae1`。使用 `Get-FileHash <文件路径>` 可核对。要接着别人的模型训练，还需取得要继续的原 run，包括训练日志、冻结配置、source 和完整恢复档。

## 2. 在项目外安装 Isaac

下面以 `D:\Isaac\sim-5.0.0` 为 Sim 安装目录、`D:\Isaac\nao-runtime` 为运行环境目录。换成自己的磁盘路径即可。

1. 准备 Windows、受 Isaac Sim 5.0 支持的 NVIDIA GPU 和驱动，以及 Git。下载 **Isaac Sim 5.0.0 Windows standalone**，解压到能直接看到 `python.bat`、`kit/`、`site/` 的目录。本项目按 **Isaac Lab v2.2.1、Python 3.11** 配置。[NVIDIA 二进制安装说明](https://isaac-sim.github.io/IsaacLab/v2.2.1/source/setup/installation/binaries_installation.html)
2. 使用 Sim 随附的 Python 创建隔离环境，取出对应版本的 Lab 源码。以下命令用于首次安装，保留同一个 PowerShell 窗口。

```powershell
$simDir = 'D:\Isaac\sim-5.0.0'
$runtimeDir = 'D:\Isaac\nao-runtime'
New-Item -ItemType Directory -Force $runtimeDir | Out-Null
git clone --branch v2.2.1 --depth 1 https://github.com/isaac-sim/IsaacLab.git "$runtimeDir\IsaacLab"
& "$simDir\kit\python\python.exe" -m venv --without-pip --system-site-packages "$runtimeDir\python"
$isaacPython = "$runtimeDir\python\Scripts\python.exe"
[IO.File]::WriteAllText("$runtimeDir\python\Lib\site-packages\isaac_binary.pth", "$simDir\site")
& $isaacPython -m pip --version
& $isaacPython -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

选择 GPU 训练时，最后一项应为 `True`。隔离环境通过 `isaac_binary.pth` 加载这份 Sim 提供的 Python 模块和 PyTorch。

## 3. 配置

### [训练入口 · sac_isaac_lab/config.json](sac_isaac_lab/config.json)

| 关键参数 | 用途 |
|---|---|
| `current_experiment` | 选择启动哪个实验。 |
| `continue_training.run` | Continue 使用的 run。默认 `latest` 选择当前实验导出的最新模型所属的 run，也可填写 `train_<id>` 或 `config_<hash>/train_<id>`。 |
| `continue_training.checkpoint` | 默认 `latest.pth` 使用该 run 导出的模型。可填写 `recovery_latest.pth` 恢复本地 Replay，或填写某个 `step_*.pth` 文件名。续训需有该 run 的 `training.csv`。 |
| `runtime_root`、`kit_state_root` | 前者定位项目外的 Isaac 运行环境，后者指定 Kit 用户状态的存放位置。 |
| `sim_device`、`learner_device` | 分别指定运行物理仿真和更新 SAC 网络的设备。 |
| `num_envs`、`headless` | 前者按实验设置并行机器人数，后者控制是否隐藏仿真窗口。 |

### [Isaac 仿真环境 · simulator_config.json](environments/isaac_lab/setup/simulator_config.json)

| 关键参数 | 用途 |
|---|---|
| `isaac_sim_path`、`isaac_lab_path` | 分别定位 Isaac Sim 安装目录和 Isaac Lab 源码目录。 |
| `isaac_python` | 指定训练与离线检查共用的 Python 解释器。以 `runtime/` 开头的路径从 `runtime_root` 解析。 |
| `asset`、`urdf_source`、`nao_meshes_path` | 分别定位仿真加载的 NAO 模型、用于生成该模型的源 URDF 和机器人网格目录。 |

### [SAC 共享训练参数 · training_config.json](sac_isaac_lab/training_config.json)

| 关键参数 | 用途 |
|---|---|
| `seed` | 设置训练随机种子。 |
| `total_env_steps` | 设置全方位行走实验的总训练预算，按所有并行机器人的环境步数合计。 |
| `sac.actor_lr`、`sac.critic_lr` | 分别设置 Actor 和 Critic 的学习率。 |
| `training.batch_size`、`training.replay_size` | 分别设置每次更新采样的经验条数和 Replay Buffer 的容量。 |
| `training.checkpoint_every` | 设置每隔多少个环境步保存一个编号 Checkpoint。 |

## 4. 安装项目依赖并验证

```powershell
& '.\environments\isaac_lab\setup\Install Isaac Dependencies.cmd'
& '.\sac_isaac_lab\Check Training.cmd'
& '.\environments\isaac_lab\diagnostics\Check Isaac Runtime.cmd'
```

第一条向已创建的环境安装 Lab 核心包、h5py 和项目所需的 Warp 版本及 Kit 扩展。第二条检查所选实验的模型、配置和资产，确认输出含 `offline_contract_passed`、`runtime_ready: true` 和 `asset_ready: true`。第三条实际启动 Isaac 空场景，确认 `status` 为 `passed`。若缺少 NAO 几何资源，先补齐 `environments/isaac_lab/assets/nao_meshes/` 和源 URDF，然后运行 `environments/isaac_lab/setup/Prepare NAO Asset.cmd`。

## 5. 启动训练和打开报告

| 目的 | 在仓库根目录的 PowerShell 中运行 |
|---|---|
| 用实验指定的初始模型新开独立 run | `& '.\sac_isaac_lab\Start Training.cmd'` |
| 接着已有 run 训练 | 在模块 `config.json` 的 `continue_training` 中选择 run 和 Checkpoint，运行 `& '.\sac_isaac_lab\Continue Training.cmd'`。 |

跑满后追加步数时，提高 `sac_isaac_lab/training_config.json` 里的 `total_env_steps`。例如已完成 260,000 步，再训练 20,000 步就改为 280,000，再运行 Continue。历史恢复档会校验冻结配置，迁移到另一台电脑还需核对其中的绝对路径和资产。

**自己的英文报告**在训练结束时打印的 `SAC report:` 路径。新 run 的目录结构如下：

```text
sac_isaac_lab/experiments/training_trials/walk_lateral_random_vx/
  training.py                     速度指令及初始模型查找逻辑
  runs/config_<hash>/train_<id>/
    effective_config.json         本次实际配置
    training.csv                  每步训练记录
    checkpoints/                  模型与恢复档
    report/index.html             可直接用浏览器打开的英文报告
```

## 其他项目目录

| 目录 | 内容 |
|---|---|
| [environments/](environments/) | Webots 与 Isaac Lab 的仿真逻辑、机器人资产、安装及诊断工具 |
| [sac_isaac_lab/runner/](sac_isaac_lab/runner/) 与 [training/](sac_isaac_lab/training/) | Isaac 实验的配置解析、训练入口、并行采样、SAC 和报告代码 |
| [DDPG-Webots 实验](ddpg_webots/experiments/README.md) | 各实验配置、初始化资产和运行入口，结果位置见所选实验的 `runs_root` |
| [SAC-Webots 实验](sac_webots/data/experiments/) | 各实验的配置、Reward、BC 与 run，训练入口位于 [sac_webots/](sac_webots/) |
| [expert/](expert/) | 专家示范及派生数据 |
