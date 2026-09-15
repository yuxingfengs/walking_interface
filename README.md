# NAO 行走：保留 Replay 的 DDPG 实验

本分支仅保留训练出 **gs000330** 的完整实验 `replay_retained_half_noise_actor_decay`，不是只发布一个模型文件。其他实验工程已从 main 的当前目录树移除；Git 历史及其他远端分支不改写。

## 保留内容

- 独立工程 `humanoid_walk_replay_retained/humanoid_sprint`：训练代码、配置、世界和纹理、Expert、BC输入模型。
- 本实验56个候选Checkpoint、最终恢复Checkpoint、训练日志、训练报告、完整Evaluation及专项Expert比较。
- 推荐速度模型 `ckpt_gs000330.pth`，最直模型 `ckpt_gs000250.pth`；同属这一次实验。
- 固定历史baseline及直接上一版训练CSV只作为比较资料，不包含旧实验工程。

## 实验设计

Critic无归一化，Critic LR全程1e-6；Actor LR前100000步1e-5、后100000步1e-6；探索噪声前半程0.002rad、后半程0；BC恢复保留Replay。27D Observation、10D Action，Replay容量2000。训练总预算200000环境步。

gs000330保存于102268环境步，它是完整Evaluation中最快的成功模型，并非最终训练模型。

## 同条件独立比较

| 模型 | 到达时间/s | 带符号横向净偏移/cm | 绝对偏移/cm | 净前进/m | 成功 | 跌倒 |
|---|---:|---:|---:|---:|---:|---:|
| gs000330 | 110.64 | -35.10 | 35.10 | 10.0291 | 1/1 | 0/1 |
| gs000250 | 115.00 | +29.55 | 29.55 | 10.0262 | 1/1 | 0/1 |
| BC | 114.96 | +50.73 | 50.73 | 10.0217 | 1/1 | 0/1 |
| 规则Expert | 115.20 | +49.66 | 49.66 | 10.0273 | 1/1 | 0/1 |

每组为相同起点、关闭噪声的一次测试，不能推出扰动下的稳定成功率。固定baseline终点为x4.5、本次为x5，不直接比较时间；直接上一版独立Evaluation缺失，仍待比较。

[完整比较与轨迹](humanoid_walk_replay_retained/humanoid_sprint/experiments/replay_retained_half_noise_actor_decay/analysis/best_vs_expert_20260915T205920Z/REPORT.md)

## 运行

1. 安装Webots和Python环境，需要PyTorch、NumPy、Matplotlib及测试/报告所需库。原实验使用Webots R2022b资源；切换仿真版本可能影响复现。
2. 在 `humanoid_walk_replay_retained/humanoid_sprint/configs/project_config.py` 设置本机Webots executable和Python路径。它们不是可移植的软件安装路径。
3. 复测gs330：进入工程目录，运行 `start_gs330_evaluation.cmd`。它从本地发布模型和当前配置建立新世界，不依赖历史记录中的原机器绝对路径；不训练，结果保存在 `.local_evaluation`。
4. 重新训练：先修改 `experiment.name` 为新的名字，再运行 `start_training.cmd`。训练结束自动生成快速报告；训练后可单独运行 `start_evaluation.cmd` 做新实验完整评估。不要覆盖已发布实验目录。

`assets/actor_bc_heading27.pth` 是必须保留的BC输入。迁移时仅调整当前配置中的BC文件位置和报告baseline路径；原训练快照、历史报告里的机器绝对路径及hash保留作溯源。历史HTML/JSON含原机器路径，在线阅读请使用各目录的REPORT.md和相对资源；不要将历史快照原路径当作新机器可用路径。

模型SHA256：`e467a02b77a2554761e2aa6222f67140b4cc8b1ca41a9a292da3400447f6e4d4`。
