# Published experiment: gs330

Training and full Evaluation completed. This release contains only replay_retained_half_noise_actor_decay. See the repository README for current status, results and portable evaluation. Earlier prepared/not-started descriptions below are historical. BC is now assets/actor_bc_heading27.pth.

# 保留 Replay 的独立行走工程

工程入口：humanoid_sprint/start_training.cmd。

当前实验：replay_retained_half_noise_actor_decay。已建立并通过检查，尚未训练。

本工程是 humanoid_walk_heading_rate 的独立代码与资源副本，训练和 BC 模型加载不依赖旧工程。用户本次明确要求建立新工程，取代此前只保留两个工程的数量限制；旧工程完整保留且未修改。

## 训练设计

| 项目 | 第1–100000步 | 第100001–200000步 |
|---|---|---|
| Actor LR | 1e-5 | 1e-6 |
| Critic LR | 1e-6 | 1e-6 |
| Gaussian noise std | 0.002 rad | 0 |
| BC恢复 | 保留Replay | 保留Replay |

无归一化 Critic、27D Observation、10D Action，保持现有 heading reward、跌倒惩罚和 Expert。Actor仍从本地BC模型初始化，Critic从头训练。总预算200000环境步，前50轮Critic-only warmup、每35步更新，之后每350步更新。

BC恢复仍重置Actor、Actor target及Actor optimizer；保留Critic及其optimizer和Replay。Replay容量仍是2000，正常按容量淘汰旧数据，并非永久保存所有失败。恢复当步仍跳过更新、重新计350步更新间隔。Webots同次运行的world reload继续携带Replay；普通Checkpoint跨运行resume未新增Replay持久化。

## 入口

- humanoid_sprint/start_training.cmd：训练结束后快速报告。
- humanoid_sprint/start_evaluation.cmd：独立Evaluation结束后完整报告。
- humanoid_sprint/show_training_report.cmd / show_evaluation_report.cmd：查看报告。
- 参数：humanoid_sprint/configs/project_config.py。

## 对照与验证

直接上一版为 improve_head_reward_1_noise2_critic1e6_no_norm，实际Actor/Critic均1e-6、全程noise0.002、BC恢复清空Replay。已复制上一版冻结配置和训练CSV供快速报告比较，未复制历史训练模型。

固定baseline仍为根工程git_sin_training中的pre_s1_20260724_v1；报告仅将其作为只读比较资料。上一版未提供独立Evaluation，本版也未训练，两个性能比较均待完成。到达时间、带符号/绝对横向偏移、净前进、成功率、跌倒率必须由同次Checkpoint Evaluation给出；历史x4.5与当前x5需注明不可直接比较。

40项测试和启动预检查通过，覆盖失败/停滞恢复保留Replay、噪声及LR半程边界、恢复后的LR、world reload、无归一化及报告编排。配置差异、源文件hash和BC hash见当前实验design目录。
