# Published experiment: gs330

Training and full Evaluation completed. This release contains only replay_retained_half_noise_actor_decay. See the repository README for current status, results and portable evaluation. Earlier prepared/not-started descriptions below are historical. BC is now assets/actor_bc_heading27.pth.

# 工程状态

当前状态：prepared_not_started。

用户批准的独立版本包含：BC恢复保留Replay、后半程noise=0、Actor LR前半1e-5后半1e-6。其余沿用直接上一版无归一化Critic。旧humanoid_walk_heading_rate工程不修改。

完成：复制独立运行资源及BC、修改配置与恢复逻辑、40项测试、启动预检查、配置溯源登记。

待完成：启动训练、训练快速报告、独立Evaluation及完整比较。未取得新性能结果。
