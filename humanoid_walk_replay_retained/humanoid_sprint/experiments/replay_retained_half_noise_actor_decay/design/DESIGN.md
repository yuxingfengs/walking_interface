# Replay 保留与半程探索实验

授权：用户明确要求独立于humanoid_walk_heading_rate建立本版本，并同时实施三项改动。当前仅准备完成，未启动。

直接上一版：improve_head_reward_1_noise2_critic1e6_no_norm，run train_20260915T184942_563909Z_617402。previous_run_config.json、previous_config.json和previous_project_config.py保存来源。旧文档描述的critic1e5版本不是本次直接上一版，以实际最近训练为准。

研究改动：

1. initialization.clear_replay_on_bc_reset=False。保留当前失败transition及其余Replay，包括停滞触发恢复。Actor/Actor target/Actor optimizer仍恢复，Critic仍保留。更新计数器和恢复当步跳过更新行为沿用。
2. noise_switch_env_steps=100000，后半程action_noise_std_after_switch=0；前半程0.002rad。关闭时不消耗探索RNG。
3. lr_switch_env_steps=100000，Actor1e-5转1e-6；Critic全程1e-6。按全局累计环境步切换，恢复BC后也使用当前阶段LR。

参考设计：[DDPG Replay复用与off-policy更新](https://spinningup.openai.com/en/latest/algorithms/ddpg.html)。保留失败经验旨在让Critic有机会再次抽样学习；后半程减小Actor步幅和停止外加探索是用户指定的实验日程，不是论文推荐的固定最优超参数。三项组合实验不能单独归因每项效果。

独立工程内含代码、世界、纹理、protos及本地BC副本，无训练运行路径指向旧工程。固定baseline只读比较仍位于共享根目录。当前配置/差异/hash/验证结果均保存在本目录。source_hashes.json可检查新工程实现；original_source_hashes.json核验旧工程原始源码未变化。

40项测试通过，launch.py train --check通过。尚无本版训练或独立Evaluation，当前vs上一版、当前vs固定pre_s1_20260724_v1均待比较；缺少新到达时间、带符号/绝对横向偏移、净前进、成功率和跌倒率。历史x4.5/当前x5协议差异及训练预算差异须保留说明。
