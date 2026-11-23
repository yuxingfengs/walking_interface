# Shared expert demonstrations

DDPG和SAC共同从本目录读取专家数据，Git中需要包含本目录。

本机维护的采集标准、复用条件和补采判据保存在 `expert/docs/DATA_REUSE.md`，该文档仅在本地提供。

各Reward版本的expert.pth各有20,165条记录，物理State、实际Action、终止、Mode和周期边界的hash一致；区别在派生分数。保持各文件hash，不把某一版Reward标签冒充另一版。

- cycle_mean_gaussian：周期均速高斯派生数据，供DDPG周期版使用。
- step_wide_gaussian：去航向项逐步宽高斯派生数据，供DDPG单步版使用。
- step_narrow_gaussian：gyro Z EMA α0.10的逐步窄高斯派生数据，峰值2、宽0.08。
- step_narrow_gaussian/ema_alpha_040：同一物理轨迹改用α0.40；仅State的派生EMA列和Reward标签改变，原始传感器、Action与边界不变。与α0.10的全量State hash会不同，不应将EMA列变化误称为新物理采集。
- sac_turn：SAC的BC原始来源；已有Actor当时筛选1,187条正转样本拟合，在线SAC不把此专家文件加入Replay。

catalog.json记录文件hash、共同物理hash和历史来源。历史模型内路径仅作来源记录；活动配置使用本目录的相对路径。
