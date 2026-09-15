# 训练设计比较记录

## 2026-09-15 Independent retained-Replay project

User explicitly authorized a new independent project `humanoid_walk_replay_retained/humanoid_sprint`, superseding the earlier two-project limit. Experiment `replay_retained_half_noise_actor_decay`; direct previous `improve_head_reward_1_noise2_critic1e6_no_norm` (completed 200000 steps, actual Actor/Critic1e-6). Old heading-rate project unchanged. New project keeps Replay on BC recovery, uses noise0.002 then0 and Actor1e-5 then1e-6 at100000/200000steps; no-normalization Critic stays1e-6. Same BC copied locally. 40 tests and launch preflight passed; prepared, not started. Training quick report and separate Evaluation full report entries included. Current-vs-previous and current-vs-fixed pre_s1_20260724_v1 pending new independent Evaluation with paired time/signed and absolute lateral/net forward/success/fall; x4.5 historical vs x5 current caveat remains. See humanoid_walk_replay_retained/PROJECT.md and active experiment design/DESIGN.md.

## 2026-09-15 No-normalization Critic experiment

Active configuration: `improve_head_reward_1_noise2_critic1e5_no_norm`; direct previous: `improve_head_reward_1_noise2_critic1e5`. Prepared, not launched. Only research change: remove both Critic BatchNorm layers using Identity, preserve hidden widths and action insertion. New plugin `algorithms.ddpg_no_norm.DDPGNoNormAlgorithm`, artifact id `ddpg_no_norm_v1`; old BN algorithm remains available. Same Actor/Critic LR1e-5, 200000steps, BC initialization/recovery, heading weight1, noise0.002, update interval350, tau1e-4. 17 tests and launch preflight passed. Configuration/source snapshots in active experiment design/. World differences are relative vs absolute resource paths only. Policy comparison vs direct previous and fixed pre_s1_20260724_v1 pending independent Evaluation; no new arrival time, signed/absolute lateral displacement, net forward, success or fall rate. Historical x4.5 vs current x5 protocol caveat remains.

## 2026-09-15 Critic LR 1e-5 experiment

User requested a separate Critic learning-rate experiment. Active `improve_head_reward_1_noise2_critic1e5`; direct previous `improve_head_reward_1_noise2`. Only effective change is Critic LR1e-6 to1e-5 in both schedule stages. Same Actor1e-5, heading weight1, noise0.002,200000steps,350joint interval,50/35warmup,Replay2000,batch64,tau1e-4,seed1,BC27/recovery. Config/model/world preflight passes. Run `train_20260915T152205_092883Z_b0a92c` launched through train-report; quick report after training, no automatic Evaluation. Current-vs-previous and current-vs-fixed pre_s1_20260724_v1 pending independent Evaluation, including paired arrival time/signed and absolute lateral/net forward/success/fall, with historical x4.5 vs current x5 caveat. Provenance and launch evidence: humanoid_walk_heading_rate/humanoid_sprint/experiments/improve_head_reward_1_noise2_critic1e5/design/DESIGN.md.

## 2026-09-15 Completed training log analysis and project cleanup

Both current walk_bc_strong_heading_70k and previous walk_bc_sincos_rate_heading_70k completed70000steps. Last10000-step sampled mean absolute heading:3.009768deg vs4.412568deg. For successful complete segments starting after35000steps, terminal x6 mean absolute heading2.148051deg vs4.491781deg; mean duration129.373333s vs128.732s. New training28falls(all by27458steps),12goals,4stalls,9BCrecoveries; previous0falls,11goals,10stalls,10recoveries. Ended-segment outcomes12/44goals and28/44falls vs11/21goals and0/21falls; this is changing-policy noisy training, not repeated fixed-model evaluation. Direction improved in training, early stability worsened; no monotonic learning claim. Logs lack worldY and exact per-segment initial coordinates, so signed/absolute lateral displacement and precise net forward cannot be supplied from these logs. Independent current-vs-previous and current-vs-fixed Evaluation remains pending. Full source-hashed report: humanoid_walk_heading_rate/humanoid_sprint/experiments/walk_bc_strong_heading_70k/analysis/training_heading/REPORT.md.

User then explicitly authorized removing all other projects. Retained only humanoid_walk_heading_rate and git_sin_training. Previous experiment identity retained via original CSV and comparison evidence inside the current analysis/reference; old models/projects removed from workspace. Historical paths in older sections below are provenance only and may no longer exist. Fixed baseline CSV recovered exactly from saved raw output and relocated to git_sin_training/baselines/pre_s1_20260724_v1; original SHA256 matches. Research documents retained in current project/research. Cleanup manifest and equivalent-Expert/config/model/source checks under current experiment analysis/cleanup. No new training or Evaluation launched.

## 2026-09-15 Stronger Expert BC and 70000-step DDPG launched

User approved launch. Active experiment `walk_bc_strong_heading_70k` in `humanoid_walk_heading_rate/humanoid_sprint`; direct previous `walk_bc_sincos_rate_heading_70k`. Expert feedback gain 0.3 -> 0.6, stride cap 0.02 -> 0.04, slew 0.0005/40ms retained. Matched yaw0.0837 tests: old Expert last-cycle heading2.718797deg vs strong1.660169deg; x5 lateral+0.600359875m vs+0.496603179m, both reached/no falls. New 8880-sample BC includes startup and balanced straight/left/right groups, 4500 independent validation samples, 500-epoch fine-tune from previous27D BC. New BC baseline `bc_strong_heading_20260915_v1`: x5 time114.96s, signed/absolute lateral+0.507275301/0.507275301m, net forward10.021676484m, reached1/1/fell0/1, last-cycle heading1.669728deg. Previous BC114.64s/+0.689699958m: new BC0.32s slower, lateral0.182424657m smaller. Single positive-heading BC trial only; no claim of learned right-heading performance or DDPG improvement. All10 Expert trials no falls;3 short validation trials intentionally end before finish. Same27D sin/cos heading+rate, heading reward0.1, fall/forward, noise0.002, LR/update/reset and70000steps. Run `train_20260915T012730_468996Z_942348` launched fast, initial PID20584; config/model preflight passes. DDPG current vs previous and fixed `pre_s1_20260724_v1` remain pending independent Evaluation with same-checkpoint time/displacement/success/fall and condition/budget caveats. Historical x4.5 differs from current x5. Previous experiments/baselines preserved. Report: `humanoid_walk_heading_rate/humanoid_sprint/experiments/walk_bc_strong_heading_70k/design/BASELINE_AND_RUN.md`.

## 2026-09-15 27D sin/cos heading + yaw rate training launched

User confirmed implementation and launch. New independent humanoid_walk_heading_rate/humanoid_sprint, experiment walk_bc_sincos_rate_heading_70k; direct previous walk_bc_three_experts_forward_fall_70k retained. Original24 inputs plus sin(error),cos(error),causal wrapped world-heading derivative (rad/s);27D Actor/Critic,10D Action,contract v6. Heading penalty0.1 restored;fall/forward retained,yaw rate observation only. Same70000steps,175joint interval,warmup50/35,LR/noise/reset/initialyaw. Converted8880 demonstrations and4500 held-out samples in original temporal order before mapping to shuffled dataset;27D BC fitted500epochs.36tests/preflight pass. New BC baseline bc_sincos_rate_20260915_v1 at matching yaw.0837/x5:114.64s,lateral+0.689699958m/absolute same,forward10.022120553m,1/1success0/1fall;old25D BC115.08s/+0.659579693m. NewBC faster.44s but larger lateral by.030120266m. Only one left-heading physical test. DDPG run train_20260915T011436_031185Z_686cab launched; no automatic newEVAL. Fixed pre_s1_20260724_v1 and oldexpert/BC baselines preserved;training comparisons pending independent Evaluation with same-checkpoint time/displacement and condition caveats. Original projects unchanged. Full report: humanoid_walk_heading_rate/humanoid_sprint/experiments/walk_bc_sincos_rate_heading_70k/design/BASELINE_AND_RUN.md.


## 2026-09-15 Three-expert BC baseline and DDPG launched

Active walk_bc_three_experts_forward_fall_70k in humanoid_walk_absolute_heading/humanoid_sprint. User accepted left-heading correction without requiring zero heading or further refinement. BC V2: original25D BC fine-tuned on8880 balanced straight/left-recovery/right-recovery samples including startup;4500 independent validation samples;best epoch442/500. Model SHA256 ab73c4ba2248d50c54baca28edc68ac33dad2e0b066f34af8f2f2916346b79b6. New baseline bc_three_experts_20260915_v2: actual initialyaw0.0837rad, x5 arrival115.08s, signed lateral+0.659579693m/absolute same, forward10.026920114m,1/1success0/1fall. FinalV2 only this physical start tested; V1 right-heading failure is retained and not hidden. Fixed pre_s1_20260724_v1 and oldexpert baseline preserved. Direct previous walk_bc_heading_state_forward_fall_70k retained. New DDPG run train_20260915T010229_121556Z_3bd2c6 launched:70000steps,175joint interval,warmup50/35,noise.002,forward/fallReward,noheadingpenalty,25DheadingState retained. Preflight passes. No automatic newDDPG EVAL; all training-performance comparisons pending independent Evaluation with matching time/displacement and budget/finish caveats. Full baseline/training report: humanoid_walk_absolute_heading/humanoid_sprint/experiments/walk_bc_three_experts_forward_fall_70k/design/BC_BASELINE.md.


## 2026-09-15 简单差分步幅Expert独立验证完成

用户授权sin左右步幅差转向试验。独立scripts/try_differential_expert.py在原10D动作中仅调HipPitch/AnklePitch的cos幅值；未修改原Expert、BC或活动70000步配置。初始yaw0、40秒u=0/±.01/±.02五组均无跌倒，±.02相对原sin完整周期heading变化-.665/+.620度，双向可控。比例反馈u=clip(-.3*error,±.02)，每40ms限变化.0005，初始±5度各与原sin对照120秒；末周期平均heading分别+2.8759/-2.8511度，原sin+5.5679/-5.2343度。四组均到达x5且无跌倒，新反馈±5均115.08s，到达同次横移+0.634037/-0.722207m、净前进10.024480/10.026821m；原sin115.12s、横移+0.891715/-0.928723m。9组总计0跌倒/0Action裁剪；各条件单次，不证明重复稳定性，反馈尚未对齐0。不是新BC/DDPG改进或固定baseline替换；起始角/预算不同需分开比较。原始轨迹、配置、源码、分析修正说明及报告：humanoid_walk_absolute_heading/humanoid_sprint/experiments/differential_expert/20260915T004751_296702Z/REPORT.md。


## 2026-09-15 保留方向State、前进/跌倒Reward，70000步（待启动）

用户已确认实施。活动实验 humanoid_walk_absolute_heading/humanoid_sprint/experiments/walk_bc_heading_state_forward_fall_70k；直接上一版walk_bc_absolute_heading_update175_140k已完成140000步/823更新，独立Evaluation待补。新配置保留25D绝对方向误差输入，heading_reward_weight=0，现有跌倒扣分保持（最终clip为-3），total_env_steps=70000；联合更新175、预热50次/35步、学习率35000步切换、BC、noise及reset均保留。配置/世界/模型预检和6项Reward/State检查通过。未启动新训练或Evaluation。前后配置、源码/BC校验值、验证结果与完整比较规则见新实验design/DESIGN.md。当前vs上一版/专家/固定pre_s1_20260724_v1均待独立Evaluation；时间、带符号/绝对偏移、净前进、成功/跌倒率待测。70k与140k预算及历史x4.5终点差异必须注明。


## 2026-09-15 update175 当前训练已启动

活动项目 humanoid_walk_absolute_heading/humanoid_sprint，实验 walk_bc_absolute_heading_update175_140k，run=train_20260914T233545_948861Z_086ac9。配置/世界/模型预检通过，已启动 Webots fast；140000 步，从 BC 开始，联合更新间隔175、Critic-only50次/间隔35，其余已批准设置保留。本次未改变研究代码或参数。前后配置、源码快照、BC留档及SHA256、运行和比较说明已补齐至该实验 design/TRAINING_RUN.md。直接上一版 walk_bc_absolute_heading_140k 保留；旧状态文件running但启动前无对应进程，不能视为正常完成。当前与上一版、专家、固定历史baseline的独立Evaluation均待比较；到达时间、带符号/绝对偏移、净前进及成功/跌倒率待测。历史x4.5与当前x5不直接作时间差。本轮不自动EVAL。


## 2026-09-15 Direct Actor restored (pending launch)

User approved walk_bc_direct_heading_140k in humanoid_walk_com. Direct previous walk_bc_residual_heading_140k retained. Full25D Actor initialized from originalsensor24BC via zero extra input column; all parameters trainable after existing50Critic-only warmup. No frozen BC backbone, residual branch or0.03rad cap in active model. HeadingReward/State and other settings retained, including Actor1e-5 first35000steps then1e-6, Critic1e-6,140000steps. Initial actions identical in test;105tests/preflight pass. No formal training or realEVAL launched. Previous residual fullEVAL pending; new comparisons against previous/expert/fixed historicalbaseline pending. Original tutorial untouched. Design and before/after snapshots: humanoid_walk_com/humanoid_sprint/experiments/walk_bc_direct_heading_140k/design/DESIGN.md.


## 2026-09-14 Frozen BC residual heading implementation (pending training)

User approved implementation in humanoid_walk_com only. Active walk_bc_residual_heading_140k; direct previous walk_sensor_forward_only_140k. Frozen fresh24D BC plus zero-initialized25D residual, bounded±0.03rad including exploration; IMU heading-error appended, target0; Reward distance/fall minus0.1*abs(error). Existing140000-step budget/LR/noise/warmup/reset retained. 103 tests and train preflight pass. Real initial-model left/right±5deg,750step diagnostic completed without falls; neither heading converged (right2.966→3.426deg,forward2.254m; left2.502→2.682deg,forward2.557m). This validates the diagnostic, not learned correction. No formal training launched; full EVAL and comparisons pending. Design/snapshots/source: humanoid_walk_com/humanoid_sprint/experiments/walk_bc_residual_heading_140k/design/DESIGN.md. Original tutorial unchanged in this change.


## 2026-09-14 No-heading evaluation completed and reporting entry added

walk_sensor_forward_only_140k: train140000 steps completed; batch eval_batch_20260914T193411_186630Z_4de8be completed79 candidates.63 reached (79.75%),0fell,15timeout,1stall. Fastestgs250:114.52s/+1.052144557m/abs same/forward10.025192947m. Lowest deviationgs55:114.96s/+0.862659044m/forward10.024389778m. Expert115.08s/+0.855568996m; fastest0.56s quicker but0.196575561m more deviation. No successful candidate simultaneously beats expert on time and deviation. Lastgs443 did not reach (forward10.001810m,lateral+1.169455m). Direct previous walk_sensor_progress700_140k has no EVAL; cannot conclude heading removal caused change. Both140000-step budgets. Historical x4.5 baseline not directly comparable. One run per candidate, not repeated success rate. Added scripts/report_evaluation.py and show_evaluation_results.lnk; existing data only, no new EVAL. Report/CSV/JSON/plots under batch/report/.

## 2026-09-14 No-heading reward training launched

User requested distance reward control in humanoid_walk_com. Active walk_sensor_forward_only_140k; previous walk_sensor_progress700_140k completed training, EVAL pending. Existing distance/fall task replaces heading task, heading weight0, no other training changes. Same fresh BC,140000 steps,noise0.002,LR/warmup/reset settings. Run train_20260914T182140_710792Z_4f5984 launched after successful preflight. No automatic EVAL. Comparison with previous/expert/fixed historical baseline pending. Original tutorial preserved; design snapshots in new experiment.

## 2026-09-14 Sensor-based stall recovery (pending launch)

Only humanoid_walk_com changed. Active walk_sensor_progress700_140k; previous walk_sensor_com_reset700_fall5_140k. Stall now uses contact/encoder/IMU kinematic odometry, signed new forward milestones0.02m with35-sample smoothing;700 steps without new milestone resets, invalid samples keep timer running. No world X in training stall decision. Reward/goal/fall/EVAL logic retained. Old running experiment pinned to old source across reload. 94 tests/preflight pass; actual odometry validation and train/EVAL pending. Tutorial unchanged. Full design and comparison status in humanoid_walk_com/humanoid_sprint/experiments/walk_sensor_progress700_140k/design/DESIGN.md.

## 2026-09-14 Earlier reset configured (pending launch)

User approved humanoid_walk_com/walk_sensor_com_reset700_fall5_140k: stall_timeout_steps700 (28s), consecutive_falls5; previous walk_sensor_com_freshbc_v2_140k. Other parameters unchanged, old running snapshot not modified. New design snapshots saved, comparison with previous/expert/historical baseline pending. Tutorial unchanged. No training/EVAL started by this change.

## 2026-09-14 Fresh sensor24 BC completed

Only humanoid_walk_com changed. Active walk_sensor_com_freshbc_v2_140k; previous walk_sensor_com_140k (matched evaluation pending). IMU pitch enabled and mounting corrected for COM. 3159 expert samples, fresh random BC,500 epochs,lr1e-4,batch256,seed1. Independent x5: expert115.08s/+0.855569m/forward10.022276m; BC116.64s/+0.969068m/forward10.026511m. Both1/1 success,0/1fall. BC slower1.56s and larger deviation0.113499m. No DDPG started. 87 tests passed. Historical x4.5 baseline incomparable. Report: humanoid_walk_com/humanoid_sprint/experiments/walk_sensor_com_freshbc_v2_140k/design/REPORT.md.

## 2026-09-14 项目分离（以此为准）

用户要求独立项目而非功能开关。新项目 humanoid_walk_com/humanoid_sprint 固定24维：左右脚接触+估算COM，无COM/接触开关。模型与输出自包含。原 humanoid_walk_tutorials 恢复加入脚底/COM前的walk_imu_masked_height_140k，18维IMU/高度槽0，历史实验数据保留。之前记在tutorial下的脚底/COM改动已撤出活动代码。原项目80测试及预检通过；新项目验证另记README。未启动训练/EVAL，比较均待完成。

## 2026-09-14 Sensor COM State (pending training)

Active walk_sensor_com_140k; previous walk_imu_footcontact_140k (untrained). User authorized encoder/model/support COM. State 24D: old 20 + relative COM xyz + valid. No dynamic simulator truth in estimator. BC zero-column migration verified; 87 tests and train preflight passed. No training/EVAL or simulator accuracy validation. Other training settings unchanged. Comparisons vs previous/expert/fixed historical baseline pending. See humanoid_walk_tutorials/humanoid_sprint/experiments/walk_sensor_com_140k/design/DESIGN.md.


## 2026-09-14 walk_imu_footcontact_140k（待比较）

20维输入新增左右脚接触，其余条件沿用上一版walk_imu_masked_height_140k。上一版无训练/EVAL结果仍保留身份；不以更早实验替换。旧BC追加两列0权重迁移，不能把原真值State下BC成绩当本实验初始成绩。82测试/预检通过；无新性能结果，各指标及baseline比较待完成。design目录保存配置、迁移模型与完整说明。

## 2026-09-14 walk_imu_masked_height_140k（待比较）

用户授权Actor/Critic移除真值输入：IMU yaw替换rotation yaw、高度槽固定0；18维兼容旧BC，未重训BC，输入语义已变化。上一版walk_heading_reward_140k，其他预算/噪声/学习率/Reward权重保持，方向项读取的yaw随输入变化。80测试/预检通过，未训练/EVAL。各性能指标及新模式初始BC均待测；上一版、专家和历史baseline不覆盖，详细指标在新实验design/DESIGN.md。

## 2026-09-14 方向奖励版完成比较

walk_heading_reward_140k相对walk_distance_fall_bcreset_all：同140000步/seed1/噪声0.002/全程BC恢复，新增方向Reward。实际训练442对443更新轮，34对40次到达x6，均0训练跌倒；EVAL x5各79候选一次，当前46到达（58.23%）/2跌倒（2.53%）/12stall/19timeout，上一版66到达（83.54%）/1跌倒（1.27%）/7stall/5timeout。不是跨seed成功率。

当前最快gs70：110.36秒，横向净偏移+0.732232437m/绝对同值，净前进10.022713129m；上一版最快gs70：110.36秒/+0.731982837m/净前进10.022841111m，当前时间不变，偏移增加0.000249600m。专家baseline115.08秒/+0.855569136m/净前进10.022275876m、1/1成功0跌倒：当前快4.72秒、偏移少0.123336700m。历史pre_s1_20260724_v1 x4.5保留，不跨终点算差值。

低偏移候选单独比较：当前gs295=113.64秒/+0.216060443m/净前进10.025156244m；上一版gs270=115.24秒/+0.194047992m/净前进10.025617658m，当前快1.60秒但偏移多0.022012451m。当前末尾gs442=111.28秒/+0.852813498m/净前进10.026197766m。结论：出现可用的速度/偏移折中，但最快结果未改善，候选到达比例下降，不认定方向项整体有效。

读取用户已完成批次eval_batch_20260913T231928_314486Z_d24523，生成seed_001/arrival_times.png、arrival_times.csv、arrival_times_summary.json、ARRIVAL_TIMES.md；没有重新运行EVAL。

## 2026-09-14 walk_heading_reward_140k（待比较）

新增世界+x方向奖励0.1*(cos(yaw)-1)，保留距离/跌倒，其他配置与walk_distance_fall_bcreset_all相同。前后配置/Reward源码保存新实验design。上一版最佳110.36秒/+0.731982837m/净前进10.022841111m，66/79到达、1/79跌倒；专家115.08秒/+0.855569136m/净前进10.022275876m，1/1到达0跌倒。新版本未训练/EVAL，各指标待比较。历史x4.5 baseline保留不混算；Reward口径变化不直接比较总Reward。

## 2026-09-14 walk_distance_fall_bcreset_all EVAL完成

训练140000步443更新、0跌倒、40次到达训练x6、3次BC恢复。独立EVAL x5/no noise，每候选一次，79候选66到达（83.54%）、1跌倒（1.27%）、7stall、5timeout。最佳同次gs70：110.36秒，横向净偏移+0.731982837m/绝对0.731982837m，净前进10.022841111m。专家baseline115.08秒/+0.855569136m/净前进10.022275876m、1/1成功0跌倒：本轮快4.72秒、偏移少0.123586300m。初始BC114.36秒/+0.628380m：快4秒但偏移大约0.103603m，未同时优于初始BC两项指标。

末尾gs443同次111.32秒/+0.914542766m/净前进10.023061031m；第三次恢复后gs305–443共29候选全部到达，但不是最佳速度。直接上一版walk_distance_fall_140k仍缺完整EVAL，待比较，不用更早实验替代。当前相比上一版同时改变noise后半程与BC恢复截止，不能单独归因；历史pre_s1_20260724_v1 x4.5保留不与x5作差值。候选成功比例不是跨seed稳健性指标。产物在humanoid_walk_tutorials/humanoid_sprint/experiments/walk_distance_fall_bcreset_all/seed_001/，批次eval_batch_20260913T223136_366140Z_53d60e，最佳权重checkpoints/eval_candidates/ckpt_gs000070.pth。

## 2026-09-14 BC恢复版最终实验条件

walk_distance_fall_bcreset_all已启动train_20260913T221743_468838Z_6ba724。与walk_distance_fall_140k相比，BC恢复全程允许、noise全程0.002（原后半程0），两项变化，不能单独归因。未运行的只改BC恢复方案保留设计快照。无新EVAL，性能与baseline比较仍待完成，用户要求不启动EVAL。

## 2026-09-14 walk_distance_fall_bcreset_all（待比较）

仅取消BC恢复的35000步截止，设置0=整轮允许；原触发阈值、LR、Reward、noise与140000步预算不变。直接上一版walk_distance_fall_140k，尚无EVAL，不替换上一版身份；新实验未启动，性能指标待比较。前后配置及完整基准说明见新实验design/DESIGN.md。学习率是否过高仍未单独验证。

## 2026-09-13 walk_distance_fall_140k（训练启动，待比较）

用户授权Reward消融：只保留前进距离和跌倒，去动作变化惩罚。直接上一版walk_warmup50_140k_noise002，未EVAL，不替换该身份。其余预算140000步/noise两阶段/LR/预热/Buffer/seed不变。前后配置及设计在新实验design目录，训练run=train_20260913T215449_139633Z_83481c。不启动EVAL，新旧时间、横向净/绝对偏移、净前进、成功率和跌倒率均待比较；专家/历史baseline保留。Reward定义变化，不能用更高Reward或更低loss替代性能证据。

2026-09-13 最终noise条件：用户要求新实验后半程无noise，前70000步0.002 rad、后70000步0。之前提出的后半程0.001方案未运行。其他条件及上一版身份不变，仍待训练/比较。

## 2026-09-13 walk_warmup50_140k_noise002（待训练/比较）

直接上一版walk_critic_warmup50，70000步、末20轮loss均值0.032010、未EVAL；保留该身份。新方案140000步，前70000步noise0.002、后0.001，其余设置不变。预算/noise两项改变，需分别看同预算前半段和后半段loss，不能直接归因。新旧EVAL时间、偏移、前进、成功/跌倒率均待比较；专家及历史baseline保留，不以loss代替性能。详见新实验design/DESIGN.md。

## 2026-09-13 warmup50 追加频率调整（待比较）

用户将冻结期Critic间隔350→35步、冻结次数保持50；解冻后350步、总70000步、其余配置不变。活动实验及有结果的上一版身份不变；未运行的50次/350步中间配置保留design/before_frequency_change.json。正常2035步完成预热，原17500步推算失效。此轮同时调整次数和频率，Actor更新预算也受影响。72项测试/启动预检通过，无新训练或EVAL结果，各性能指标仍待比较。

## 2026-09-13 walk_critic_warmup50：待比较

正常训练 Critic-only 预热 20→50 次，其他参数不变、总预算70000步；直接上一版 walk_critic_bn_stats_sync，不以离线 Critic-only 诊断替换上一版。前后配置及基准指标见 humanoid_walk_tutorials/humanoid_sprint/experiments/walk_critic_warmup50/design/DESIGN.md。尚未训练或 EVAL，新到达时间、带符号/绝对偏移、净前进、成功率、跌倒率均待比较。延长预热会减少 Actor 更新机会，后续解释结果时需明确这一预算差异。专家 x5 baseline 和历史 pre_s1_20260724_v1 均保留。

## 2026-09-13 固定 Actor 的 Critic-only 对照

来源 walk_critic_bn_stats_sync gs40；额外 5000 次 Critic 更新，固定 Replay=2000、LR=1e-6、Actor 更新为 0，Actor Target 对齐冻结 Actor。训练 loss 降低但固定验证排序始终 8 对/11 错/1 不确定；新梯度动作实测在两个状态均出现 Q 上升而实际回报下降。本次不构成新 Actor 性能结论，无新增全程到达时间、偏移、成功率/跌倒率；原实验与直接上一版/专家/历史固定 baseline 的比较保留。条件、结果及既有性能参照详见 `.trial_logs/critic_only_control/critic_only_control/seed_001/runs/train_20260913T211558_433122Z_bb3d90/REPORT.md`，原始阶段结果为同目录 critic_only_summary.json。仅两个 BC 到达状态、配对非独立，尚需验证冻结 Actor 自身轨迹分布；没有自动改动生产训练。

## 2026-09-13 walk_continuous_reset_v2（待正式比较）

用户授权训练reset修复。以完整世界重载恢复一致物理状态，保存恢复完整训练状态与Replay/RNG，不通过普通resume覆盖LR或BC；同一步update/log完成后再重载。包含先前检查修正的首步Reward关节顺序，其他配置保持Actor1e-5/Critic1e-6、Buffer2000、70000预算、350间隔、预热20、BC截止35000、训练x6/评估x5。上一版walk_continuous_steps_v1：36模型15到达，最佳114.36秒/+0.641865m。专家baseline115.08秒/+0.855569m不变。冻结BC重置6/6轨迹一致只是环境修复验证，不是训练后性能结论。正式训练和独立评估待进行，新旧配置与验证保存在experiments/walk_continuous_reset_v2/design。

## 2026-09-13 诊断补充与结论限制

冻结BC在fresh环境x5为114.36秒/+0.62838039m/净前进10.02512884m；gs75同为114.36秒、绝对偏移多0.013485m，因此此前相对解析专家的优势不等于DDPG优于BC。反复物理reset可令零学习BC停滞，6次中4次到x6、2次停滞；该诊断终点x6、最长240秒，不能当作正式x5/120秒评估成功率。fresh对照轨迹完全相同。报告见.trial_logs/bc_reset_check/frozen_bc_reset_diagnostic/seed_001/REPORT.md。

修正连续采样reset首步Reward的关节映射实现错误（按关节名恢复Action顺序），已有结果属修正前；后续新实验必须登记此修正，尚无修正后性能数据。训练物理reset修复仍待实施，当前不能将退化单独归因于Critic或学习率。

## 2026-09-13 连续采样结果完成

walk_continuous_steps_v1批次eval_batch_20260913T033313_790361Z_100ac1已完成36模型各一次：15/36到达（41.67%）、11停滞提前结束、10超时、0跌倒。最快gs75=114.36s、横向净偏移+0.6418654248m（绝对同值）、净前进10.0261000279m，均同次评估。与同环境x5专家baseline115.08s/+0.8555691365m/10.0222758761m相比，快0.72s，绝对偏移减少0.2137037116m；专家1/1成功、0跌倒。本轮新增stall失败截断，专家未用此条件且已成功，成功轨迹数值可参考，失败统计条件须注明。仅一个训练seed，不称稳定优于专家。

gs105至最终gs198均未到达，最终gs198前进1.691867m，不能以最佳结果代表训练整体稳定。直接上一版walk_actor1e6_critic1e6_buffer2000无完整结果，继续待比较。新旧预算/Reward等变化见design，固定历史x4.5结果不直接对比。图、逐模型CSV、JSON和ARRIVAL_TIMES.md在experiments/walk_continuous_steps_v1/seed_001，失败标记分停滞与其他，统一置于上限但不作到达用时解释。

## 2026-09-13 walk_continuous_steps_v1（待比较）

用户批准连续采样与逐步Reward。实际上一版为walk_actor1e6_critic1e6_buffer2000（配置实际Actor1e-5、Critic1e-6、Buffer2000，名字不作为参数证据），其结果未知保留待比较；主baseline为专家x5=115.08s/+0.855569m/10.022276m、1/1成功0跌倒。新设计70000交互、350步更新、35000步后禁BC恢复，日志35步仅统计；停滞/跌倒恢复语义、Reward边界及采样推进均变化，不是单参数实验或等价整理。来源为Spinning Up DDPG按交互步组织Replay/更新的设计。文件位于experiments/walk_continuous_steps_v1/design/previous_config.json、after_config.json、DESIGN.md。只完成单元及隔离短运行验证；没有新到达时间、偏移、净前进、成功率或跌倒率结论。

## 2026-09-13 用户指定专家为当前baseline（x5）

用户明确“重新测专家，专家就是baseline”。当前主比较基准改为expert_sin_x5_20260913_v1；pre_s1_20260724_v1及104.24秒仅保留历史身份，终点x4.5，不直接与x5新结果算差值。新基准是解析sin/cos专家OpenLoopExpertPolicy，不加载BC或DDPG权重；使用模型同一EvalController、环境快照、5步sensor warmup、64步settle、phase、40ms控制步长，评估终点x5、120秒上限。

实测1/1到达、0/1跌倒：115.08秒，带符号横向净偏移+0.8555691364693203m（绝对值同值），净前进10.022275876147972m，2877控制步。全部来自同次评估，单次成绩不表示多seed统计。源run为walk_actor1e5_critic1e6_buffer3000/seed_001/runs/eval_20260913T030448_307902Z_b4b8fc；完整结果与环境及世界hash保存在experiments/historical_reports/expert_sin_x5_20260913_v1.json。统一配置baseline_id已更新，绘图按匹配终点/环境/世界选择专家基准，旧图旧结果保留。尚未重新评估所有DDPG模型到x5。

## 2026-09-13 Actor1e-5/Critic1e-6/Buffer3000评估作图完成

读取用户已完成的eval_batch_20260913T025459_206795Z_2e28f8，未重新运行EVAL。36个候选各一次，13/36到达（36.11%）、23超时、0跌倒；gs90起全部在120秒内未到达。最快gs50=109.08s/横向净偏移+0.5217828394m/绝对偏移0.5217828394m/净前进9.5219757916m，均同次评估。比固定baseline最快慢4.84s、绝对偏移少0.003843m（baseline7/7成功、0跌倒）；比专家慢0.36s、偏移多0.130569m。直接上一版错误0.5实验无完整评估，继续待比较；额外参考最近有效bcreset10001：本轮最快慢0.80s、偏移少0.086082m，候选成功率从36/36降至13/36。Actor学习率及Buffer同时改变，不能单独归因。当前2000rollout，与固定baseline预算不同，单训练seed不推断稳定优势。图、CSV、来源JSON及报告在experiments/walk_actor1e5_critic1e6_buffer3000/seed_001/arrival_times.*及ARRIVAL_TIMES.md。

## 2026-09-13 学习率纠正与Replay容量3000（待比较）

用户明确Actor=1e-5、Critic=1e-6，并授权选择合适Replay容量。新实验walk_actor1e5_critic1e6_buffer3000，容量3000（40ms/transition约120秒交互，扩大近期数据覆盖，沿用项目历史使用过的容量；不是已验证最优值），门槛320、batch64不变。其余2000rollout、Critic预热20、训练终点x6/评估x4.5、BC恢复截止1000、noise0不变，从BC开始、不resume。实际快照seed_001/runs/train_20260913T025006_517929Z_d96b0f/effective_config.json，前配置design/previous_config.json；已启动，未自动EVAL，性能待比较。

直接上一版walk_lr0p5_bcreset1000为助手误解小数导致的错误试验，用户已纠正；其身份及产物保留，无完整评估，不作为有效改进证据。后续报告仍标记该直接上一版待比较，并额外对照最近有效实验walk_critic_warmup_lr6_goal5p0_bcreset10001（gs50=108.28s/+0.607865m），固定baseline仍pre_s1_20260724_v1。不得静默跳过错误试验或覆盖固定baseline。

## 2026-09-13 walk_lr0p5_bcreset1000（待比较）

用户要求学习率改0.5再训练一次；Actor/Critic均从1e-6改为0.5，其余保持上一版实际设置：BC初始化、不resume、2000rollout、Buffer1000/门槛320、Critic-only预热20、训练终点x6、评估x4.5、BC恢复截止1000。沿用现有DDPG，仅进行学习率对照，不预设大步长有效。直接上一版walk_critic_warmup_lr6_goal5p0_bcreset10001，固定baseline仍pre_s1_20260724_v1。前配置保存在experiments/walk_lr0p5_bcreset1000/design/previous_config.json，实际运行配置在seed_001/runs/train_20260913T024415_347211Z_18a44b/effective_config.json。已启动训练，未授权自动EVAL，所有性能指标待独立评估后同时对比上一版及固定baseline。

## 2026-09-13 walk_critic_warmup_lr6_goal5p0_bcreset10001完成

36/36候选到达、0/36跌倒，每个模型一次；最快gs50=108.28s/横向净偏移+0.6078650563m/绝对偏移0.6078650563m/净前进9.5220247500m，均来自同次评估。相对直接上一版walk_critic_warmup_lr6_goal5p0最快：快0.48s、绝对偏移多0.095634m；相对固定pre_s1_20260724_v1最快：慢4.04s、绝对偏移多0.082239m；相对原专家快0.44s、偏移多0.216651m。上一版56/56到达、0跌倒，固定baseline7/7到达、0跌倒。

实际训练快照：2000rollout、训练终点x6.0、BC恢复截止1000，Actor/Critic均1e-6、Buffer1000/门槛320、Critic预热20；上一版3000rollout、终点x5.0、BC恢复不设截止。评估仍x4.5、单模型单回合、shared_webots_world_reload_v3。训练预算及条件不同，最快模型也早于1000rollout，不能把本次速度变化归因于截止规则。固定baseline搜索预算更小，不能据单seed判定稳定改进。

训练run：train_20260913T022136_399656Z_d0519e；评估batch：eval_batch_20260913T023038_812398Z_176c2a。产物位于humanoid_walk_tutorials/humanoid_sprint/experiments/walk_critic_warmup_lr6_goal5p0_bcreset10001/seed_001/ARRIVAL_TIMES.md、arrival_times.csv、arrival_times_summary.json、arrival_times.png/svg。保留固定baseline及历史结果。

## 2026-09-13 walk_critic_warmup_lr6_goal5p0完成

3000rollout/104384交互/298轮更新（20轮Critic-only），BC恢复3次，训练跌倒1次。training.stop_x=5.0，Evaluation仍x4.5，其他参数继承此前用户配置。56个非纯Critic候选各评估一次，56/56成功、0/56跌倒。最快时间108.76s有gs25/35/250/290并列，以偏移最小选gs290：带符号+0.512231m、绝对0.512231m、净前进9.525637m。相对固定baseline最快104.24s/+0.525626m慢4.52s、绝对偏移少0.013395m；相对原专家108.72s/+0.391214m慢0.04s、偏移多0.121017m。固定baseline只有7候选/13653历史交互，搜索预算和初态不可视为完全一致。

直接上一版walk_critic_warmup_lr6的评估批次失败，未用部分结果或更早版本替代，对上一版维持待比较。图和精确数据在humanoid_walk_tutorials/humanoid_sprint/experiments/walk_critic_warmup_lr6_goal5p0/seed_001/arrival_times.png、arrival_times.csv、arrival_times_summary.json及ARRIVAL_TIMES.md。灰线标明BC恢复，恢复后用时回落是本轮观察，不代表禁用恢复必然改善。新增1000rollout后的BC恢复限制尚未训练，不能归入本轮结论。

## 2026-09-13 BC恢复截止1000rollout（待比较）

实验walk_critic_warmup_lr6_goal5p0_bcreset1000，直接上一版walk_critic_warmup_lr6_goal5p0，固定baseline pre_s1_20260724_v1继续保留。唯一新增条件initialization.bc_reset_max_rollouts=1000：过阈值后只重置机器人，不再重置Actor及优化器，不清Replay；按累计rollout含续训判断。前后实际配置在新实验design目录。44项检查通过，未训练，不能据此声称改进。本次在途的3000rollout及用户授权的Evaluation/绘图仍属于上一版，不可标为新规则实验。

## 2026-09-13 训练终点延后至x5.0（待比较）

实验walk_critic_warmup_lr6_goal5p0，直接上一版walk_critic_warmup_lr6。只将训练终点4.7→5.0，让评估终点4.5附近及之后的状态获得继续采样；评估配置和世界文件不变。依据为扩大训练轨迹覆盖、保持评估条件固定的单变量对照设计，不代表已确认末段表现差由reset造成。

当前用户已设Actor1e-6、3000rollout、stall40，与更早已评估的walk_critic_warmup20_1不同，不能将后者替换成直接上一版。前后实际配置记录在humanoid_walk_tutorials/humanoid_sprint/experiments/walk_critic_warmup_lr6_goal5p0/design/。尚未训练/Evaluation，对直接上一版及固定baseline pre_s1_20260724_v1均待比较，后续报告同次评估的时间、带符号偏移及绝对值、净前进、成功率、跌倒率和实际预算。

## 2026-09-13 walk_critic_warmup20_1

紧邻上一轮walk_critic_warmup10；声明的对照walk_reproduce104_20260912另列，固定baseline保留。实际预热20/Buffer1000/1000rollout、34819交互、Actor78/Critic196更新，2次BC恢复、1次训练跌倒；当前3000rollout不是本轮快照。排除纯Critic阶段后16/16模型到达、0评估跌倒。最快gs40=108.84s/+0.483664m（绝对同值）/9.525812m；最直gs95=115.16s/+0.398108m/9.525537m。

对上轮最快慢0.04s、偏移少0.038953m；对固定baseline最快慢4.60s、偏移少0.041962m；对原专家慢0.12s、偏移多0.092450m。上轮最直115.40s/+0.400393m，本轮最直小幅改善0.24s/0.002285m，但单seed且预热与Buffer同时变化，不宣称因果改善或多seed稳定性。固定baseline搜索预算不同。详细数据和来源在humanoid_walk_tutorials/humanoid_sprint/experiments/walk_critic_warmup20_1/seed_001/EVAL_REPORT.md及evaluation_comparison.json/csv。

## 2026-09-13 Critic预热实际训练与评估结果

最新walk_critic_warmup10，紧邻上一轮walk_critic_warmup20_20260913；两者实际训练快照均预热10轮、Buffer5000，名称中的20仅为旧名。配置声明的对照walk_reproduce104_20260912也保留比较。最新34825交互/98更新轮次（Actor88/Critic196步），前轮34794/99，无预热34832/98；均配置1000rollout、seed1。与无预热版本同时改变了Buffer容量，不能视作仅预热的单变量实验。

两轮各20个保存Checkpoint评估完成：最新20/20成功、前轮19/20成功，均0评估跌倒；最新训练跌倒1次。最快gs5/10=108.76s/+0.505413m，权重与BC完全相同。实际训练后的最快两轮均gs15=108.80s/+0.522616m（绝对0.522616m）/净前进9.521970m。最新最小偏移成功gs85=115.40s/+0.400393m/9.522453m，前轮gs90=116.76s/+0.393473m/9.527715m，存在速度与偏移取舍。

当前训练后最快对无预热最快gs40：慢0.60s、绝对偏移减少0.112311m；对固定baseline最快gs15：慢4.56s、绝对偏移减少0.003009m；对原专家：慢0.08s、绝对偏移增加0.131402m。全部候选无同时优于专家时间与偏移的模型。固定baseline候选7个、历史13653交互，与本次搜索预算不等；不声称精确历史复现或多seed提升。

完整证据：humanoid_walk_tutorials/humanoid_sprint/experiments/walk_critic_warmup10/seed_001/EVAL_REPORT.md、evaluation_comparison.csv/json/png；warmup_verification.json验证实际预热及BC恢复。两批manifest为005354_904519（最新）与005422_358d61（前轮），原文件保留。先前预热20/Buffer3000的design文件是实施时提案，实际训练以对应run快照为准。

## 2026-09-13 Critic-only 预热实验（待比较）

实验 walk_critic_warmup20_20260913，直接上一版 walk_reproduce104_20260912，固定 baseline pre_s1_20260724_v1 保留。配置及BC hash在 humanoid_walk_tutorials/humanoid_sprint/experiments/walk_critic_warmup20_20260913/design/。

唯一训练变化：Buffer320达标后前20个既定更新轮次仅更新Critic/critic_target（每轮2步），Actor/actor_target/Actor optimizer不变；第21轮恢复联合训练。每10rollout更新、1000rollout预算、Actor1e-5/Critic1e-6、noise0、原BC及环境保持。预热计入总更新轮次：相同交互预算下实验组少20次Actor更新，Critic更新不额外增加，需分别累计optimizer实际步数。

检验早期Critic误差影响BC Actor的假设，参考TD3价值误差及延迟策略更新思想：https://arxiv.org/abs/1802.09477；不是TD3替换。40项检查和train --check通过，尚未正式训练或Evaluation。对上一版和固定baseline均待比较；后续使用成对seed及相同交互预算，按同一Checkpoint同次评估报告时间、带符号偏移及绝对值、净前进、成功率和跌倒率。

## 固定改进前 baseline

2026-09-11 用户指定：以后每次修改训练设计，都与直接上一版以及本固定 baseline 比较。

- ID：`pre_s1_20260724_v1`，S1 修改前的 2026-07-24 独立 Evaluation。
- 冻结数据：`git_sin_training/baselines/pre_s1_20260724_v1/pre_s1_20260724.csv`。
- CSV SHA256：`463CF1723419B789E84209EFE3D326DBF16D20E7583A45F11D03CB5C18D6F482`。
- 来源说明：同目录 `pre_s1_20260724.md`，包含原始 CSV 与本地历史会话路径。
- 范围：gs5、10、15、20、25、30、35 共七个 Checkpoint；seed=1，每个评估一回合，均成功且无跌倒。不是同一模型七次重复实验，也不是多 seed 统计。
- 条件：终点世界坐标 x=4.5 m，phase_mode=continuous_time，phase_frequency_hz=4，phase_reset_value=0。其他条件需查历史记录，未确认项不得声称一致。
- 后续 S4 的 118.96 秒记录不属于本 baseline。冻结数据不追加新结果；若纠错，另建带来源的新修订并保留旧记录。

| Checkpoint update | 到达时间 (s) | 净前进 (m) | 横向净偏移 (m) |
|---|---:|---:|---:|
| 5 | 107.36 | 9.5229 | +0.4236 |
| 10 | 115.84 | 9.5334 | +0.1146 |
| 15 | 104.24 | 9.5159 | +0.5256 |
| 20 | 116.64 | 9.5336 | +0.4067 |
| 25 | 115.12 | 9.5190 | +0.2099 |
| 30 | 114.36 | 9.5232 | +0.1168 |
| 35 | 106.32 | 9.5244 | +0.4033 |

保留整批作为参照。最快模型为 gs15：104.24 s / 0.5256 m；最小绝对净偏移模型为 gs10：115.84 s / 0.1146 m。不能把 104.24 s 与 0.1146 m 拼成一个模型的成绩。

## 每次修改的记录与比较流程

1. 修改前登记版本 ID、直接上一版 ID、修改内容、实际训练配置及独立实验目录；保存配置快照，记录初始化/恢复 Checkpoint 来源。有文件时记录其 hash；不可获取的历史文件明确标记未知。
2. 使用训练所得 Checkpoint 做独立 Evaluation，保留原始结果和评估配置。记录训练环境交互步数、update 数、seed、评估次数、世界/初始姿态、终点、超时限制及 phase 设置。BC 单独加载的成绩另列，不代替 DDPG 训练结果。
3. 比较表同时列出当前版、直接上一版和固定 baseline，并给出两组差值。以原始精度计算：时间差 = 当前时间 − 参照时间；偏移差 = 当前 |横向净偏移| − 参照 |横向净偏移|。负值表示对应指标降低。横向净偏移是结束 Y − 开始 Y，不是轨迹最大偏移。
4. 尽量在相同环境交互预算、相同评估条件下比较。匹配 baseline 的 Checkpoint 时保留其 update 标识；如果改变更新频率，同 update 数不代表同训练预算。历史交互预算无法确认时，标为“历史参考，训练预算未对齐”。终点或其他条件不一致时标记不可直接比较，不能仅用耗时判断进退。
5. 完整列出已评估 Checkpoint，事先记录模型选择规则，避免只挑有利结果。对整批 baseline 可以逐 Checkpoint 对照，或分别对照最快和最小偏移的两组完整成绩，不默认合并成单个分数。
6. 同时报告成功率、跌倒率、净前进。失败或超时不能算成更快到达；更快但偏移更大应写明取舍。单回合结果仅能描述本次表现，不能宣称统计显著提升。
7. 训练或评估未完成时写“待比较”，记录缺失项。上一版结果缺失时保持其身份；可额外提供最近已评估版本，但不能替代直接上一版。文档登记不代表自动启动长程训练；后续执行遵循用户授权。

## 版本记录

后续每次训练设计修改追加一行，完成评估后补齐结果文件与两组比较结论。

| 版本 ID | 直接上一版 | 修改/来源 | 实验及结果 | 对上一版 | 对固定 baseline |
|---|---|---|---|---|---|
| pre_s1_20260724_v1 | 不适用 | S1 前历史评估，2026-09-11 固定 | historical_reports/pre_s1_20260724.csv（完整路径见上文） | 不适用 | 固定参照 |
| train_aligned104_v1 | train_goal_reset_v2（上版200rollout/19update，未评估） | Actor1e-5、Critic1e-6、warmup320；实际1000rollout/34680条Transition/100update、3次goal reset、8训练跌倒；保留新sin BC/phase/摆臂 | git_sin_training/outputs/train_aligned104_v1/REPORT.md及comparison.csv；20不同模型12到达/8超时/0评估跌倒；最快gs5=110.32s/+0.698209m，复测一致 | 待比较：直接上一版暂无Evaluation，不替代为更早版 | 最快比104.24s慢6.08s/多偏0.172584m；同update差值见CSV；新BC/源码/初始状态和实际预算不同 |
| git_sin_goal_reset_v1 | git_sin_bc_20260912（git_sin_training/outputs/train） | 用户授权到终点先记录末步、物理重置继续学习，保留Replay/网络/优化器/更新节奏；原配置200rollout；初始网络逐项与上版一致 | git_sin_training/outputs/train_goal_reset_v1；200rollout/6974条Transition/19update/1goal reset/0训练跌倒；5检查通过 | 待比较：用户明确要求不运行EVAL | 待比较：未独立评估，训练完成不代表性能改进 |
| git_sin_bc_20260912 | walk_reproduce104_20260912 | 用户授权Git6a11732源码、新sin专家采集2781样本、新BC500epochs、统一连续phase和摆臂；Git Actor/Critic1e-6、warmup640；实际79rollout/2738记录Transition/6update，到终点停止 | git_sin_training/RESULTS.md及results.csv；6/6模型成功、0跌倒；最快gs1=109.56s/+0.894002m，未超过新专家108.84s/+0.819705m | 最快慢1.36s/多偏0.259074m；代码/参数/预算/手臂条件不同，历史参考 | 对最快慢5.32s/多偏0.368376m；训练预算不同；不是104秒完整历史复原 |
| run_20260912T032929_073219Z_132c94 | walk_lr1e5_update10 / train_20260912T031324_580491Z_aa7141；关系纠正见comparison_parent.json | 尽量恢复104s条件：noise0、stall20、1000段，Actor1e-5/Critic1e-6、warmup320；34832交互/98update/3次BC恢复；旧BC与Critic初始状态无法复原 | walk_reproduce104_20260912；reproduction104_20260912.md及.csv/.png；20/20成功、0跌倒；gs40=108.20s/+0.634928m | 待比较：直接上一版尚无独立Evaluation | 最快慢3.96s/多偏0.109302m；共同前7平均快0.211429s/多偏0.222128m；完整训练预算不同 |
| run_20260911T235242_bd0966（完成planned_gaussian_001） | run_20260911T231825_3b68d7 | Gaussian0.01rad、3000段/104859交互/gs295、8次BC重置 | comparison_20260912_gaussian.md及.csv；46/59到达、0跌倒 | 上版49/59；最终慢2.32s/多偏0.007167m；前7慢0.1714s/少偏0.038009m | 前7快1.20s/多偏0.200676m；最快仍慢4.52s |
| planned_gaussian_001 | run_20260911T231825_3b68d7 / walk_lr1e5_buffer320 | 训练Gaussian Action noise std0.01rad；其余Actor1e-5/Critic1e-6、warmup320、3000段保持；实际裁剪动作进入Replay及Reward | walk_lr1e5_buffer320_gaussian；36测试及预检通过，尚未训练 | 待比较 | 待比较 |
| run_20260911T231825_3b68d7（完成planned_lr1e5_buffer320） | run_20260911T224647_07f374，配置显式指定 | Actor1e-5、warmup320、3000段/104519交互/gs294、8次BC重置 | comparison_20260912_lr1e5_buffer320.md及.csv；49/59成功、0跌倒 | 最新慢2.92s/偏移少0.039542m；前7平均慢1.2171s/偏移多0.040318m | 前7平均快1.3714s/偏移多0.238685m；最快仍慢3.84s |
| planned_lr1e5_buffer320 | run_20260911T224647_07f374 / walk_newtest_earlyupdate_3 | 用户授权Actor1e-5、warmup320；其余保留Critic1e-6、3000段；新实验BC启动，resume=never | walk_lr1e5_buffer320；配置及模型预检通过，尚未训练 | 待比较 | 待比较 |
| run_20260911T224647_07f374 | run_20260911T212845_ac077d（会话顺序，配置未填） | 3000段/104616交互/gs299，warmup256、Actor/Critic1e-6，两次BC重置 | walk_newtest_earlyupdate_3；comparison_20260912_3000rollouts.md及.csv；60/60成功、0跌倒 | 最终快1.44s/偏移多0.027685m；前7平均慢0.04s/偏移多0.000738m | 前7平均快2.5886s/偏移多0.198367m；最快仍慢4.44s |
| run_20260911T212845_ac077d | 延长预算对照为run_20260911T204933_881ec3；previous_experiment为空，训练时间紧邻le5另列参考 | Actor/Critic1e-6、warmup512、2000段/69620交互/gs199 | walk_newtest_earlyupdate_2；comparison_20260912_2000rollouts.md及.csv；40/40成功、0跌倒 | 同gs5–95与1000段实验一致；最终慢1s/偏移少0.093047m；对le5另见报告 | 前7平均快2.6286s/偏移多0.19763m；最快仍慢4.52s |
| run_20260911T204933_881ec3 | 按已知训练时间为run_20260911T200559_1b2caa；配置未填；额外对比上次查看的le5 | Actor/Critic1e-6、warmup512、seed修复、1000段/34797交互/99updates | walk_newtest_earlyupdate；comparison_20260911_lr1e6.md及.csv；20/20成功、0跌倒、无训练BC重置 | 对时间上版同gs55快4.96s/偏移少0.02503m；对le5前7平均快1.2743s/偏移少0.03416m | 前7均成功，平均快2.6286s、偏移多0.19763m；最快仍慢4.52s |
| run_20260911T205339_75b17c | run_20260911T200559_1b2caa（会话顺序；配置previous_experiment为空） | seed修复后新训练；实际Actor1e-5/Critic1e-6、warmup512、1000rollouts/34911交互/97updates | walk_newtest_earlyupdate_le5；historical_reports/comparison_20260911_le5.md及.csv；20模型17成功、0跌倒 | 同gs5–35为7/7对6/7；最新快4.88s/偏移少0.01343m但预算不同；同gs55慢2.76s/偏移少0.07731m | 同gs5–35均7/7；平均快1.3543s、偏移多0.23179m；最快慢4.68s |
| seed_before_network_init_v1 | 修复前工作区；最近已评估训练 run_20260911T200559_1b2caa，后续用户参数编辑尚无新训练结果 | 初始化前设置seed；逐模型独立进程评估。保留用户当前Actor/Critic lr=1e-6、warmup512、1000rollouts | 待新训练；尚未指定新实验目录。启动时会保存实际快照。旧gs50/55复评验证在 .trial_logs/reproducibility_fix | 待比较；旧模型复评不能代表seed修复后的训练 | 待比较；固定baseline不变 |
| run_20260911T200559_1b2caa | 同目录上一轮归档日志 metrics_20260911-220605.csv，最后保留gs495；未填写previous_experiment | warmup256，update间隔10；本轮停止于日志update56，最新保存gs55 | historical_reports/diagnosis_20260911_fresh_eval.md；11个模型逐个新进程复测，取代原批量比较结论 | 最新gs55为113.84s/0.5430m，对上一轮gs495慢4.16s、偏移减少0.0492m；预算不同 | 10/11成功，同gs5–35为6/7对baseline7/7；最快gs5为109.84s/0.6078m，对baseline最快慢5.60s/偏移增加0.0821m |

2026-09-11 已完成比较及独立进程复测纠正，以 `humanoid_walk_tutorials/humanoid_sprint/experiments/historical_reports/diagnosis_20260911_fresh_eval.md` 为最新结论。原批量比较保留供追溯，固定 baseline 数据保持不变。


## 2026-09-12 评估协议回归验证

采用 shared_webots_world_reload_v3，固定既有模型验证重复一致性，不创建新训练版本。新项目 gs30 与原项目 gs40 各四次一致，与各自旧独立进程结果相同。节点回位协议漂移产物保留但不与稳定评估混合统计；无新训练改进结论，不覆盖固定 baseline 或上一版记录。详细条件、成功/跌倒、时间/偏移/净前进及证据见 EVAL_REPRODUCIBILITY.md。
# 2026-09-13 reset修复版：1000 rollout等效预算（已完成）

EVAL批次eval_batch_20260913T040236_443774Z_b0cfcb，16候选各一次：9/16成功（56.25%）、0/16跌倒、7超时、0stall。本轮最佳gs80到x5=114.64秒、横向净偏移+0.567631896m/绝对偏移0.567631896m、净前进10.023167454m。专家baseline=115.08秒/+0.855569136m/净前进10.022275876m，1/1成功、0跌倒；当前快0.44秒、偏移少0.287937241m。直接上一版最佳gs75=114.36秒/+0.641865425m/净前进10.026100028m，15/36成功（41.67%）、0跌倒；当前慢0.28秒、偏移少0.074233529m。上一版70000步、本轮35000步，候选数不同，仅单seed各模型单次，不宣称整体稳定提升。历史104.24秒baseline为x4.5，不作差值；未训练BC对照114.36秒/+0.628380m，当前未超越其速度。99更新轮次，26903步发生一次BC恢复；后续恢复后的最佳仍不能视作持续DDPG提升。完整报告及曲线：experiments/walk_continuous_reset_v2/seed_001/ARRIVAL_TIMES.md（相对humanoid_sprint）。

当前walk_continuous_reset_v2，run=train_20260913T035637_052194Z_9802bc，35000实际交互步；直接上一版walk_continuous_steps_v1为70000步。Actor1e-5/Critic1e-6、Replay2000/门槛320、350步更新、Critic预热20、BC恢复截止35000步。完整配置与产物在humanoid_walk_tutorials/humanoid_sprint/experiments/walk_continuous_reset_v2。用户授权训练后EVAL全部有效候选，x5专家115.08秒为当前baseline；历史104.24秒终点不同不直接比较。结果待比较。
# 2026-09-13 两阶段学习率：待比较

启动前追加用户要求：训练stall_timeout_steps由1400改为2800（40→80等效轮），检测窗口35步和0.02m阈值保持原值，EVAL超过40轮的阈值保持原值。修改前配置保留在新实验design/before_stall_extension.json。本轮新增停滞容忍变化，性能依旧待比较，不把未训练配置当作实验结果。

新实验walk_lr_two_stage_2000，上一版walk_continuous_reset_v2，不覆盖已有结果。新预算70000步（2000×35），Actor第1～35000步1e-5、35001～70000步1e-6，Critic全程1e-6；其他条件延续上一版，BC恢复仅前35000步。参考分段学习率衰减思想降低后期更新幅度；增加预算与调度同时改变，结果不能单独归因于其中一项。新实验design/previous_config.json及after_config.json记录实际设置。上一版最佳114.64秒/+0.567631896m/净前进10.023167454m，9/16成功、0跌倒；当前x5专家baseline115.08秒/+0.855569136m/净前进10.022275876m，1/1成功、0跌倒。新训练与EVAL尚未执行，到达时间、偏移、净前进、成功率、跌倒率均待比较；历史x4.5固定baseline保留但不同终点不作直接差值。
# 2026-09-13 两阶段无噪声完成，追加noise001

walk_lr_two_stage_2000：70000步、199更新；EVAL36候选23成功（63.89%）、13超时、0跌倒/停滞提前结束。最佳gs25=114.68秒、横向净偏移+0.630222312m（绝对相同）、净前进10.022669018m。对直接上一版walk_continuous_reset_v2（35000步）慢0.04秒、偏移多0.062590416m；对x5专家快0.40秒、偏移少0.225346825m。预算不同、单seed各模型一次；x4.5历史baseline不可直接比较。未同时优于初始BC114.36秒/+0.628380m。

按用户已批准条件，下一版walk_lr_two_stage_noise001加入训练高斯Action noise std=0.001rad，其他训练参数与本轮一致，从同一BC初始化；EVAL无noise。新旧配置存新实验design/previous_config.json和after_config.json。性能待比较；直接上一版就是walk_lr_two_stage_2000，不替换成更早最佳。
# 2026-09-13 noise001完成：未同时超越初始BC

当前walk_lr_two_stage_noise001（70000步、199更新），直接上一版walk_lr_two_stage_2000（同预算同学习率/Buffer/停滞设置，训练noise为唯一配置差异，另有实验身份）。本轮EVAL36候选9成功（25%）/27超时/0跌倒/0stall；上一版23/36成功（63.89%）/13超时/0跌倒。最佳当前gs75：115.12秒，横向净偏移+0.537047749m、绝对0.537047749m，净前进10.025229823m；上一版gs25：114.68秒/+0.630222312m/净前进10.022669018m。当前慢0.44秒、偏移少0.093174563m。专家baseline115.08秒/+0.855569136m/净前进10.022275876m、1/1成功0跌倒，当前慢0.04秒、偏移少0.318521388m。初始BC114.36秒/+0.628380m，当前慢0.76秒，未同时超过两指标；不再追加实验。x4.5固定历史baseline依旧保留，不跨终点比较。各模型仅测一次、单训练seed，候选到达比例不是跨seed成功率；不能断言所有Action noise都无效。时间和偏移均取同一最佳模型同次评估。

噪声版run=train_20260913T043054_366569Z_07bc90，EVAL=eval_batch_20260913T043659_891209Z_b05fe0；结果位于humanoid_walk_tutorials/humanoid_sprint/experiments/walk_lr_two_stage_noise001/seed_001/，ARRIVAL_TIMES.md及arrival_times_summary.json记录精确数据，noise_comparison.png为两轮曲线。
# 2026-09-13 Critic BatchNorm修正：待比较

当前walk_critic_bn_modes，直接上一版walk_lr_two_stage_noise001，唯一训练设计变更manage_critic_batchnorm=True：明确Target/Actor阶段冻结BN统计，在线Critic只在优化阶段更新统计量。70000步、Actor两阶段1e-5→1e-6、Critic1e-6、noise0.001、Buffer2000及stall2800不变。设计快照见新实验design目录。直接上一版最佳gs75=115.12秒、横向+0.537047749m/绝对同值、净前进10.025229823m，9/36到达、0跌倒；专家x5=115.08秒/+0.855569136m/净前进10.022275876m，1/1到达0跌倒。当前训练与独立EVAL尚未进行，上述指标均待比较；历史x4.5固定baseline保留不跨终点作差值。不能以代码验证通过宣称训练改善。
# 2026-09-13 Critic BN修正版完成：最佳109.04秒

当前walk_critic_bn_modes，与直接上一版walk_lr_two_stage_noise001同为70000步/199更新、seed1、noise0.001、同环境与两阶段LR，仅BN模式管理不同。最佳gs40=109.04秒，带符号横向净偏移+0.608753675m、绝对0.608753675m、净前进10.021873348m，来自同一模型同次x5评估。上一版gs75=115.12秒/+0.537047749m/净前进10.025229823m：当前快6.08秒、偏移多0.071705927m。当前36候选8到达（22.22%）、9跌倒（25%）、15stall、4超时；上一版9/36到达（25%）、0跌倒、27超时。候选比例不是跨seed成功率。

专家x5 baseline=115.08秒/+0.855569136m/净前进10.022275876m，1/1到达0跌倒：当前快6.04秒、偏移少0.246815461m。初始BC对照=114.36秒/+0.628380m：当前快5.32秒、偏移少约0.019626m，满足用户最佳模型同时超过两项指标的要求；仍为单seed单次评估，不能视作稳定训练结论。历史pre_s1_20260724_v1的104.24秒终点x4.5，不能与本轮x5直接计算优劣。gs90及以后全部未到达，后期退化问题仍在。

产物：humanoid_walk_tutorials/humanoid_sprint/experiments/walk_critic_bn_modes/seed_001/；run=train_20260913T190939_907875Z_7dbb62，eval_batch_20260913T191645_553460Z_8c905d；最佳checkpoints/eval_candidates/ckpt_gs000040.pth，SHA256=929126a33902cb5a5ea755d8e9324e367cfc500f500b6d39439d4d2a387fd3f6。完整报告ARRIVAL_TIMES.md及arrival_times_summary.json，曲线arrival_times.png。训练5次跌倒，EVAL跌倒另外统计。
# 2026-09-13 Target BN统计量同步：待比较

新实验walk_critic_bn_stats_sync，直接上一版walk_critic_bn_modes（最佳gs40=109.04秒、横向净偏移+0.608753675m/绝对同值、净前进10.021873348m，8/36到达、9/36跌倒）。唯一训练变更为BN运行统计量在Target更新时直接复制，网络参数仍tau=1e-4；70000步、两阶段LR、noise0.001、Buffer等不变。参考Stable-Baselines3 TD3实现区分权重软更新与running stats直接复制，意在避免Target统计量长期滞后；不能保证性能改善。配置前后快照保留新实验design目录。训练/EVAL未进行，到达时间、带符号/绝对偏移、净前进、成功率、跌倒率均待比较。后续同时比较直接上一版与expert_sin_x5_20260913_v1（115.08秒/+0.855569136m/净前进10.022275876m，1/1到达0跌倒）；历史x4.5固定baseline保留，不跨终点直接比较。
# 2026-09-13 BN统计量同步版EVAL完成

完整候选统计：当前12/36到达（33.33%）、9/36跌倒（25%）、7stall、8超时；直接上一版8/36到达（22.22%）、9/36跌倒（25%）、15stall、4超时。冻结Critic动作诊断不是新训练设计/新baseline；只用于诊断局部预测排序，详细报告在.trial_logs/critic_action_diagnostic目录，20个仿真分支无参数更新。

walk_critic_bn_stats_sync完成70000步/199更新，训练跌倒28次。EVAL批次eval_batch_20260913T210307_377101Z_f472e3完整36候选；最快gs40=109.00秒，横向净偏移+0.601792946m/绝对同值，净前进10.022778553m。对直接上一版walk_critic_bn_modes最佳109.04秒/+0.608753675m，快0.04秒、偏移少0.006960729m；对x5专家115.08秒/+0.855569136m，快6.08秒、偏移少0.253776190m；对初始BC114.36秒/+0.628380m，快5.36秒、偏移少约0.026587m。前后训练预算和其他参数匹配，单seed各候选一次；0.04秒仅一个控制步，不据此宣称稳健提升。历史104.24秒为x4.5，仍不跨终点比较。用户已授权冻结Actor验证Critic，诊断正在独立目录执行，不是新训练。

## 2026-09-15 Absolute world-heading diagnostic (training launched)

User approved accurate absolute heading error for both Actor and Critic. Independent project humanoid_walk_absolute_heading/humanoid_sprint; source humanoid_walk_com and tutorial unchanged. Experiment walk_bc_absolute_heading_140k; direct previous walk_bc_direct_heading_140k remains pending full train/EVAL, not replaced by residual experiment. World torso yaw from atan2(R[3],R[0]); input25=wrap(target-worldyaw), target0; original24 BC features unchanged. Heading reward uses same absolute error, weight0.1. Observation contract v5; initial BC input25 column verified zero, weights unchanged, metadata migrated explicitly. Same140000steps/LR/noise0.002/warmup50/reset/initialyaw. Three pipeline tests and preflight pass. Training launched run train_20260914T232239_410372Z_58fd7c (running). No learned correction conclusion or EVAL yet. Compare all candidates against direct previous, expert_sin_x5_20260913_v1 and fixed pre_s1_20260724_v1 with condition caveats; report time and signed/absolute lateral drift from same evaluation, forward progress and success/fall rates. No automatic EVAL started. README and start_training.cmd/start_evaluation.cmd in new project.

## 2026-09-15 Five-rollout update interval (pending launch)

User confirmed five nominal 35-step rollouts: update_every_steps 350 -> 175 in humanoid_walk_absolute_heading. New experiment walk_bc_absolute_heading_update175_140k; direct previous walk_bc_absolute_heading_140k. Existing run retains its immutable350-step snapshot. Critic-only warmup50, interval35, buffer threshold320, BC weights, LR schedule, noise and140000-step budget unchanged. This is an update-to-data ratio control: twice the joint update frequency, not evidence of better learning. Config/world/model preflight passed; new run not launched. Before/after configs saved in experiments/walk_bc_absolute_heading_update175_140k/design. Comparisons against direct previous, expert_sin_x5_20260913_v1 and fixed pre_s1_20260724_v1 remain pending independent Evaluation (same-checkpoint time, signed/absolute lateral displacement, forward progress, success/fall rates and condition caveats).
