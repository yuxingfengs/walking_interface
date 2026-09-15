# Published experiment: gs330

Training and full Evaluation completed. This release contains only replay_retained_half_noise_actor_decay. See the repository README for current status, results and portable evaluation. Earlier prepared/not-started descriptions below are historical. BC is now assets/actor_bc_heading27.pth.

# replay_retained_half_noise_actor_decay

Independent project: see ../PROJECT.md and ../STATE_MACHINE.md.

Run start_training.cmd for training followed by a quick report; run start_evaluation.cmd separately for Evaluation and a full report.

Prepared, not started. BC recovery retains Replay; noise .002 then 0; Actor LR 1e-5 then 1e-6 at 100000/200000 steps; Critic LR 1e-6 with no normalization.
