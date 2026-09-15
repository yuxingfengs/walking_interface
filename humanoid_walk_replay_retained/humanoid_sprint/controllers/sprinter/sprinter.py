from controller import Supervisor

import torch
import numpy as np
import random
import time
from pathlib import Path
import sys
import os

CONTROLLER_DIR = Path(__file__).resolve().parent
CONTROLLERS_DIR = CONTROLLER_DIR.parent
COMMON_DIR = CONTROLLER_DIR.parent / "common"
for directory in (str(COMMON_DIR), str(CONTROLLERS_DIR)):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from experiment_paths import ExperimentPaths
from exploration import GaussianActionNoise
from recovery_policy import within_bc_reset_window
from action_data import load_initial_actions_csv
from algorithms import load_algorithm
from envs.nao_locomotion_env import NaoLocomotionEnv
from webots_platform.webots_runtime import WebotsRuntime
from tasks import load_task
from contracts import (
    ACTION_DIM,
)
from run_config import (
    RUN_CONFIG,
    ACTOR_LR,
    BC_CHECKPOINT_FILENAME,
    CONSECUTIVE_FALL_CYCLE_LIMIT,
    EVAL_CHECKPOINT_EVERY_UPDATES,
    EXPERIMENT_NAME,
    MIN_BUFFER_FOR_TRAIN,
    STALL_CYCLE_LIMIT,
    STALL_MAX_ABS_X_PROGRESS_M,
    STOP_X,
    USE_BC_INITIALIZATION,
    SEED,
    TRAIN_EPISODES,
    TRAIN_EVERY_EPISODES,
    TASK_MODULE,
    TASK_CLASS,
    ALGORITHM_MODULE,
    ALGORITHM_CLASS,
    task_config,
    algorithm_config,
)
from configs.runtime import require_launch, runtime_settings, SNAPSHOT_ENV

from helperfunction import (
    _prepare_logging,
    _save_meta,
    rotate_metrics_csv,
    log_session_event,
    log_episode,
    log_update,
    log_checkpoint_event,
)
class Config:
    ACTOR_LR = ACTOR_LR

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


cfg = Config()

INITIAL_ACTIONS = load_initial_actions_csv(
    str(CONTROLLER_DIR / "initial_actions.csv"),
    expected_dim=ACTION_DIM, # 动作维度
    to_radians=False,        # 如果是度就设 True
    device="cpu",
    dtype=torch.float32
)

class Sprinter(Supervisor):
    def __init__(self, cfg=cfg, seed=SEED):
        self.seed = seed
        self.set_global_seed(seed)
        super().__init__()

        self.timeStep = RUN_CONFIG["webots"]["control_time_step_ms"]
        self.device = cfg.DEVICE
        self.runtime = WebotsRuntime(self, **runtime_settings(RUN_CONFIG, mode='train'))
        self.task = load_task(TASK_MODULE, TASK_CLASS, task_config())
        self.env = NaoLocomotionEnv(self.runtime, self.task)
        self.algorithm = load_algorithm(
            ALGORITHM_MODULE,
            ALGORITHM_CLASS,
            algorithm_config(device=self.device, task_id=self.task.task_id),
        )
    
    def set_global_seed(self,seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def initialize(self):
        self.env.initialize()



    def get_translation(self):
        return self.env.position

    def get_center_of_mass(self):
        return self.env.center_of_mass_height
    def hard_reset(self, settle_steps: int = RUN_CONFIG["environment"]["settle_steps"]):
        """Compatibility delegate; new code should call env.reset()."""
        self.env.reset(settle_steps=settle_steps)
        self.algorithm.set_evaluation_mode()

    
    def train_model(self, num_episodes: int,
                    *,
                    log_dir: str = None,
                    min_buffer_for_train: int,
                    train_every_episodes: int,
                    stop_x: float,
                    consecutive_fall_cycle_limit: int,
                    stall_cycle_limit: int,
                    stall_max_abs_x_progress_m: float,
                    bc_checkpoint_path: str,
                    use_bc_initialization: bool = False,
                    resume_checkpoint_path: str = None,
                    resume_metadata: dict = None,
                    seed: int = 1,
                    eval_checkpoint_every_updates: int = 5,
                    stop_ckpt_path: str = None,
                    experiment_name: str = EXPERIMENT_NAME):

        if RUN_CONFIG['training']['sampling_mode'] == 'continuous':
            from continuous_training import train_continuous
            return train_continuous(self, RUN_CONFIG, resume_checkpoint_path=resume_checkpoint_path,
                                    resume_metadata=resume_metadata,
                                    use_bc_initialization=use_bc_initialization)

        if seed != self.seed:
            raise ValueError("Training seed must match the seed used to initialize networks")
        exploration = GaussianActionNoise(RUN_CONFIG["training"]["action_noise_std_rad"], seed)
        if resume_checkpoint_path is not None:
            exploration.load_state_dict((resume_metadata or {}).get("exploration"))
        print(f"[exploration] Gaussian std={exploration.std_rad} rad; seed={exploration.seed}")
        self.resume_checkpoint_path = resume_checkpoint_path
        experiment_paths = ExperimentPaths(experiment_name, seed)
        experiment_paths.ensure_directories()
        log_dir = str(Path(log_dir)) if log_dir is not None else str(experiment_paths.train_log_dir)
        stop_ckpt_path = str(Path(stop_ckpt_path)) if stop_ckpt_path is not None else str(experiment_paths.resume_checkpoint)

        is_resuming = resume_checkpoint_path is not None
        if not is_resuming:
            rotate_metrics_csv(log_dir)
            self.global_step = 0
            self.ep_offset = 0
            self.best_reward = float("-inf")

        _prepare_logging(self, log_dir=log_dir, resume=is_resuming)
        if resume_metadata:
            for field in ("global_step", "ep_offset", "best_reward"):
                if field in resume_metadata:
                    setattr(self, field, resume_metadata[field])

        batch_size = self.algorithm.batch_size

        log_session_event(
            self,
            event_name="session_start",
            event_value=True,
            seed=seed,
            num_episodes=int(num_episodes),
            train_every_episodes=int(train_every_episodes),
            min_buffer_for_train=int(min_buffer_for_train),
            use_bc_initialization=bool(use_bc_initialization),
            resume_checkpoint_path=resume_checkpoint_path,
            task_id=self.task.task_id,
            algorithm_id=self.algorithm.algorithm_id,
            sequence_length=int(self.task.config.sequence_length),
            phase_mode=self.task.config.phase_mode,
            phase_frequency_hz=float(self.task.config.phase_frequency_hz),
            phase_reset_value=float(self.task.config.phase_reset_value),
            stop_x=float(stop_x),
            consecutive_fall_cycle_limit=int(consecutive_fall_cycle_limit),
            stall_cycle_limit=int(stall_cycle_limit),
            stall_max_abs_x_progress_m=float(stall_max_abs_x_progress_m),
            bc_checkpoint_path=str(bc_checkpoint_path),
            note="train_model call started",
        )

        if use_bc_initialization:
            log_session_event(
                self,
                event_name="use_bc_initialization",
                event_value=True,
                seed=seed,
                use_bc_initialization=True,
                note="Actor and target Actor were initialized from the Behavior Cloning checkpoint before training",
            )

        last_eval_checkpoint_step = None

        def save_eval_candidate(*, event_name: str, episode: int, note: str) -> bool:
            nonlocal last_eval_checkpoint_step
            if self.algorithm.actor_update_count == 0:
                return False
            try:
                global_step = int(getattr(self, "global_step", 0))
                eval_checkpoint_path = experiment_paths.eval_checkpoint(global_step)
                self.algorithm.save_checkpoint(
                    str(eval_checkpoint_path),
                    metadata={
                        "exploration": exploration.state_dict(),
                        "global_step": global_step,
                        "phase_mode": self.task.config.phase_mode,
                        "phase_frequency_hz": self.task.config.phase_frequency_hz,
                        "phase_reset_value": self.task.config.phase_reset_value,
                    },
                )
                log_checkpoint_event(
                    self,
                    event_name=event_name,
                    episode=episode,
                    global_step=global_step,
                    path=str(eval_checkpoint_path),
                    note=note,
                )
                last_eval_checkpoint_step = global_step
                print(f"[eval-ckpt] saved -> {eval_checkpoint_path}")
                return True
            except Exception as error:
                print(f"[eval-ckpt][save-error] {error}")
                return False

        self._buffer_ready = bool(getattr(self, "_buffer_ready", False))
        total_falls = 0
        consecutive_fall_cycles = 0
        consecutive_stall_cycles = 0
        bc_reset_count = 0
        episodes_since_train = 0
        skip_training_after_bc_reset = False
        log_bc_reset_first_action = False
        prev_action = torch.zeros_like(INITIAL_ACTIONS[0])
        final_ep_index = -1

        # The unified Env now owns reset timing, phase origin, Webots stepping,
        # Observation construction, Action adaptation, and Reward finalization.
        self.task.configure(stop_x=stop_x)
        self.env.reset(settle_steps=RUN_CONFIG["environment"]["settle_steps"])

        for ep in range(num_episodes):
            final_ep_index = ep

            observation = self.env.begin_rollout(0)
            cycle_start_x = float(self.env.position[0])

            fell = False
            reached_goal = False
            env_steps, seq_preva = [], []

            self.algorithm.set_evaluation_mode()

            for t in range(self.task.config.sequence_length):
                action = exploration.apply(self.algorithm.act(observation))

                if t == 0 and log_bc_reset_first_action:
                    print(
                        "[bc-reset][first-action] "
                        f"com={self.get_center_of_mass():.3f} "
                        f"roll={float(observation.values[1]):.3f} "
                        f"yaw={float(observation.values[2]):.3f} "
                        f"action={action.detach().cpu().tolist()}"
                    )
                    log_bc_reset_first_action = False

                env_step = self.env.step(action, t)
                cur_x = float(env_step.info["position"][0])
                reached_goal = bool(env_step.info["reached_goal"])

                if bool(env_step.info["fell"]):
                    total_falls += 1
                    fell = True

                env_steps.append(env_step)
                seq_preva.append(prev_action)

                observation = env_step.next_observation
                prev_action = torch.tensor(env_step.action.values, dtype=action.dtype, device=action.device)

                if fell:
                    self.env.reset(settle_steps=RUN_CONFIG["environment"]["settle_steps"])
                    prev_action = torch.zeros_like(INITIAL_ACTIONS[0])
                    break

                if reached_goal:
                    print(
                        f"[goal] task={self.task.task_id} x={cur_x:.3f} "
                        f"@ episode={ep}, t={t}"
                    )
                    break

            seq_len = len(env_steps)

            ep_total_reward = 0.0
            for i in range(seq_len):
                transition = self.env.finalize_transition(
                    env_steps[i],
                    seq_preva[i],
                    rollout_fell=fell,
                    rollout_length=seq_len,
                    step_index=i,
                )
                ep_total_reward += float(transition.reward)
                self.algorithm.observe(transition)

            cycle_x_progress_m = float(self.env.position[0]) - cycle_start_x
            if fell:
                consecutive_fall_cycles += 1
                consecutive_stall_cycles = 0
            elif abs(cycle_x_progress_m) <= stall_max_abs_x_progress_m:
                consecutive_fall_cycles = 0
                consecutive_stall_cycles += 1
            else:
                consecutive_fall_cycles = 0
                consecutive_stall_cycles = 0

            bc_reset_reason = None
            if consecutive_fall_cycles >= consecutive_fall_cycle_limit:
                bc_reset_reason = "consecutive_fall_cycles"
            elif consecutive_stall_cycles >= stall_cycle_limit:
                bc_reset_reason = "consecutive_stall_cycles"

            bc_reset_applied = False
            reset_trigger_fall_cycles = consecutive_fall_cycles
            reset_trigger_stall_cycles = consecutive_stall_cycles
            completed_rollouts = int(self.ep_offset) + ep + 1
            bc_reset_allowed = within_bc_reset_window(RUN_CONFIG["initialization"], completed_rollouts)
            if bc_reset_reason is not None and RUN_CONFIG["initialization"]["reset_to_bc_on_failure"] and bc_reset_allowed:
                bc_reset_applied = self.algorithm.load_actor_initialization(
                    bc_checkpoint_path,
                    reset_optimizer=True,
                    actor_lr=cfg.ACTOR_LR,
                )
                if bc_reset_applied:
                    bc_reset_count += 1
                    consecutive_fall_cycles = 0
                    consecutive_stall_cycles = 0
                    # BC 权重刚恢复时，确保下一回合从干净的物理状态开始，
                    # 并禁止本回合用旧 Replay Buffer / Critic 立刻更新该 Actor。
                    self.env.reset(settle_steps=RUN_CONFIG["environment"]["settle_steps"])
                    prev_action = torch.zeros_like(INITIAL_ACTIONS[0])
                    # 旧策略采集的数据与恢复后的 BC Actor 分布不匹配；
                    # 清空后必须重新收集足够 transition 才允许 DDPG 更新。
                    self.algorithm.clear_experience()
                    self._buffer_ready = False
                    episodes_since_train = 0
                    skip_training_after_bc_reset = True
                    log_bc_reset_first_action = True
                print(
                    f"[bc-reset] reason={bc_reset_reason} applied={bc_reset_applied} "
                    f"fall_cycles={reset_trigger_fall_cycles}/{consecutive_fall_cycle_limit} "
                    f"stall_cycles={reset_trigger_stall_cycles}/{stall_cycle_limit} "
                    f"x_progress={cycle_x_progress_m:.4f}m reset_count={bc_reset_count}"
                )
            elif bc_reset_reason is not None and RUN_CONFIG["initialization"]["reset_to_bc_on_failure"] and not bc_reset_allowed:
                # Falls already reset the robot above. Stalls still need a physical reset.
                if not fell:
                    self.env.reset(settle_steps=RUN_CONFIG["environment"]["settle_steps"])
                    prev_action = torch.zeros_like(INITIAL_ACTIONS[0])
                consecutive_fall_cycles = 0
                consecutive_stall_cycles = 0
                print(f"[physical-reset] rollout={completed_rollouts}; BC recovery window ended; "
                      "Actor, Critic, optimizers and Replay Buffer retained")

            log_episode(
                self,
                episode=int(self.ep_offset) + ep,
                global_step=int(getattr(self, "global_step", 0)),
                reward=float(ep_total_reward),
                fell=bool(fell),
                reached_goal=bool(reached_goal),
                seq_len=int(seq_len),
                use_bc_initialization=bool(use_bc_initialization and ep == 0),
                cycle_x_progress_m=float(cycle_x_progress_m),
                consecutive_fall_cycles=int(reset_trigger_fall_cycles),
                consecutive_stall_cycles=int(reset_trigger_stall_cycles),
                bc_reset_applied=bool(bc_reset_applied),
                bc_reset_count=int(bc_reset_count),
                bc_reset_reason=bc_reset_reason,
                bc_reset_allowed=bc_reset_allowed,
            )

            if bc_reset_reason is not None:
                log_session_event(
                    self,
                    event_name="bc_reset" if bc_reset_allowed else "physical_reset_after_bc_window",
                    event_value=bool(bc_reset_applied),
                    episode=int(self.ep_offset) + ep,
                    global_step=int(getattr(self, "global_step", 0)),
                    reason=bc_reset_reason,
                    path=bc_checkpoint_path,
                    cycle_x_progress_m=float(cycle_x_progress_m),
                    consecutive_fall_cycles=int(reset_trigger_fall_cycles),
                    consecutive_fall_cycle_limit=int(consecutive_fall_cycle_limit),
                    consecutive_stall_cycles=int(reset_trigger_stall_cycles),
                    stall_cycle_limit=int(stall_cycle_limit),
                    stall_max_abs_x_progress_m=float(stall_max_abs_x_progress_m),
                    bc_reset_count=int(bc_reset_count),
                    note=("BC recovery applied; Replay Buffer cleared" if bc_reset_applied else
                          "No BC recovery; Actor, Critic, optimizers and Replay Buffer retained"),
                )

            if reached_goal:
                log_session_event(
                    self,
                    event_name="goal_reset",
                    event_value=float(cur_x),
                    episode=int(self.ep_offset) + ep,
                    global_step=int(getattr(self, "global_step", 0)),
                    reason=f"task_goal:{self.task.task_id}",
                    x_value=float(cur_x),
                    note="Goal reached; episode will terminate and training will continue after reset",
                )

            if (
                not self._buffer_ready
                and self.algorithm.experience_size >= min_buffer_for_train
            ):
                self._buffer_ready = True
                print(f"[buffer] ready now: len={self.algorithm.experience_size}")
                log_session_event(
                    self,
                    event_name="buffer_ready",
                    event_value=True,
                    episode=int(self.ep_offset) + ep,
                    global_step=int(getattr(self, "global_step", 0)),
                    note=f"buffer len={self.algorithm.experience_size}",
                )

            if skip_training_after_bc_reset:
                log_session_event(
                    self,
                    event_name="bc_reset_update_skipped",
                    event_value=True,
                    episode=int(self.ep_offset) + ep,
                    global_step=int(getattr(self, "global_step", 0)),
                    note="Skipped the pending DDPG update so the restored BC Actor starts the next rollout unchanged",
                )
                print("[bc-reset] skipped pending DDPG update; next rollout uses the restored BC Actor unchanged")
                skip_training_after_bc_reset = False
            else:
                episodes_since_train += 1

                if (self._buffer_ready
                        and (episodes_since_train >= train_every_episodes)
                        and (self.algorithm.experience_size > batch_size)):
                    update = self.algorithm.update()
                    if update.metrics.get("training_stage") == "critic_warmup":
                        completed = update.metrics["critic_warmup_completed"]
                        planned = update.metrics["critic_warmup_updates"]
                        if completed in (1, planned):
                            print(f"[critic-warmup] {completed}/{planned}; Actor unchanged; "
                                  f"joint training starts after round {planned}")
                    episodes_since_train = 0
                    self.global_step = int(getattr(self, "global_step", 0)) + 1

                    log_update(
                        self,
                        episode=int(self.ep_offset) + ep,
                        global_step=int(self.global_step),
                        actor_loss=float(update.actor_loss),
                        critic_loss=float(update.critic_loss),
                        buffer_len=int(self.algorithm.experience_size),
                        total_falls=int(total_falls),
                        metrics=dict(update.metrics),
                        episodes_in_span=int(train_every_episodes),
                    )

                    if (
                        eval_checkpoint_every_updates
                        and self.global_step % eval_checkpoint_every_updates == 0
                    ):
                        save_eval_candidate(
                            event_name="eval_ckpt_saved",
                            episode=int(self.ep_offset) + ep,
                            note=(
                                "saved for independent evaluation every "
                                f"{eval_checkpoint_every_updates} gradient updates"
                            ),
                        )

            if reached_goal:
                self.env.reset(settle_steps=RUN_CONFIG["environment"]["settle_steps"])
                prev_action = torch.zeros_like(INITIAL_ACTIONS[0])
                print(f"[goal-reset] episode={ep} reset to initial position; training continues")

        
        finished_episodes = final_ep_index + 1 if final_ep_index >= 0 else 0
        self.ep_offset = int(getattr(self, "ep_offset", 0)) + finished_episodes
        self.algorithm.save_checkpoint(
                        stop_ckpt_path,
                        metadata={
                        "exploration": exploration.state_dict(),
                            "global_step": int(getattr(self, "global_step", 0)),
                            "ep_offset": int(getattr(self, "ep_offset", 0)),
                            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                            "reason": "session_end",
                            "total_falls": int(total_falls),
                            "seed": int(seed),
                            "phase_mode": self.task.config.phase_mode,
                            "phase_frequency_hz": self.task.config.phase_frequency_hz,
                            "phase_reset_value": self.task.config.phase_reset_value,
                        },
                    )
        if last_eval_checkpoint_step != int(getattr(self, "global_step", 0)):
            save_eval_candidate(
                event_name="final_eval_ckpt_saved",
                episode=int(self.ep_offset),
                note="saved as the final independent-evaluation candidate for this training session",
            )
        log_session_event(
            self,
            event_name="session_end",
            event_value=True,
            seed=seed,
            episode=int(getattr(self, "ep_offset", 0)),
            global_step=int(getattr(self, "global_step", 0)),
            note="train_model call finished",
        )

        _save_meta(self)


def main():
    """Webots training entrypoint. Edit run_config.py to select a run."""
    require_launch()
    from world_reload_state import load_continuation, restore_rng
    continuation = load_continuation(os.environ[SNAPSHOT_ENV])
    paths = ExperimentPaths(EXPERIMENT_NAME, SEED)
    experiment_exists = paths.root.exists()
    resume_checkpoint_exists = (RUN_CONFIG["initialization"]["resume"] != "never" and paths.resume_checkpoint.is_file())

    if continuation is not None:
        print(f"[startup] Experiment '{EXPERIMENT_NAME}': continuing after world reload.")
    elif not experiment_exists:
        print(f"[startup] Experiment '{EXPERIMENT_NAME}' not found; creating a new experiment.")
    elif resume_checkpoint_exists:
        print(f"[startup] Found resume checkpoint: {paths.resume_checkpoint}")
    elif RUN_CONFIG["initialization"]["resume"] == "never":
        print(f"[startup] Ordinary checkpoint resume disabled; starting a new training session.")
    else:
        print(f"[startup] Experiment '{EXPERIMENT_NAME}' exists but has no resume checkpoint; starting fresh.")

    paths.ensure_directories()

    controller = Sprinter(cfg, seed=SEED)
    controller.initialize()

    resume_metadata = {}
    if continuation is not None:
        controller.algorithm = continuation['algorithm']
        controller.algorithm.set_evaluation_mode()
        resume_metadata = continuation['metadata']
        restore_rng(continuation['rng'])
        resume_checkpoint_exists = True
        print(f"[world-reload] restored learner, Replay={controller.algorithm.experience_size}, "
              f"env_steps={resume_metadata['continuous_training']['env_steps']}")
    elif resume_checkpoint_exists:
        resume_metadata = controller.algorithm.load_checkpoint(str(paths.resume_checkpoint))

    startup_bc = USE_BC_INITIALIZATION and (not resume_checkpoint_exists or RUN_CONFIG["initialization"]["bc_on_resume"])
    if continuation is not None:
        startup_bc = False
    if resume_checkpoint_exists:
        if RUN_CONFIG['training']['sampling_mode'] == 'continuous':
            state = resume_metadata.get('continuous_training')
            if state is None:
                raise ValueError('Start a new experiment from BC; legacy checkpoints cannot resume continuous training')
            cutoff = RUN_CONFIG['initialization']['bc_reset_max_env_steps']
            if cutoff and state['env_steps'] >= cutoff:
                startup_bc = False
        elif not within_bc_reset_window(RUN_CONFIG['initialization'], int(resume_metadata.get('ep_offset', 0)) + 1):
            startup_bc = False
    if startup_bc:
        controller.algorithm.load_actor_initialization(
            str(CONTROLLER_DIR / BC_CHECKPOINT_FILENAME),
            reset_optimizer=False,
            actor_lr=ACTOR_LR,
        )

    print(f"[startup] Experiment: {EXPERIMENT_NAME}")
    print(f"[startup] Task: {controller.task.task_id}")
    print(f"[startup] Algorithm: {controller.algorithm.algorithm_id}")
    print(f"[startup] BC Actor reset: {'enabled' if startup_bc else 'disabled'}")

    controller.train_model(
        TRAIN_EPISODES,
        min_buffer_for_train=MIN_BUFFER_FOR_TRAIN,
        train_every_episodes=TRAIN_EVERY_EPISODES,
        stop_x=STOP_X,
        consecutive_fall_cycle_limit=CONSECUTIVE_FALL_CYCLE_LIMIT,
        stall_cycle_limit=STALL_CYCLE_LIMIT,
        stall_max_abs_x_progress_m=STALL_MAX_ABS_X_PROGRESS_M,
        bc_checkpoint_path=str(CONTROLLER_DIR / BC_CHECKPOINT_FILENAME),
        use_bc_initialization=startup_bc,
        resume_checkpoint_path=str(paths.resume_checkpoint) if resume_checkpoint_exists else None,
        resume_metadata=resume_metadata,
        seed=SEED,
        eval_checkpoint_every_updates=EVAL_CHECKPOINT_EVERY_UPDATES,
        experiment_name=EXPERIMENT_NAME,
    )


if __name__ == "__main__":
    main()
