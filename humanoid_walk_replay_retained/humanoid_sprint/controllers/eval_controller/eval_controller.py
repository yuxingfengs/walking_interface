from controller import Supervisor
import csv
import json
import os
import random
import re
from pathlib import Path
from typing import Optional, Dict, Any, List
import sys

import numpy as np
import torch

CONTROLLER_DIR = Path(__file__).resolve().parent
CONTROLLERS_DIR = CONTROLLER_DIR.parent
COMMON_DIR = CONTROLLER_DIR.parent / "common"
for directory in (str(COMMON_DIR), str(CONTROLLERS_DIR)):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from checkpointing import load_actor_weights
from evaluation_batch import finish_evaluation, PROTOCOL
from eval_stall import EvalStallMonitor
from experiment_paths import ExperimentPaths
from models import BoundedActor
from envs.nao_locomotion_env import NaoLocomotionEnv
from webots_platform.webots_runtime import WebotsRuntime
from tasks import load_task
from contracts import (
    ACTION_DIM,
    STATE_DIM,
)
from run_config import (
    RUN_CONFIG,
    EVAL_EPISODES,
    EXPERIMENT_NAME,
    EVAL_METRICS_FILENAME,
    MAX_STEPS,
    ONLY_LATEST,
    PHASE_FREQUENCY_HZ,
    PHASE_MODE,
    PHASE_RESET_VALUE,
    SEED,
    SELECTED_GLOBAL_STEP,
    START_GLOBAL_STEP,
    STOP_X,
    TASK_MODULE,
    TASK_CLASS,
    task_config,
)


from configs.runtime import require_launch, runtime_settings, evaluation_checkpoint_paths, SNAPSHOT_ENV

# =========================
# Config
# =========================

SEQUENCE_LENGTH = RUN_CONFIG["training"]["rollout_steps"]

# =========================
# Model (eval only)
# =========================

class EvalAgent:
    def __init__(self, device: torch.device, task_id: str):
        self.device = device
        self.task_id = task_id
        from configs.runtime import create_actor
        self.actor = create_actor(RUN_CONFIG).to(device)

    def load_actor_from_checkpoint(self, filepath: str):
        load_actor_weights(
            self.actor,
            filepath,
            self.device,
            expected_task_id=self.task_id,
        )

    @torch.no_grad()
    def act(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        action = self.actor(state).squeeze(0)
        return action


# =========================
# Logging helpers (standalone)
# =========================

def prepare_eval_logging(host, log_dir: str) -> None:
    host.log_dir = Path(log_dir)
    host.log_dir.mkdir(parents=True, exist_ok=True)
    host.metrics_csv = host.log_dir / "metrics.csv"


def append_metric(host, row: Dict[str, Any], filename: str = "metrics.csv") -> None:
    p = Path(getattr(host, "log_dir", ".")) / filename
    p.parent.mkdir(parents=True, exist_ok=True)

    if not p.exists():
        fieldnames = list(row.keys())
        with p.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerow(row)
        return

    with p.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        old_fields = r.fieldnames or []
        old_rows = list(r)

    new_fields = [k for k in row.keys() if k not in old_fields]
    if not new_fields:
        with p.open("a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=old_fields)
            w.writerow(row)
        return

    merged = []
    seen = set()
    for k in list(old_fields) + list(row.keys()):
        if k not in seen:
            merged.append(k)
            seen.add(k)

    tmp = p.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=merged)
        w.writeheader()
        for old in old_rows:
            w.writerow({k: old.get(k, "") for k in merged})
        w.writerow({k: row.get(k, "") for k in merged})
    tmp.replace(p)


def extract_global_step_from_ckpt_name(filename: str) -> Optional[int]:
    m = re.search(r"ckpt_gs(\d+)\.pth$", filename)
    return int(m.group(1)) if m else None


def summarize_eval_run_results(results: List[dict], stop_x: float) -> Dict[str, Any]:
    if not results:
        return {
            "eval_episodes": 0,
            "eval_success_rate": 0.0,
            "eval_ep_len_mean": None,
            "forward_mean": None,
            "forward_std": None,
            "lateral_mean": None,
            "lateral_std": None,
            "reach_time_mean": None,
            "reach_time_std": None,
            "fell_rate": None,
        }

    forwards = [float(r.get("forward", 0.0)) for r in results]
    laterals = [float(r.get("lateral", 0.0)) for r in results]
    steps = [int(r.get("steps", 0)) for r in results]
    fells = [1 if bool(r.get("fell", False)) else 0 for r in results]

    successes = []
    reach_times = []
    for r in results:
        reached = r.get("reached", None)
        if reached is None:
            reached = float(r.get("forward", 0.0)) >= float(stop_x)
        successes.append(1 if reached else 0)

        rt = r.get("reach_time", None)
        if rt is not None:
            reach_times.append(float(rt))

    return {
        "eval_episodes": len(results),
        "eval_success_rate": float(np.mean(successes)),
        "eval_ep_len_mean": float(np.mean(steps)),
        "forward_mean": float(np.mean(forwards)),
        "forward_std": float(np.std(forwards, ddof=0)),
        "lateral_mean": float(np.mean(laterals)),
        "lateral_std": float(np.std(laterals, ddof=0)),
        "reach_time_mean": float(np.mean(reach_times)) if reach_times else None,
        "reach_time_std": float(np.std(reach_times, ddof=0)) if reach_times else None,
        "fell_rate": float(np.mean(fells)),
    }


def log_eval_result(
    host,
    *,
    global_step: int,
    seed: int,
    checkpoint_path: str,
    summary: Dict[str, Any],
    filename: str = "metrics.csv",
) -> None:
    row = {
        "type": "eval",
        "global_step": int(global_step),
        "episode": int(getattr(host, "ep_offset", 0)),
        "seed": int(seed),
        "eval_episodes": summary.get("eval_episodes"),
        "eval_success_rate": summary.get("eval_success_rate"),
        "eval_ep_len_mean": summary.get("eval_ep_len_mean"),
        "forward_mean": summary.get("forward_mean"),
        "forward_std": summary.get("forward_std"),
        "lateral_mean": summary.get("lateral_mean"),
        "lateral_std": summary.get("lateral_std"),
        "reach_time_mean": summary.get("reach_time_mean"),
        "reach_time_std": summary.get("reach_time_std"),
        "fell_rate": summary.get("fell_rate"),
        "phase_mode": PHASE_MODE,
        "phase_frequency_hz": PHASE_FREQUENCY_HZ,
        "phase_reset_value": PHASE_RESET_VALUE,
        "task_id": host.task.task_id,
        "checkpoint_path": checkpoint_path,
        "note": "independent_eval_from_saved_checkpoint",
    }
    payload = json.loads(Path(os.environ[SNAPSHOT_ENV]).read_text(encoding="utf-8"))
    row.update(eval_protocol=payload.get('reproducibility_protocol', PROTOCOL), eval_run_id=payload["run_id"],
               controller_pid=os.getpid(), episodes=host.last_eval_episodes,
               checkpoint_sha256=payload["sources"][0]["sha256"])
    append_metric(host, row, filename=filename)
    (Path(os.environ[SNAPSHOT_ENV]).parent / "evaluation_result.json").write_text(
        json.dumps(row, indent=2), encoding="utf-8")


# =========================
# Standalone Eval Controller
# =========================

class EvalController(Supervisor):
    def __init__(self):
        super().__init__()
        self.timeStep = RUN_CONFIG["webots"]["control_time_step_ms"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.runtime = WebotsRuntime(self, **runtime_settings(RUN_CONFIG))
        self.task = load_task(TASK_MODULE, TASK_CLASS, task_config())
        self.agent = EvalAgent(self.device, self.task.task_id)
        self.env = NaoLocomotionEnv(self.runtime, self.task)

    def initialize(self):
        self.env.initialize()

    def run_trained_actor(
        self,
        checkpoint_path: str,
        num_episodes: int = 5,
        stop_x: float = 4.5,
        max_steps: int = 6000,
        sequence_length: int = None,
    ):
        print(f"[eval] loading checkpoint from {checkpoint_path}")
        self.agent.load_actor_from_checkpoint(checkpoint_path)

        seq_len = sequence_length or SEQUENCE_LENGTH
        self.task.configure(sequence_length=seq_len, stop_x=stop_x)
        results = []
        dt = self.timeStep / 1000.0
        first_episode_max_steps = max(1, int(max_steps * RUN_CONFIG["evaluation"]["first_episode_step_fraction"]))

        for ep in range(num_episodes):
            print(f"[eval] episode {ep + 1}/{num_episodes}")

            observation = self.env.reset(settle_steps=RUN_CONFIG["environment"]["settle_steps"])
            start_state = dict(observation=list(observation.values),
                position=list(self.env.position), rotation=list(self.runtime.rotation.getSFRotation()),
                velocity=list(self.getSelf().getVelocity()), simulation_time=self.getTime())
            trajectory, observations, actions = [], [], []
            start_pos = self.env.position
            start_x, start_y = start_pos[0], start_pos[1]
            state = torch.tensor(observation.values, dtype=torch.float32, device=self.device)

            fell = False
            reach_time = None
            reached = False
            end_reason = 'timeout'
            stall = EvalStallMonitor(start_x,
                window_steps=RUN_CONFIG['evaluation']['stall_window_steps'],
                cycle_limit=RUN_CONFIG['evaluation']['stall_cycle_limit'],
                max_abs_progress_m=RUN_CONFIG['evaluation']['stall_max_abs_x_progress_m'])
            step_counter = 0
            episode_max_steps = first_episode_max_steps if ep == 0 else max_steps

            for t in range(episode_max_steps):
                with torch.no_grad():
                    action = self.agent.act(state)
                observations.append(state.detach().cpu().tolist())
                actions.append(action.detach().cpu().tolist())

                env_step = self.env.step(action, t)
                trajectory.append(list(self.env.position))
                step_counter += 1

                if bool(env_step.info["reached_goal"]):
                    reached = True
                    end_reason = 'goal'
                    reach_time = step_counter * dt
                    print(
                        f"[eval] reached task goal ({self.task.task_id}) "
                        f"at t={reach_time:.2f} s"
                    )
                    break

                state = torch.tensor(
                    env_step.next_observation.values,
                    dtype=torch.float32,
                    device=self.device,
                )

                if bool(env_step.info["fell"]):
                    print(f"[eval] fell at step {step_counter}")
                    fell = True
                    end_reason = 'fall'
                    break

                if stall.observe(env_step.info['position'][0]):
                    end_reason = 'stall'
                    print(f'[eval] stalled for {stall.stall_cycles} windows; ending current evaluation at step {step_counter}')
                    break

            end_pos = self.env.position
            forward_dist = end_pos[0] - start_x
            lateral_dist = end_pos[1] - start_y
            elapsed_time = step_counter * dt
            print(
                f"[eval] result: forward={forward_dist:.3f} m, "
                f"lateral={lateral_dist:.3f} m, fell={fell}, "
                f"time={elapsed_time:.2f} s, steps={step_counter}, reach_time={reach_time}"
            )

            results.append({
                "episode": ep + 1,
                "forward": forward_dist,
                "lateral": lateral_dist,
                "fell": fell,
                "time": elapsed_time,
                "steps": step_counter,
                "reach_time": reach_time,
                "reached": reached,
                "end_reason": end_reason,
                "stall_cycles": stall.stall_cycles,
                "start_state": start_state,
            })
            folder = Path(os.environ[SNAPSHOT_ENV]).parent
            np.savez_compressed(folder / f'rollout_{ep + 1:03d}.npz',
                states=np.asarray(observations), actions=np.asarray(actions), trajectory=np.asarray(trajectory))

            if ep == 0 and reach_time is None and RUN_CONFIG["evaluation"]["stop_after_first_failure"]:
                print(
                    f"[eval] first episode ended: {end_reason}; skipping remaining episodes"
                )
                break

        self.last_eval_episodes = results
        return results


def evaluate_saved_checkpoints(
    controller: EvalController,
    *,
    seed: int,
    eval_episodes: int = 5,
    stop_x: float = 4.5,
    max_steps: int = 6000,
    experiment_name: str = EXPERIMENT_NAME,
    start_global_step: int = 0,
    selected_global_step: Optional[int] = None,
    metrics_filename: str = "metrics.csv",
    only_latest: bool = False,
):
    experiment_paths = ExperimentPaths(experiment_name, seed)
    experiment_paths.ensure_directories()
    log_dir = experiment_paths.evaluation_log_dir
    prepare_eval_logging(controller, log_dir=str(log_dir))

    ckpt_dir = experiment_paths.eval_candidates_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpts = sorted(
        ckpt_dir.glob("ckpt_gs*.pth"),
        key=lambda p: extract_global_step_from_ckpt_name(p.name) or -1
    )
    selected_paths = evaluation_checkpoint_paths()
    if selected_paths is not None:
        ckpts = [p for p in ckpts if str(p.resolve()) in selected_paths]

    if not ckpts:
        print(f"[eval] no checkpoints found in {ckpt_dir}")
        return


    if only_latest:
        ckpts = [ckpts[-1]]

    pending = []
    for ckpt in ckpts:
        gs = extract_global_step_from_ckpt_name(ckpt.name)
        if gs is None:
            continue
        if gs < start_global_step:
            continue
        if selected_global_step is not None and gs != selected_global_step:
            continue
        pending.append((gs, ckpt))

    if not pending:
        print("[eval] no pending checkpoints to evaluate")
        return

    if len(pending) != 1:
        raise RuntimeError("Evaluation requires exactly one Checkpoint per Controller initialization; use launch.py eval")

    for gs, ckpt in pending:
        print(f"[eval] evaluating {ckpt.name} (global_step={gs})")

        results = controller.run_trained_actor(
            checkpoint_path=str(ckpt),
            num_episodes=eval_episodes,
            stop_x=stop_x,
            max_steps=max_steps,
        )

        summary = summarize_eval_run_results(results, stop_x=stop_x)

        log_eval_result(
            controller,
            global_step=gs,
            seed=seed,
            checkpoint_path=str(ckpt),
            summary=summary,
            filename=metrics_filename,
        )

        print(
            f"[eval] done gs={gs} | "
            f"success_rate={summary['eval_success_rate']:.3f} | "
            f"forward_mean={summary['forward_mean']:.3f} | "
            f"fell_rate={summary['fell_rate']:.3f}"
        )


def main():
    require_launch()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    controller = EvalController()
    try:
        controller.initialize()
        evaluate_saved_checkpoints(
            controller,
            seed=SEED,
            eval_episodes=EVAL_EPISODES,
            stop_x=STOP_X,
            max_steps=MAX_STEPS,
            experiment_name=EXPERIMENT_NAME,
            start_global_step=START_GLOBAL_STEP,
            selected_global_step=SELECTED_GLOBAL_STEP,
            metrics_filename=EVAL_METRICS_FILENAME,
            only_latest=ONLY_LATEST,
        )
    except BaseException:
        controller.simulationQuit(1)
        raise
    else:
        try:
            payload = json.loads(Path(os.environ[SNAPSHOT_ENV]).read_text(encoding='utf-8'))
            finish_evaluation(controller, payload)
        except BaseException:
            controller.simulationQuit(1)
            raise



if __name__ == "__main__":
    main()
