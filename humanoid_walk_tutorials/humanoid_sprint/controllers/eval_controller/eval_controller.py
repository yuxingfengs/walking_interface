from controller import Supervisor
import math
import csv
import re
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn


# =========================
# Config
# =========================

STATE_DIM = 18
ACTION_DIM = 10
SEQUENCE_LENGTH = 35

INITIAL_STATE = [
    2.813063620922607e-07, -2.8354001787624697e-07,
    2.661361167672216e-07, 2.680192760864757e-07,
    2.967163694572752e-07, 2.9883194662671264e-07,
    2.4820736847423014e-07, 2.4949180207911914e-07,
    1.9713857411518186e-07, 1.9756885276656546e-07,
    -1.866298555697199e-07, 1.8715301552270632e-07,
    [-5.021522061737612, 0.9999547666320551, 0.7058214811932133]
]


# =========================
# Model (eval only)
# =========================

class Actor(nn.Module):
    def __init__(self, state_dim: int, hidden_dims, action_dim: int):
        super().__init__()

        if isinstance(hidden_dims, int):
            hidden_dims = (hidden_dims,)

        layers = []
        inp = state_dim
        for h in hidden_dims:
            layers += [nn.Linear(inp, h), nn.ReLU()]
            inp = h
        layers.append(nn.Linear(inp, action_dim))
        self.net = nn.Sequential(*layers)

        self.register_buffer("joint_min", torch.tensor([
            -0.379, -1.774, -0.092, -1.189, -0.397,
            -0.738, -1.774, -0.092, -1.186, -0.769
        ], dtype=torch.float32))
        self.register_buffer("joint_max", torch.tensor([
            0.790, 0.484, 2.113, 0.923, 0.769,
            0.450, 0.484, 2.113, 0.932, 0.38
        ], dtype=torch.float32))

    def forward(self, state):
        if isinstance(state, (list, tuple, np.ndarray)):
            state = torch.tensor(state, dtype=torch.float32, device=self.net[0].weight.device)
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float32, device=self.net[0].weight.device)

        if state.dim() == 1:
            state = state.unsqueeze(0)

        raw = self.net(state)
        action = torch.max(torch.min(raw, self.joint_max), self.joint_min)
        return action


class EvalAgent:
    def __init__(self, device: torch.device):
        self.device = device
        self.actor = Actor(STATE_DIM, 256, ACTION_DIM).to(device)

    def load_actor_from_checkpoint(self, filepath: str):
        ckpt = torch.load(filepath, map_location=self.device)
        if "actor_state_dict" in ckpt:
            self.actor.load_state_dict(ckpt["actor_state_dict"])
        elif "actor" in ckpt:
            self.actor.load_state_dict(ckpt["actor"])
        else:
            raise KeyError(f"No actor weights found in checkpoint: {filepath}")
        self.actor.to(self.device)
        self.actor.eval()

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


def build_eval_ckpt_dir(seed: int, base_dir: str = "check_points") -> Path:
    return Path(base_dir) / f"seed_{seed}" / "eval_ckpts"


def load_existing_eval_steps(metrics_csv: Path) -> set:
    if not metrics_csv.exists():
        return set()

    done = set()
    with metrics_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("type", "")).strip().lower() != "eval":
                continue
            gs = row.get("global_step", "")
            if gs != "":
                try:
                    done.add(int(gs))
                except Exception:
                    pass
    return done


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
        "checkpoint_path": checkpoint_path,
        "note": "independent_eval_from_saved_checkpoint",
    }
    append_metric(host, row, filename=filename)


# =========================
# Standalone Eval Controller
# =========================

class EvalController(Supervisor):
    def __init__(self):
        super().__init__()
        self.timeStep = 40
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent = EvalAgent(self.device)

    def initialize(self):
        self.timeStep = 40

        self.RShoulderPitch = self.getDevice('RShoulderPitch')
        self.LShoulderPitch = self.getDevice('LShoulderPitch')
        self.RShoulderPitch.setPosition(1.1)
        self.LShoulderPitch.setPosition(1.1)

        self.joint_names = [
            'RHipYawPitch', 'LHipYawPitch',
            'RHipRoll', 'LHipRoll',
            'RHipPitch', 'LHipPitch',
            'RKneePitch', 'LKneePitch',
            'RAnklePitch', 'LAnklePitch',
            'RAnkleRoll', 'LAnkleRoll'
        ]

        self.motors = {name: self.getDevice(name) for name in self.joint_names}
        self.motor_order = [
            'LHipRoll', 'LHipPitch', 'LKneePitch', 'LAnklePitch', 'LAnkleRoll',
            'RHipRoll', 'RHipPitch', 'RKneePitch', 'RAnklePitch', 'RAnkleRoll'
        ]
        self.motors_list = [self.motors[name] for name in self.motor_order]

        self.sensors = {}
        for name in self.joint_names:
            sensor_name = name + "S"
            sensor = self.getDevice(sensor_name)
            sensor.enable(self.timeStep)
            self.sensors[sensor_name] = sensor
            setattr(self, sensor_name.replace("S", "Sensor"), sensor)

        self.accelerometer = self.getDevice("accelerometer")
        self.imu = self.getDevice("inertial unit")
        self.gyro = self.getDevice("gyro")
        self.accelerometer.enable(self.timeStep)
        self.imu.enable(self.timeStep)
        self.gyro.enable(self.timeStep)

        for _ in range(5):
            self.step(self.timeStep)

        self.nao = self.getFromDef("NAO")
        self.translation = self.nao.getField("translation")
        self.rotation = self.nao.getField("rotation")


        self.capture_initial_state()

        self.pre_position = self.get_translation()
        self.start_position = self.get_translation()
        self.current_position = self.get_translation()
        self.pre_yaw = self.get_yaw()
        self.initYaw = self.get_yaw()
        self.start_yaw = self.get_yaw()
        self.current_yaw = self.get_yaw()

    def _set_default_motor_velocity(self, scale: float = 0.8):
        for m in getattr(self, "motors_list", []):
            try:
                vmax = m.getMaxVelocity()
            except Exception:
                vmax = 6.0
            m.setVelocity(float(vmax) * float(scale))

    def get_translation(self):
        return self.translation.getSFVec3f()

    def get_rotation(self):
        return self.rotation.getSFRotation()

    def get_yaw(self):
        roll, pitch, yaw = self.imu.getRollPitchYaw()
        return yaw

    def get_center_of_mass(self):
        return self.get_translation()[2]

    def capture_initial_state(self):
        self.init_translation = self.get_translation()
        self.init_rotation = self.get_rotation()
        self.initYaw = float(self.get_yaw())

        self.init_q = []
        for m in getattr(self, "motors_list", []):
            try:
                q0 = m.getTargetPosition()
            except Exception:
                q0 = 0.0
            self.init_q.append(q0)

        self.pre_position = self.init_translation
        self.start_position = self.init_translation

    def resetRobot(self, initial_state):
        self.step(self.timeStep)

        joint_order = [
            'RHipRoll', 'LHipRoll',
            'RHipYawPitch', 'LHipYawPitch',
            'RHipPitch', 'LHipPitch',
            'RKneePitch', 'LKneePitch',
            'RAnklePitch', 'LAnklePitch',
            'RAnkleRoll', 'LAnkleRoll'
        ]

        for joint_name, angle in zip(joint_order, initial_state[:12]):
            if joint_name in self.motors:
                self.motors[joint_name].setPosition(angle)

        self.translation.setSFVec3f(initial_state[12])
        self.rotation.setSFRotation([0, 0, 1, 0])

        for _ in range(64):
            self.step(self.timeStep)

    def hard_reset(self, settle_steps: int = 64, margin: float = 1e-3, use_supervisor: bool = True):
        try:
            self.resetRobot(INITIAL_STATE)
        except Exception:
            pass

        for m, q0 in zip(getattr(self, "motors_list", []), getattr(self, "init_q", [])):
            try:
                lo = m.getMinPosition() + margin
                hi = m.getMaxPosition() - margin
            except Exception:
                lo, hi = -1e9, 1e9
            m.setVelocity(0.0)
            m.setPosition(min(max(q0, lo), hi))

        if use_supervisor and getattr(self, "supervisor", None) is not None:
            try:
                node = self.supervisor.getSelf()
                if node is not None:
                    node.resetPhysics()
            except Exception:
                pass

        for _ in range(int(settle_steps)):
            self.step(self.timeStep)

        self.pre_position = self.get_translation()
        self.start_position = self.get_translation()
        self.start_yaw = float(self.get_yaw())
        self._set_default_motor_velocity(scale=0.8)

    def apply_action_to_robot(self, action):
        joint_order = [
            'LHipRoll', 'LHipPitch', 'LKneePitch', 'LAnklePitch', 'LAnkleRoll',
            'RHipRoll', 'RHipPitch', 'RKneePitch', 'RAnklePitch', 'RAnkleRoll'
        ]
        for i, name in enumerate(joint_order):
            self.motors[name].setPosition(action[i].item())

    def get_current_state(self, step):
        roll, pitch, yaw1 = self.imu.getRollPitchYaw()

        x, y, z, theta = self.rotation.getSFRotation()
        qw = math.cos(theta / 2)
        sin_t2 = math.sin(theta / 2)
        qx = x * sin_t2
        qy = y * sin_t2
        qz = z * sin_t2
        yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))

        gyro_x, gyro_y, gyro_z = self.gyro.getValues()
        phase = (step % SEQUENCE_LENGTH) / SEQUENCE_LENGTH

        joint_order = [
            'RHipYawPitch', 'LHipYawPitch',
            'RHipRoll', 'LHipRoll',
            'RHipPitch', 'LHipPitch',
            'RKneePitch', 'LKneePitch',
            'RAnklePitch', 'LAnklePitch',
            'RAnkleRoll', 'LAnkleRoll'
        ]
        joint_angles = [self.sensors[name + 'S'].getValue() for name in joint_order]

        cm_raw = self.get_translation()[2]
        CM_MIN, CM_MAX = 0.6, 0.72
        cm_norm = (cm_raw - CM_MIN) / (CM_MAX - CM_MIN)
        cm_norm = min(max(cm_norm, 0.0), 1.0)

        state = [phase, roll, yaw, gyro_x, gyro_y, cm_norm] + joint_angles
        return state

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
        results = []
        dt = self.timeStep / 1000.0

        for ep in range(num_episodes):
            print(f"[eval] episode {ep + 1}/{num_episodes}")

            self.hard_reset(settle_steps=64, use_supervisor=True)
            self.step(self.timeStep)

            start_pos = self.get_translation()
            start_x, start_y = start_pos[0], start_pos[1]

            raw = self.get_current_state(0)
            state = torch.tensor(raw, dtype=torch.float32, device=self.device)

            fell = False
            reach_time = None
            step_counter = 0

            for t in range(max_steps):
                with torch.no_grad():
                    action = self.agent.act(state)

                self.apply_action_to_robot(action)
                self.step(self.timeStep)
                step_counter += 1

                pos = self.get_translation()
                cur_x, cur_y = pos[0], pos[1]

                if stop_x is not None and cur_x >= stop_x:
                    reach_time = step_counter * dt
                    print(f"[eval] reached x >= {stop_x:.3f} m at t={reach_time:.2f} s")
                    break

                raw_next = self.get_current_state((t + 1) % seq_len)
                state = torch.tensor(raw_next, dtype=torch.float32, device=self.device)

                if self.get_center_of_mass() < 0.6:
                    print(f"[eval] fell at step {step_counter}")
                    fell = True
                    break

            end_pos = self.get_translation()
            forward_dist = end_pos[0] - start_x
            lateral_dist = end_pos[1] - start_y
            elapsed_time = step_counter * dt
            reached = (stop_x is not None and end_pos[0] >= stop_x)

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
            })

        return results


def evaluate_saved_checkpoints(
    controller: EvalController,
    *,
    seed: int,
    eval_episodes: int = 5,
    stop_x: float = 4.5,
    max_steps: int = 6000,
    ckpt_dir_base: str = "check_points",
    log_dir_base: str = "runs/walk_exp1",
    only_latest: bool = False,
):
    log_dir = f"{log_dir_base}/seed_{seed}"
    prepare_eval_logging(controller, log_dir=log_dir)

    ckpt_dir = build_eval_ckpt_dir(seed=seed, base_dir=ckpt_dir_base)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpts = sorted(
        ckpt_dir.glob("ckpt_gs*.pth"),
        key=lambda p: extract_global_step_from_ckpt_name(p.name) or -1
    )

    if not ckpts:
        print(f"[eval] no checkpoints found in {ckpt_dir}")
        return

    metrics_csv = Path(log_dir) / "metrics.csv"
    evaluated_steps = load_existing_eval_steps(metrics_csv)

    if only_latest:
        ckpts = [ckpts[-1]]

    pending = []
    for ckpt in ckpts:
        gs = extract_global_step_from_ckpt_name(ckpt.name)
        if gs is None:
            continue
        if gs in evaluated_steps and not only_latest:
            continue
        pending.append((gs, ckpt))

    if not pending:
        print("[eval] no pending checkpoints to evaluate")
        return

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
            filename="metrics.csv",
        )

        print(
            f"[eval] done gs={gs} | "
            f"success_rate={summary['eval_success_rate']:.3f} | "
            f"forward_mean={summary['forward_mean']:.3f} | "
            f"fell_rate={summary['fell_rate']:.3f}"
        )


def main():
    # 直接在这里改参数也行
    seed = 1
    eval_episodes = 5
    stop_x = 4.5
    max_steps = 6000
    ckpt_dir_base = "check_points"
    log_dir_base = "runs/walk_exp1"
    only_latest = False

    controller = EvalController()
    controller.initialize()

    evaluate_saved_checkpoints(
        controller,
        seed=seed,
        eval_episodes=eval_episodes,
        stop_x=stop_x,
        max_steps=max_steps,
        ckpt_dir_base=ckpt_dir_base,
        log_dir_base=log_dir_base,
        only_latest=only_latest,
    )


if __name__ == "__main__":
    main()