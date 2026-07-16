# -*- coding: utf-8 -*-
import os
import csv
import json, re, time
import copy
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Iterable, Literal, Any

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch

# =========================
# 基础工具 & 日志
# =========================

def rotate_metrics_csv(
    log_dir: str,
    metrics_name: str = "metrics.csv",
    *,
    scheme: str = "timestamp",     # "timestamp" 或 "index"
    keep: int = 5,                 # 最多保留多少个历史备份
    copy_header: bool = False,     # 备份后是否创建一个带表头的新 metrics.csv
    encoding: str = "utf-8"
) -> Optional[str]:
    """
    轮转/归档 metrics.csv。
    - 若文件不存在，返回 None（不报错）。
    - 成功时返回备份文件的完整路径字符串。
    - scheme="timestamp": 生成 metrics_YYYYmmdd-HHMMSS.csv
      scheme="index":     生成 metrics.001.csv, metrics.002.csv, ...
    - copy_header=True 时会把原文件的首行（表头）复制到新的空 metrics.csv 里。
    """
    p = Path(log_dir) / metrics_name
    if not p.exists():
        return None

    out_dir = p.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    header_line = None
    if copy_header:
        try:
            with p.open("r", encoding=encoding, newline="") as f:
                header_line = f.readline().rstrip("\n\r")
        except Exception:
            header_line = None

    if scheme == "timestamp":
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        dst = out_dir / f"{p.stem}_{ts}{p.suffix}"
        counter = 1
        while dst.exists():
            dst = out_dir / f"{p.stem}_{ts}_{counter:02d}{p.suffix}"
            counter += 1
    elif scheme == "index":
        idx = 1
        def fmt(i: int) -> Path: return out_dir / f"{p.stem}.{i:03d}{p.suffix}"
        dst = fmt(idx)
        while dst.exists():
            idx += 1
            dst = fmt(idx)
    else:
        raise ValueError("scheme 必须是 'timestamp' 或 'index'")

    p.replace(dst)

    if copy_header and header_line:
        try:
            with (out_dir / metrics_name).open("w", encoding=encoding, newline="") as f:
                f.write(header_line + "\n")
        except Exception:
            pass

    # 清理旧备份
    try:
        if scheme == "timestamp":
            backups = sorted(out_dir.glob(f"{p.stem}_*{p.suffix}"),
                             key=lambda x: x.stat().st_mtime, reverse=True)
        else:
            backups = sorted(out_dir.glob(f"{p.stem}.*{p.suffix}"),
                             key=lambda x: x.stat().st_mtime, reverse=True)
        for old in backups[keep:]:
            try: old.unlink()
            except Exception: pass
    except Exception:
        pass

    return str(dst)


def load_initial_actions_csv(
    filename: str,
    expected_dim: Optional[int] = None,
    to_radians: bool = False,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> List[torch.Tensor]:
    """从 CSV 文件读取 initial actions，返回 tensor 列表。"""
    rows = []
    with open(filename, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for line in reader:
            if not line: continue
            rows.append([float(x) for x in line])

    arr = np.asarray(rows, dtype=np.float32)

    if expected_dim is not None and arr.shape[1] != expected_dim:
        raise ValueError(
            f"initial_actions dim mismatch: got {arr.shape[1]}, expect {expected_dim}"
        )

    if to_radians:
        arr = np.deg2rad(arr)

    return [torch.tensor(a, dtype=dtype, device=device) for a in arr]


def _prepare_logging(host, log_dir: str = "runs/walk_exp1", resume: bool = True) -> None:
    """初始化/恢复日志目录与计数器到 host（self），并用“训练+评估的并集表头”初始化 CSV。"""
    host.log_dir = Path(log_dir)
    host.log_dir.mkdir(parents=True, exist_ok=True)
    host.metrics_csv = host.log_dir / "metrics.csv"
    host.meta_json   = host.log_dir / "meta.json"

    # 恢复计数器
    host.global_step = getattr(host, "global_step", 0)
    host.ep_offset   = getattr(host, "ep_offset", 0)
    host.best_reward = getattr(host, "best_reward", float("-inf"))

    if resume and host.meta_json.exists():
        meta = json.loads(host.meta_json.read_text(encoding="utf-8"))
        host.global_step = int(meta.get("global_step", host.global_step))
        host.ep_offset   = int(meta.get("ep_offset", host.ep_offset))
        host.best_reward = float(meta.get("best_reward", host.best_reward))

    # 初始化并集表头（统一使用 timestamp）
    if not host.metrics_csv.exists():
        header = [
            # —— 通用/索引字段
            "global_step", "episode", "type", "timestamp",
            # —— 会话 / 事件字段
            "event_name", "event_value", "seed", "path", "reason", "x_value",
            "num_episodes", "train_every_episodes", "min_buffer_for_train", "resume_logs",
            "bc_reset_on_eval", "window_return_sum", "window_size",
            # —— 训练相关
            "actor_loss", "critic_loss", "reward", "avg_recent_reward", "fell", "seq_len",
            "buffer_len", "total_falls", "episodes_in_span",
            "actor_action_var", "actor_action_diff",
            "td_error_mean", "td_error_std", "td_mse",
            "q_current_mean", "q_target_mean", "bellman_gap",
            "actor_grad_norm", "critic_grad_norm_mean", "actor_lr", "critic_lr", "q_values",
            # —— 评估相关（学习曲线）
            "eval_return_mean", "eval_return_sem", "eval_return_std",
            "eval_return_p25", "eval_return_median", "eval_return_p75",
            "eval_success_rate", "eval_ep_len_mean", "eval_episodes",
            # —— 备注
            "note",
        ]
        with host.metrics_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()


def _append_metric(self, row: Dict[str, Any], filename: str = "metrics.csv") -> None:
    """
    统一追加一行到 log_dir/metrics.csv。
    - 训练行：{type:'update', ...}
    - 评估行：{type:'eval',   eval_return_mean/sem/p25/p75/...}
    自动处理：
      * 若 CSV 不存在 -> 写表头
      * 若出现新列 -> 自动“升级表头”，重写旧行并补空，再追加新行
    """
    log_dir = getattr(self, "log_dir", None) or "."
    p = Path(log_dir) / filename
    p.parent.mkdir(parents=True, exist_ok=True)

    if "global_step" not in row and hasattr(self, "global_step"):
        row["global_step"] = int(getattr(self, "global_step"))
    row.setdefault("timestamp", int(time.time()))

    if not p.exists():
        fieldnames = list(row.keys())
        with p.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerow(row)
        return

    with p.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        old_fieldnames: List[str] = r.fieldnames or []
        rows = list(r)

    new_fields = [k for k in row.keys() if k not in old_fieldnames]
    if not new_fields:
        with p.open("a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=old_fieldnames)
            w.writerow(row)
        return

    preferred_prefix = [
        "global_step", "episode", "type", "timestamp",
        "actor_loss", "critic_loss",
        "reward", "avg_recent_reward", "fell", "seq_len",
        # 评估优先
        "phase",
        "eval_group_n", "eval_keep_top_k",
        "eval_return_topk_sum",        # ★ 你的主评估指标
        "eval_return_sum", "eval_episodes", "eval_dropped_min",
        "bc_reset",
        # 旧版统计
        "eval_return_mean", "eval_return_sem", "eval_return_std",
        "eval_return_p25", "eval_return_median", "eval_return_p75",
        "eval_success_rate", "eval_ep_len_mean",
        # 训练状态
        "buffer_len", "total_falls", "episodes_in_span",
        # 收敛监测
        "actor_action_var", "actor_action_diff",
        "td_error_mean", "td_error_std", "td_mse",
        "q_current_mean", "q_target_mean", "bellman_gap",
        "actor_grad_norm", "critic_grad_norm_mean",
        "actor_lr", "critic_lr",
        "q_values",
        "note",
    ]

    merged, seen = [], set()
    for k in preferred_prefix + old_fieldnames + list(row.keys()):
        if k not in seen:
            merged.append(k); seen.add(k)

    tmp = p.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=merged)
        w.writeheader()
        for old in rows:
            w.writerow({k: old.get(k, "") for k in merged})
        w.writerow({k: row.get(k, "") for k in merged})
    tmp.replace(p)


def _save_meta(host) -> None:
    """保存 host 的 meta 信息到文件。"""
    meta = {
        "global_step": int(getattr(host, "global_step", 0)),
        "ep_offset":   int(getattr(host, "ep_offset", 0)),
        "best_reward": float(getattr(host, "best_reward", float("-inf"))),
    }
    host.meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def log_session_event(
    host,
    *,
    event_name: str,
    event_value: Any = None,
    episode: Optional[int] = None,
    global_step: Optional[int] = None,
    seed: Optional[int] = None,
    note: Optional[str] = None,
    **extra,
) -> None:
    row = {
        "type": "session_event",
        "event_name": event_name,
        "event_value": event_value,
        "episode": int(episode if episode is not None else getattr(host, "ep_offset", 0)),
        "global_step": int(global_step if global_step is not None else getattr(host, "global_step", 0)),
    }
    if seed is not None:
        row["seed"] = int(seed)
    if note is not None:
        row["note"] = str(note)
    row.update(extra)
    _append_metric(host, row)


def log_episode(
    host,
    *,
    episode: int,
    global_step: int,
    reward: float,
    fell: bool,
    seq_len: int,
    **extra,
) -> None:
    row = {
        "type": "episode",
        "episode": int(episode),
        "global_step": int(global_step),
        "reward": float(reward),
        "fell": bool(fell),
        "seq_len": int(seq_len),
    }
    row.update(extra)
    _append_metric(host, row)


def log_update(
    host,
    *,
    episode: int,
    global_step: int,
    actor_loss: float,
    critic_loss: float,
    buffer_len: int,
    total_falls: int,
    metrics: Optional[Dict[str, Any]] = None,
    episodes_in_span: Optional[int] = None,
    **extra,
) -> None:
    metrics = metrics or {}
    row = {
        "type": "update",
        "episode": int(episode),
        "global_step": int(global_step),
        "actor_loss": float(actor_loss),
        "critic_loss": float(critic_loss),
        "buffer_len": int(buffer_len),
        "total_falls": int(total_falls),
        "episodes_in_span": int(episodes_in_span) if episodes_in_span is not None else None,
        "actor_action_var": metrics.get("actor_action_var"),
        "actor_action_diff": metrics.get("actor_action_diff"),
        "td_error_mean": metrics.get("td_error_mean"),
        "td_error_std": metrics.get("td_error_std"),
        "td_mse": metrics.get("td_mse"),
        "q_current_mean": metrics.get("current_q_mean"),
        "q_target_mean": metrics.get("target_q_mean"),
        "bellman_gap": metrics.get("bellman_gap"),
        "actor_grad_norm": metrics.get("actor_grad_norm"),
        "critic_grad_norm_mean": metrics.get("critic_grad_norm_mean"),
        "actor_lr": metrics.get("actor_lr"),
        "critic_lr": metrics.get("critic_lr"),
        "q_values": json.dumps(metrics.get("q_values_sample", [])),
    }
    row.update(extra)
    _append_metric(host, row)


def log_checkpoint_event(
    host,
    *,
    event_name: str,
    path: str,
    episode: int,
    global_step: int,
    note: Optional[str] = None,
    **extra,
) -> None:
    row = {
        "type": "checkpoint_event",
        "event_name": event_name,
        "episode": int(episode),
        "global_step": int(global_step),
        "path": str(path),
    }
    if note is not None:
        row["note"] = str(note)
    row.update(extra)
    _append_metric(host, row)




def snapshot_ddpg(ddpg, *, global_step: int = 0, ep_offset: int = 0, extra_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    meta = {
        "global_step": int(global_step),
        "ep_offset": int(ep_offset),
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }
    if extra_meta:
        meta.update(extra_meta)

    actor_opt = getattr(ddpg, "actor_optimizer", None)
    critic_opt = getattr(ddpg, "critic_optimizer", None)

    return {
        "actor": copy.deepcopy(ddpg.actor.state_dict()),
        "critic": copy.deepcopy(ddpg.critic.state_dict()),
        "actor_target": copy.deepcopy(ddpg.actor_target.state_dict()),
        "critic_target": copy.deepcopy(ddpg.critic_target.state_dict()),
        "actor_opt": copy.deepcopy(actor_opt.state_dict()) if actor_opt is not None else None,
        "critic_opt": copy.deepcopy(critic_opt.state_dict()) if critic_opt is not None else None,
        "meta": meta,
    }


def save_ddpg_snapshot(ddpg, filepath: str, *, global_step: int = 0, ep_offset: int = 0, extra_meta: Optional[Dict[str, Any]] = None) -> str:
    snap = snapshot_ddpg(ddpg, global_step=global_step, ep_offset=ep_offset, extra_meta=extra_meta)
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    torch.save(snap, filepath)
    return filepath


def build_best_meta_path(best_model_path: str) -> str:
    return os.path.join(
        os.path.dirname(best_model_path),
        os.path.basename(best_model_path) + ".meta.json"
    )


def load_best_window_return_alltime(best_model_path: str, default: float = float("-inf")) -> float:
    meta_path = build_best_meta_path(best_model_path)
    try:
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            return float(meta.get("best_window_return_alltime", default))
    except Exception as e:
        print(f"[best][meta-load-warning] {e}")
    return float(default)


def save_best_window_meta(best_model_path: str, *, best_window_return_alltime: float, episode: int, global_step: int, window_size: int) -> str:
    meta_path = build_best_meta_path(best_model_path)
    Path(meta_path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "best_window_return_alltime": float(best_window_return_alltime),
        "best_model_path": best_model_path,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "episode": int(episode),
        "global_step": int(global_step),
        "window_size": int(window_size),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return meta_path

# =========================
# 模型保存/加载
# =========================

def save_checkpoint(host, filepath="check_points/ddpg_checkpoint.pth"):
    checkpoint = {
        "actor_state_dict": host.actor.state_dict(),
        "critic_state_dict": host.critic.state_dict(),
        "actor_target_state_dict": host.actor_target.state_dict(),
        "critic_target_state_dict": host.critic_target.state_dict(),
        "actor_optimizer": host.actor_optimizer.state_dict(),
        "critic_optimizer": host.critic_optimizer.state_dict(),
        "train_logs": getattr(host, "_train_logs", None),
    }
    torch.save(checkpoint, filepath)
    print(f"Saved DDPG checkpoint (with target networks) to {filepath}")


def load_checkpoint(
    host,
    filepath="check_points/ddpg_checkpoint.pth",
    actor_lr: float = 1e-6,
    critic_lr: float = 1e-6,
):
    if not os.path.exists(filepath):
        print(f" Checkpoint not found at {filepath}")
        return

    checkpoint = torch.load(filepath, map_location="cpu")
    name, _ = next(host.critic.named_parameters())
    print(f" Before load: critic parameter '{name}' first 5 vals =",
          host.critic.state_dict()[name].view(-1)[:5])

    host.actor.load_state_dict(checkpoint["actor_state_dict"])
    host.critic.load_state_dict(checkpoint["critic_state_dict"])
    host.actor.to(host.device)
    host.critic.to(host.device)

    if "actor_target_state_dict" in checkpoint:
        host.actor_target.load_state_dict(checkpoint["actor_target_state_dict"])
        host.actor_target.to(host.device)
        print(" Loaded actor_target from checkpoint.")
    else:
        print("actor_target not found in checkpoint.")

    if "critic_target_state_dict" in checkpoint:
        host.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
        host.critic_target.to(host.device)
        print(" Loaded critic_target from checkpoint.")
    else:
        print(" critic_target not found in checkpoint.")

    print(f"▶ After load: critic parameter '{name}' first 5 vals =",
          host.critic.state_dict()[name].view(-1)[:5])
    print(f"▶ After load: critic_target parameter '{name}' first 5 vals =",
          host.critic_target.state_dict()[name].view(-1)[:5])

    if "actor_optimizer" in checkpoint:
        host.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
    if "critic_optimizer" in checkpoint:
        host.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])

    for opt in (host.actor_optimizer, host.critic_optimizer):
        for state in opt.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(host.device)

    if actor_lr is not None:
        for pg in host.actor_optimizer.param_groups:
            pg["lr"] = actor_lr
        print(f" Actor optimizer learning rate set to {actor_lr}")
    if critic_lr is not None:
        for pg in host.critic_optimizer.param_groups:
            pg["lr"] = critic_lr
        print(f" Critic optimizer learning rate set to {critic_lr}")

    print(f" Loaded checkpoint from {filepath} onto {host.device}")

def load_actor_from_checkpoint(
    host,
    filepath="check_points/best_ddpg_model.pth",
    actor_lr: float = 1e-5,
):
    """
    只从checkpoint恢复actor相关（actor + actor_target + actor_optimizer）
    - host: 你的DDPG对象
    - filepath: checkpoint路径
    - actor_lr: 可选，恢复后强制设置actor优化器学习率
    """
    if not os.path.exists(filepath):
        print(f"[actor-load] Checkpoint not found at {filepath}")
        return

    checkpoint = torch.load(filepath, map_location="cpu")

    # —— 加载 actor 权重 ——
    if "actor_state_dict" in checkpoint:
        host.actor.load_state_dict(checkpoint["actor_state_dict"])
        print("[actor-load] loaded actor_state_dict.")
    elif "actor" in checkpoint:
        host.actor.load_state_dict(checkpoint["actor"])
        print("[actor-load] loaded actor (old key).")
    else:
        raise KeyError("Checkpoint missing actor_state_dict/actor key")

    host.actor.to(host.device)

    # —— 可选加载 actor_target ——
    if hasattr(host, "actor_target"):
        if "actor_target_state_dict" in checkpoint:
            host.actor_target.load_state_dict(checkpoint["actor_target_state_dict"])
            host.actor_target.to(host.device)
            print("[actor-load] loaded actor_target_state_dict.")
        elif "actor_target" in checkpoint:
            host.actor_target.load_state_dict(checkpoint["actor_target"])
            host.actor_target.to(host.device)
            print("[actor-load] loaded actor_target (old key).")
        else:
            print("[actor-load] actor_target not found in checkpoint.")

    # —— 可选加载 actor_optimizer ——
    if hasattr(host, "actor_optimizer") and host.actor_optimizer is not None:
        opt_sd = checkpoint.get("actor_optimizer") or checkpoint.get("actor_optimizer_state_dict")
        if opt_sd is not None:
            host.actor_optimizer.load_state_dict(opt_sd)
            # 把优化器 state 移到正确device
            for state in host.actor_optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(host.device)
            print("[actor-load] loaded actor_optimizer state.")
        if actor_lr is not None:
            for pg in host.actor_optimizer.param_groups:
                pg["lr"] = actor_lr
            print(f"[actor-load] Actor optimizer learning rate set to {actor_lr}")

    print(f"[actor-load] Loaded actor from {filepath} onto {host.device}")
def load_bc(
    host,
    filepath: str = "actor_bc.pth",
    *,
    reset_actor_optimizer: bool = True,   # 若接着做RL训练，建议True（保留参数位方便扩展）
    actor_lr: float = 1e-6,
    optimizer_cls=torch.optim.Adam
):
    if not os.path.exists(filepath):
        print(f" Behavior cloning model not found at {filepath}")
        return

    state_dict = torch.load(filepath, map_location="cpu")
    host.actor.load_state_dict(state_dict)
    host.actor_target.load_state_dict(state_dict)   # 同步 target actor
    host.actor.to(host.device)
    host.actor_target.to(host.device)
    print(f"Loaded behavior cloning model into actor and target actor from '{filepath}'")


# =========================
# 绘图：Q/R 趋势（含底部裁剪）
# =========================

def _save_line_chart(
    x: np.ndarray,
    yq: np.ndarray,
    yr: np.ndarray,
    out_main: str,
    out_latest: str,
    title: str,
    x_label: str,
    y_limits: Tuple[float, float],
    draw_zero_line: bool = True,
    outliers: Optional[Dict[str, List[int]]] = None
) -> None:
    """
    保存折线图。注意：
    - 不对 y 做任何 nan 填充；NaN 会自然产生“断线”
    - 只对 NaN 的索引做底部标注，避免同色重复点
    """
    x  = np.asarray(x, dtype=float)
    yq = np.asarray(yq, dtype=float)
    yr = np.asarray(yr, dtype=float)

    fig, ax = plt.subplots(figsize=(9.6, 5.4), dpi=120)
    line_q, = ax.plot(x, yq, label="Qμ (mean Q)", linewidth=1.8)
    line_r, = ax.plot(x, yr, label="Rμ (mean reward)", linewidth=1.8)

    if draw_zero_line:
        ax.axhline(0.0, linestyle="--", linewidth=1.0, alpha=0.6)

    ax.set_xlim(float(np.min(x)), float(np.max(x)))
    ax.set_ylim(*y_limits)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Mean value")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="lower left")

    if outliers:
        y_min = y_limits[0]
        if outliers.get("q_low_idx"):
            xi = x[np.asarray(outliers["q_low_idx"], dtype=int)]
            ax.scatter(xi, np.full_like(xi, y_min), s=12, marker="o",
                       color=line_q.get_color(), zorder=3, linewidths=0)
        if outliers.get("r_low_idx"):
            xi = x[np.asarray(outliers["r_low_idx"], dtype=int)]
            ax.scatter(xi, np.full_like(xi, y_min), s=12, marker="o",
                       color=line_r.get_color(), zorder=3, linewidths=0)

    fig.tight_layout()
    fig.savefig(out_main, bbox_inches="tight")
    fig.savefig(out_latest, bbox_inches="tight")
    plt.close(fig)


def plot_q_reward_over_updates_gs(
    log_dir,
    qmu_list,
    rmu_list,
    global_step: int,
    base_filename: str = "summary_q_reward",
    title: str = "Qμ & Rμ over updates",
    last_n: Optional[int] = None,
    lower_quantile: float = 0.01,
    annotate_low_outliers: bool = True
) -> Tuple[str, str]:
    n = min(len(qmu_list), len(rmu_list))
    plots_dir = Path(log_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(base_filename).stem
    suf  = ".png"
    out_gs = str(plots_dir / f"{stem}_gs{int(global_step):06d}{suf}")
    out_latest = str(plots_dir / f"{stem}_latest{suf}")

    if n == 0:
        print(f"[plot] nothing to plot: q={len(qmu_list)}, r={len(rmu_list)}")
        return out_gs, out_latest

    start = max(0, n - last_n) if last_n is not None else 0
    yq = np.asarray(qmu_list[start:n], dtype=float)
    yr = np.asarray(rmu_list[start:n], dtype=float)
    x  = np.arange(1, n + 1)[start:]

    allv = np.concatenate([yq, yr], axis=0)
    lo = np.quantile(allv, lower_quantile)
    hi = float(np.max(allv))
    pad = 0.05 * (hi - lo + 1e-9)
    ylim = (lo - pad, hi + pad)

    outliers = None
    if annotate_low_outliers:
        q_low_idx = [i for i, v in enumerate(yq) if v < ylim[0]]
        r_low_idx = [i for i, v in enumerate(yr) if v < ylim[0]]
        outliers = {"q_low_idx": q_low_idx, "q_high_idx": [],
                    "r_low_idx": r_low_idx, "r_high_idx": []}

    _save_line_chart(
        x, yq, yr, out_main=out_gs, out_latest=out_latest,
        title=title, x_label="Update step (train() call index)",
        y_limits=ylim, draw_zero_line=True, outliers=outliers
    )
    return out_gs, out_latest


def plot_q_reward_from_csv_gs(
    log_dir,
    global_step: int,
    metrics_name: str = "metrics.csv",
    base_filename: str = "summary_q_reward_from_csv",
    title: str = "Qμ & Rμ over updates",
    last_n: Optional[int] = None,
    lower_quantile: float = 0.01,
    annotate_low_outliers: bool = True
) -> Tuple[str, str]:
    """
    从 CSV 解析 Qμ/Rμ，按下分位阈值将“底部极值”设为 NaN（主线断开），
    仅对这些 NaN 点做底部同色小点标注，避免同色重复。
    """
    csv_path = Path(log_dir) / metrics_name
    plots_dir = Path(log_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(base_filename).stem
    suf = ".png"
    out_gs = str(plots_dir / f"{stem}_gs{int(global_step):06d}{suf}")
    out_latest = str(plots_dir / f"{stem}_latest{suf}")

    if not csv_path.exists():
        print(f"[plot] metrics.csv not found: {csv_path}")
        return out_gs, out_latest

    pat_q = re.compile(r"Qμ=([-\d\.eE]+)")
    pat_r = re.compile(r"Rμ=([-\d\.eE]+)")
    q_vals, r_vals = [], []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            note = (row.get("note") or "")
            mq, mr = pat_q.search(note), pat_r.search(note)
            if mq and mr:
                try:
                    q_vals.append(float(mq.group(1)))
                    r_vals.append(float(mr.group(1)))
                except ValueError:
                    pass

    n = min(len(q_vals), len(r_vals))
    if n == 0:
        print(f"[plot] no Qμ/Rμ parsed from {csv_path}")
        return out_gs, out_latest

    start = max(0, n - last_n) if last_n is not None else 0
    yq = np.asarray(q_vals[start:n], dtype=float)
    yr = np.asarray(r_vals[start:n], dtype=float)
    x  = np.arange(1, n + 1, dtype=float)[start:]

    # 阈值（仅下分位），包含等于阈值的点；处理并列最小值
    allv = np.concatenate([yq, yr], axis=0)
    lo_thr = np.quantile(allv, lower_quantile)
    floor  = np.nextafter(float(lo_thr), np.inf)
    hi     = float(np.max(allv))

    yq_plot = yq.copy()
    yr_plot = yr.copy()
    q_low_mask = yq_plot <= floor
    r_low_mask = yr_plot <= floor
    yq_plot[q_low_mask] = np.nan
    yr_plot[r_low_mask] = np.nan

    kept = np.concatenate(
        [yq_plot[~np.isnan(yq_plot)], yr_plot[~np.isnan(yr_plot)]], axis=0
    )
    lo_for_axis = float(np.min(kept)) if kept.size > 0 else float(lo_thr)
    pad = 0.05 * (hi - lo_for_axis + 1e-9)
    ylim = (lo_for_axis - pad, hi + pad)

    outliers = None
    if annotate_low_outliers:
        q_low_idx = np.flatnonzero(np.isnan(yq_plot)).tolist()
        r_low_idx = np.flatnonzero(np.isnan(yr_plot)).tolist()
        outliers = {"q_low_idx": q_low_idx, "q_high_idx": [],
                    "r_low_idx": r_low_idx, "r_high_idx": []}

    _save_line_chart(
        x, yq_plot, yr_plot, out_main=out_gs, out_latest=out_latest,
        title=title, x_label="Update step (train() call index)",
        y_limits=ylim, draw_zero_line=True, outliers=outliers
    )
    return out_gs, out_latest


# =========================
# 评估学习曲线（独立绘图）
# =========================

def plot_eval_trend_from_csv(
    csv_path: str,
    out_path: str = "eval_trend.png",
    *,
    step_cols: Iterable[str] = ("step", "global_step", "update_idx", "update_index"),
    mean_cols: Iterable[str] = ("eval_return_mean", "eval_return", "eval", "eval/return_mean", "eval/return"),
    sem_cols:  Iterable[str] = ("eval_return_sem", "eval/sem", "eval_return_stderr"),
    std_cols:  Iterable[str] = ("eval_return_std", "eval/std"),
    n_cols:    Iterable[str] = ("eval_episodes", "eval_n", "eval/episodes", "eval_count"),  # 加入 eval_episodes
    p25_cols:  Iterable[str] = ("eval_return_p25", "eval/p25"),
    p75_cols:  Iterable[str] = ("eval_return_p75", "eval/p75"),
    ema_alpha: float = 0.9,
    ci_k: float = 1.96,
    title: str = "Deterministic evaluation over updates",
    x_label: str = "Update step",
    y_label: str = "Evaluation return",
    use_quantile_if_available: bool = True,
    show: bool = False
) -> str:
    """
    从 CSV 读取评估均值，绘制：均值折线 + 误差带（SEM或分位数）+ EMA(α=ema_alpha)。
    返回保存的图片路径 out_path。
    """
    def pick_first(df: pd.DataFrame, names: Iterable[str]) -> Optional[str]:
        for n in names:
            if n in df.columns:
                return n
        return None

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    step_col = pick_first(df, step_cols)
    mean_col = pick_first(df, mean_cols)
    if mean_col is None:
        raise ValueError(f"找不到评估均值列（候选: {list(mean_cols)}）")

    x = pd.to_numeric(df[step_col], errors="coerce").values if step_col else (np.arange(len(df)) + 1)
    y = pd.to_numeric(df[mean_col], errors="coerce").values

    lower = upper = None
    used_band = None
    if use_quantile_if_available:
        p25_col = pick_first(df, p25_cols)
        p75_col = pick_first(df, p75_cols)
        if p25_col and p75_col:
            lower = pd.to_numeric(df[p25_col], errors="coerce").values
            upper = pd.to_numeric(df[p75_col], errors="coerce").values
            used_band = "quantile (p25–p75)"

    if lower is None or upper is None:
        sem_col = pick_first(df, sem_cols)
        if sem_col:
            sem = pd.to_numeric(df[sem_col], errors="coerce").values
        else:
            std_col = pick_first(df, std_cols)
            n_col   = pick_first(df, n_cols)
            sem = None
            if std_col and n_col:
                std = pd.to_numeric(df[std_col], errors="coerce").values
                n   = pd.to_numeric(df[n_col], errors="coerce").values
                with np.errstate(divide="ignore", invalid="ignore"):
                    sem = std / np.sqrt(np.clip(n, 1, None))
        if sem is not None:
            lower = y - ci_k * sem
            upper = y + ci_k * sem
            used_band = f"SEM band (±{ci_k}·SEM)"

    valid = np.isfinite(x) & np.isfinite(y)
    if lower is not None and upper is not None:
        valid &= np.isfinite(lower) & np.isfinite(upper)

    x_plot = x[valid]
    y_plot = y[valid]
    lower_plot = lower[valid] if lower is not None else None
    upper_plot = upper[valid] if upper is not None else None

    y_series = pd.Series(y_plot)
    y_ema = y_series.ewm(alpha=ema_alpha, adjust=False).mean().values

    plt.figure(figsize=(10, 4.5), dpi=120)
    plt.plot(x_plot, y_plot, label="Eval mean", linewidth=1.2)
    if lower_plot is not None and upper_plot is not None:
        plt.fill_between(x_plot, lower_plot, upper_plot, alpha=0.2,
                         label=used_band if used_band else "Uncertain band")
    plt.plot(x_plot, y_ema, label=f"EMA (α={ema_alpha})", linewidth=1.8)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    if show: plt.show()
    plt.close()
    return out_path


# =========================
# 小工具（多个图会复用）
# =========================

def _ema(series: pd.Series, alpha=0.9) -> pd.Series:
    return series.ewm(alpha=alpha, adjust=False).mean()

def _rolling_std(series: pd.Series, window=50) -> pd.Series:
    return series.rolling(window, min_periods=max(5, window//5)).std()

def _pick_first(df: pd.DataFrame, names: Iterable[str]) -> Optional[str]:
    for n in names:
        if n in df.columns:
            return n
    return None

def _parse_from_note(df: pd.DataFrame, regexes: Iterable[str]) -> Optional[pd.Series]:
    if "note" not in df.columns:
        return None
    note = df["note"].fillna("").astype(str)
    for rgx in regexes:
        pat = re.compile(rgx)
        vals = []
        for s in note:
            m = pat.search(s)
            if m:
                try: vals.append(float(m.group(1)))
                except ValueError: vals.append(np.nan)
            else:
                vals.append(np.nan)
        ser = pd.Series(vals, index=df.index)
        if ser.notna().any():
            return ser
    return None

def _bottom_trim(s: pd.Series, q=0.05) -> pd.Series:
    """仅裁掉底部 q 分位以下的值（含等于阈值），避免y轴下界被拖拽。"""
    base = s.dropna().values
    if base.size == 0:
        return s.copy()
    lo = np.quantile(base, q)
    floor = np.nextafter(float(lo), np.inf)
    return s.mask(s <= floor, np.nan)

def _lastk_slope(xv: np.ndarray, yv: np.ndarray, k=100):
    mask = np.isfinite(xv) & np.isfinite(yv)
    xv, yv = xv[mask], yv[mask]
    if xv.size < 5:
        return np.nan, np.nan
    if xv.size > k:
        xv, yv = xv[-k:], yv[-k:]
    X = np.vstack([xv, np.ones_like(xv)]).T
    beta, _, _, _ = np.linalg.lstsq(X, yv, rcond=None)
    pred = X @ beta
    ss_res = float(np.sum((yv - pred) ** 2))
    ss_tot = float(np.sum((yv - np.mean(yv)) ** 2)) if yv.size > 1 else np.nan
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return float(beta[0]), r2

def _std_ratio_first_last(y: pd.Series, k=100) -> float:
    v = y[np.isfinite(y.values)]
    if v.size < 20:
        return np.nan
    k = min(k, v.size // 2)
    if k < 5:
        return np.nan
    first = float(pd.Series(v[:k]).std(ddof=0))
    last  = float(pd.Series(v[-k:]).std(ddof=0))
    return (last / first) if first > 0 else np.nan

def _cross_corr(a: np.ndarray, b: np.ndarray, maxlag=50):
    """返回 (lags, corr)，正滞后=Q 领先 R。"""
    a = np.asarray(a); b = np.asarray(b)
    am = np.nanmean(a); bm = np.nanmean(b)
    a = np.where(np.isfinite(a), a, am)
    b = np.where(np.isfinite(b), b, bm)
    if len(a) < 3 or len(b) < 3:
        return np.array([0]), np.array([np.nan])
    maxlag = min(maxlag, len(a) // 3) if len(a) > 30 else min(10, len(a) - 1)
    lags = np.arange(-maxlag, maxlag + 1)
    ccs = []
    for k in lags:
        if k < 0:
            cc = np.corrcoef(a[:k], b[-k:])[0, 1]
        elif k > 0:
            cc = np.corrcoef(a[k:], b[:-k])[0, 1]
        else:
            cc = np.corrcoef(a, b)[0, 1]
        ccs.append(cc)
    return lags, np.array(ccs)


# =========================
# 论文风格：从 metrics.csv 直接画评估学习曲线（筛选 type='eval'）
# =========================

BandMode = Literal["auto", "sem", "quantile", "std"]

def plot_eval_curve_from_metrics_csv(
    metrics_csv: str,
    out_png: str,
    *,
    ema_alpha: float = 0.9,
    band: BandMode = "auto",
    last_n: Optional[int] = None,
    title: str = "Evaluation learning curve (noise-free)",
) -> Optional[str]:
    """
    从 metrics.csv 中筛选评估行(type=='eval')，绘制学习曲线：
      - 折线：eval_return_mean
      - 不确定度带：优先 ±SEM；若无，则 p25~p75；再无，则 ±STD；再不行尝试用 STD/√n 估计
      - 趋势：EMA(α)
    返回：输出图片路径；若数据不足返回 None
    """
    p = Path(metrics_csv)
    if not p.exists():
        print(f"[plot] file not found: {p}")
        return None

    df = pd.read_csv(p)
    df.columns = [c.strip() for c in df.columns]

    if "type" in df.columns:
        df = df[df["type"].astype(str).str.lower() == "eval"].copy()
        if df.empty:
            print("[plot] no eval rows in CSV (type=='eval').")
            return None

    step_col = _pick_first(df, ["global_step", "step", "update_idx", "update_index"])
    x = pd.to_numeric(df[step_col], errors="coerce").values if step_col else (np.arange(len(df)) + 1)

    y_col = _pick_first(df, ["eval_return_mean"])
    if y_col is None:
        if "note" in df.columns:
            pat = re.compile(r"Rμ=([-\d\.eE]+)")
            vals = []
            for s in df["note"].fillna("").astype(str):
                m = pat.search(s); vals.append(float(m.group(1)) if m else np.nan)
            df["Rmu_parsed"] = vals
            y_col = "Rmu_parsed"
        else:
            y_col = _pick_first(df, ["Rμ","Rmu","R_mean","reward_mean","avg_recent_reward","reward"])
    if y_col is None:
        print("[plot] cannot find a usable mean column (eval_return_mean / Rμ / reward_mean).")
        return None

    y = pd.to_numeric(df[y_col], errors="coerce").values

    if last_n is not None and last_n > 0:
        x = x[-last_n:]; y = y[-last_n:]
        df = df.tail(last_n).copy()

    sem = pd.to_numeric(df.get("eval_return_sem", np.nan), errors="coerce").values
    std = pd.to_numeric(df.get("eval_return_std", np.nan), errors="coerce").values
    q25 = pd.to_numeric(df.get("eval_return_p25", np.nan), errors="coerce").values
    q75 = pd.to_numeric(df.get("eval_return_p75", np.nan), errors="coerce").values
    n_ep= pd.to_numeric(df.get("eval_episodes", np.nan), errors="coerce").values

    def has(arr) -> bool: return np.isfinite(arr).any()

    if band == "auto":
        if has(sem): band = "sem"
        elif has(q25) and has(q75): band = "quantile"
        elif has(std): band = "std"
        else:
            if has(std) and has(n_ep):
                sem = std / np.maximum(np.sqrt(np.maximum(n_ep, 1)), 1.0)
                band = "sem"
            else:
                band = "sem"  # 但无数据可画时会跳过

    y_series = pd.Series(y)
    y_ema = y_series.ewm(alpha=ema_alpha, adjust=False).mean().values

    y_low = y_high = None
    if band == "sem" and has(sem):
        y_low = y - sem; y_high = y + sem; band_label = "±1 SEM"
    elif band == "quantile" and has(q25) and has(q75):
        y_low = q25; y_high = q75; band_label = "p25–p75"
    elif band == "std" and has(std):
        y_low = y - std; y_high = y + std; band_label = "±1 STD"
    else:
        band_label = None

    allv = [y[np.isfinite(y)], y_ema[np.isfinite(y_ema)]]
    if y_low is not None and y_high is not None:
        allv += [y_low[np.isfinite(y_low)], y_high[np.isfinite(y_high)]]
    allv = np.concatenate([a for a in allv if a.size > 0]) if len(allv) else np.array([])
    ylim = None
    if allv.size >= 5:
        lo = np.quantile(allv, 0.01); hi = np.quantile(allv, 0.99)
        pad = 0.05 * (hi - lo + 1e-9); ylim = (lo - pad, hi + pad)

    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4.6), dpi=120)
    plt.plot(x, y, linewidth=1.2, label=f"{y_col}")
    if y_low is not None and y_high is not None and band_label:
        plt.fill_between(x, y_low, y_high, alpha=0.18, label=band_label)
    plt.plot(x, y_ema, linewidth=1.8, label=f"EMA (α={ema_alpha})")
    if ylim is not None: plt.ylim(*ylim)
    plt.xlabel("Update step")
    plt.ylabel("Eval return (noise-free)")
    plt.title(title)
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[plot] saved -> {out_path}")
    return str(out_path)


# =========================
# 无噪声收敛分析仪表盘（可选）
# =========================

def plot_no_noise_convergence(
    csv_path: str,
    out_dir: str = ".",
    *,
    ema_alpha: float = 0.9,
    std_window: int = 50,
    lower_quantile_trim: float = 0.05,
    max_lag: int = 50,
) -> Dict[str, str]:
    """
    基于“无噪声”的训练日志，输出 4 张图 + 汇总CSV + 文字诊断。
    仅依赖 Rμ（回报均值）、Qμ（价值均值），critic_loss 可选。
    返回: {名字: 文件路径}
    """
    out: Dict[str, str] = {}
    out_path = Path(out_dir); out_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    step_col = _pick_first(df, ("step", "global_step", "update_idx", "update_index"))
    x = pd.to_numeric(df[step_col], errors="coerce").values if step_col else (np.arange(len(df)) + 1)

    r_col = _pick_first(df, ("Rμ", "Rmu", "R_mu", "R_mean", "mean_reward", "reward_mean", "R", "rmu"))
    if r_col is None:
        r_ser = _parse_from_note(df, (r"Rμ=([-\d\.eE]+)", r"Rmu=([-\d\.eE]+)"))
        if r_ser is not None:
            df["Rmu_parsed"] = r_ser; r_col = "Rmu_parsed"
    r = pd.to_numeric(df[r_col], errors="coerce") if r_col else pd.Series([np.nan] * len(df))

    q_col = _pick_first(df, ("Qμ", "Qmu", "Q_mu", "Q_mean", "mean_q", "q_mean", "Q", "qmu"))
    if q_col is None:
        q_ser = _parse_from_note(df, (r"Qμ=([-\d\.eE]+)", r"Qmu=([-\d\.eE]+)"))
        if q_ser is not None:
            df["Qmu_parsed"] = q_ser; q_col = "Qmu_parsed"
    q = pd.to_numeric(df[q_col], errors="coerce") if q_col else pd.Series([np.nan] * len(df))

    loss_col = _pick_first(df, ("critic_loss", "train/critic_loss", "loss_critic"))
    crit = pd.to_numeric(df[loss_col], errors="coerce") if loss_col else None

    r_trim = _bottom_trim(r, lower_quantile_trim)
    q_trim = _bottom_trim(q, lower_quantile_trim)

    r_ema = _ema(r_trim, ema_alpha)
    q_ema = _ema(q_trim, ema_alpha)
    r_std = _rolling_std(r_trim, std_window)
    q_std = _rolling_std(q_trim, std_window)
    r_ema_d = r_ema.diff().abs()

    # 图1：R（无噪声）趋势
    plt.figure(figsize=(10, 4.5), dpi=120)
    plt.plot(x, r_trim, label="R (deterministic, trimmed)", linewidth=1.2)
    if np.isfinite(r_std).any():
        up = (r_ema + r_std).values; lo = (r_ema - r_std).values
        plt.fill_between(x, lo, up, alpha=0.15, label=f"±{std_window}-win std (around EMA)")
    plt.plot(x, r_ema, label=f"EMA (α={ema_alpha})", linewidth=1.8)
    plt.title("Deterministic return (R) over updates")
    plt.xlabel("Update step"); plt.ylabel("Return")
    plt.grid(True, linestyle=":"); plt.legend(); plt.tight_layout()
    out["plot_R_trend"] = str(out_path / "no_noise_R_trend.png")
    plt.savefig(out["plot_R_trend"]); plt.close()

    # 图2：Qμ 趋势
    plt.figure(figsize=(10, 4.5), dpi=120)
    plt.plot(x, q_trim, label="Qμ (trimmed)", linewidth=1.2)
    if np.isfinite(q_std).any():
        up = (q_ema + q_std).values; lo = (q_ema - q_std).values
        plt.fill_between(x, lo, up, alpha=0.15, label=f"±{std_window}-win std (around EMA)")
    plt.plot(x, q_ema, label=f"EMA (α={ema_alpha})", linewidth=1.8)
    plt.title("Qμ over updates")
    plt.xlabel("Update step"); plt.ylabel("Qμ")
    plt.grid(True, linestyle=":"); plt.legend(); plt.tight_layout()
    out["plot_Q_trend"] = str(out_path / "no_noise_Q_trend.png")
    plt.savefig(out["plot_Q_trend"]); plt.close()

    # 图3：平台检测（|ΔEMA(R)|）
    plt.figure(figsize=(10, 4.5), dpi=120)
    plt.plot(x, r_ema_d, label="|Δ EMA(R)| per update", linewidth=1.6)
    thr = float(np.nanpercentile(r_ema_d.values, 30))  # p30 作为阈值
    plt.axhline(thr, linestyle="--", linewidth=1.0, label=f"threshold ≈ p30={thr:.3g}")
    plt.title("Plateau detector: |Δ EMA(R)|")
    plt.xlabel("Update step"); plt.ylabel("|Δ EMA|")
    plt.grid(True, linestyle=":"); plt.legend(); plt.tight_layout()
    out["plot_R_ema_derivative"] = str(out_path / "no_noise_R_ema_derivative.png")
    plt.savefig(out["plot_R_ema_derivative"]); plt.close()

    # 图4：Q→R 互相关（正滞后=Q 领先）
    lags, ccs = _cross_corr(q_trim.values, r_trim.values, maxlag=max_lag)
    plt.figure(figsize=(10, 4.5), dpi=120)
    plt.plot(lags, ccs, linewidth=1.6, label="corr(Q lag k, R)")
    plt.axvline(0, linestyle="--", linewidth=1.0)
    plt.title("Cross-correlation Q→R (positive lag = Q leads)")
    plt.xlabel("Lag (updates)"); plt.ylabel("Correlation")
    plt.grid(True, linestyle=":"); plt.legend(); plt.tight_layout()
    out["plot_Q_to_R_xcorr"] = str(out_path / "no_noise_Q_to_R_xcorr.png")
    plt.savefig(out["plot_Q_to_R_xcorr"]); plt.close()

    # 汇总指标 & 文字诊断
    r_slope, r_r2 = _lastk_slope(x, r_trim.values, 100)
    q_slope, q_r2 = _lastk_slope(x, q_trim.values, 100)
    r_std_ratio   = _std_ratio_first_last(r_trim, 100)
    q_std_ratio   = _std_ratio_first_last(q_trim, 100)
    summary = pd.DataFrame({
        "metric": ["R (deterministic)", "Qμ"],
        "slope_last100": [r_slope, q_slope],
        "R2_last100": [r_r2, q_r2],
        "std_ratio_last/first": [r_std_ratio, q_std_ratio],
    })
    out["summary_csv"] = str(out_path / "no_noise_convergence_summary.csv")
    summary.to_csv(out["summary_csv"], index=False)

    flags = []
    def _near_zero(val, ref_std):
        if not np.isfinite(val): return False
        return abs(val) <= max(1e-3, ref_std * 0.01)

    if _near_zero(r_slope, float(pd.Series(r_trim).std(skipna=True))):
        flags.append("R 的最近100步趋势斜率接近 0")
    if np.isfinite(r_std_ratio) and r_std_ratio < 0.7:
        flags.append("R 的后段方差比前段收缩明显 (<0.7)")
    if np.nanmax(ccs) > 0 and lags[np.nanargmax(ccs)] > 0:
        flags.append("Q→R 的正滞后相关为峰，Q 对后续 R 的预测力较强")
    plateau_hits = int(np.sum(r_ema_d.values[-50:] <= thr)) if np.isfinite(thr) and len(r_ema_d) >= 50 else 0
    if plateau_hits >= 35:
        flags.append("最近 ~50 步有大量 |ΔEMA(R)| 低于阈值，出现平台迹象")

    diagnosis = "；".join(flags) if flags else "未见明确的收敛平台（或数据不足）。"
    out["diagnosis_txt"] = str(out_path / "no_noise_convergence_diagnosis.txt")
    Path(out["diagnosis_txt"]).write_text(diagnosis, encoding="utf-8")

    return out


def plot_eval_trend_compact(
    csv_path: str,
    out_path: str = "eval_trend.png",
    *,
    step_cols: Iterable[str] = ("global_step", "step", "update_idx", "update_index"),
    mean_cols: Iterable[str] = ("eval_return_mean", "eval_return", "eval/return_mean"),
    ema_alpha: float = 0.9,
    title: str = "Eval (span sum, compact view)",
    x_label: str = "Update step",
    y_label: str = "Eval value",
    mark_bc_reset: bool = True,
    bc_col: str = "bc_reset",
    mark_lows: bool = True,
    low_threshold: Optional[float] = None,
    low_quantile: float = 0.05,   # 低回报自动阈值分位数
    focus_low_q: float = 0.10,    # ★ 仅抬下限用
    figsize=(10, 4.6),
    show: bool = False,
) -> str:
    def pick_first(df: pd.DataFrame, names: Iterable[str]) -> Optional[str]:
        for n in names:
            if n in df.columns:
                return n
        return None

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    if "type" in df.columns:
        df = df[df["type"].astype(str).str.lower() == "eval"].copy()
        if df.empty:
            raise ValueError("CSV 中没有 type='eval' 的评估行。")

    step_col = pick_first(df, step_cols)
    y_col    = pick_first(df, mean_cols)
    if y_col is None:
        raise ValueError(f"找不到评估均值列（候选: {list(mean_cols)}）")

    x = pd.to_numeric(df[step_col], errors="coerce").values if step_col else (np.arange(len(df)) + 1)
    y = pd.to_numeric(df[y_col],    errors="coerce").values
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    df = df.loc[valid].reset_index(drop=True)

    y_ema = pd.Series(y).ewm(alpha=ema_alpha, adjust=False).mean().values

    # 低回报标记
    if mark_lows:
        thr = float(np.quantile(y, low_quantile)) if low_threshold is None else float(low_threshold)
        low_mask = y <= thr
    else:
        low_mask = np.zeros_like(y, dtype=bool)

    # ★ 仅抬下限，不截上限
    y_lo = float(np.quantile(y, focus_low_q))
    y_hi = float(np.max(y))
    pad_lo = 0.05 * (y_hi - y_lo + 1e-9)
    pad_hi = 0.05 * (y_hi - y_lo + 1e-9)
    ylim = (y_lo - pad_lo, y_hi + pad_hi)   # 上界保留最大值，+少量余量

    # 画图
    plt.figure(figsize=figsize, dpi=120)
    plt.plot(x, y, linewidth=1.2, label="Eval")
    plt.plot(x, y_ema, linewidth=1.8, label=f"EMA (α={ema_alpha})")

    if mark_lows and low_mask.any():
        plt.scatter(x[low_mask], y[low_mask], marker="v", s=48, label="Low (falls?)")

    # 连续 True 只标段首
    if mark_bc_reset and (bc_col in df.columns):
        bc_vals = df[bc_col].astype(int).values
        first_idxs, prev = [], 0
        for i, v in enumerate(bc_vals):
            if v == 1 and prev == 0:
                first_idxs.append(i)
            prev = v
        for i0 in first_idxs:
            plt.scatter([x[i0]], [y[i0]], s=70, marker="x", color="red")
            plt.annotate("BC reset", (x[i0], y[i0]),
                         textcoords="offset points", xytext=(8, 8), fontsize=9)

    plt.xlabel(x_label); plt.ylabel(y_label); plt.title(title)
    plt.ylim(*ylim); plt.grid(True, linestyle=":"); plt.legend()
    plt.tight_layout(); plt.savefig(out_path)
    if show: plt.show()
    plt.close()
    return out_path

# =========================
# Evaluation helpers
# =========================

def summarize_eval_results(results: List[Dict[str, Any]], target_x: Optional[float] = None) -> Dict[str, Any]:
    if not results:
        return {
            "num_runs": 0,
            "success_rate": 0.0,
            "return_mean": 0.0,
            "return_std": 0.0,
            "return_sem": 0.0,
            "return_p25": 0.0,
            "return_median": 0.0,
            "return_p75": 0.0,
            "ep_len_mean": 0.0,
            "forward_mean": 0.0,
            "forward_std": 0.0,
            "lateral_mean": 0.0,
            "lateral_std": 0.0,
            "reach_time_mean": None,
            "reach_time_std": None,
        }

    returns = np.asarray([float(r.get("episode_return", 0.0)) for r in results], dtype=float)
    forwards = np.asarray([float(r.get("forward", 0.0)) for r in results], dtype=float)
    laterals = np.asarray([abs(float(r.get("lateral", 0.0))) for r in results], dtype=float)
    steps = np.asarray([float(r.get("steps", 0.0)) for r in results], dtype=float)
    fells = np.asarray([1 if bool(r.get("fell", False)) else 0 for r in results], dtype=int)
    reaches = np.asarray([1 if bool(r.get("reached", False)) else 0 for r in results], dtype=int)
    success = reaches if target_x is not None else (1 - fells)

    valid_reach_time = [float(r["reach_time"]) for r in results if r.get("reach_time") is not None]

    n = len(results)
    return {
        "num_runs": int(n),
        "success_rate": float(success.mean()) if n else 0.0,
        "return_mean": float(np.mean(returns)) if n else 0.0,
        "return_std": float(np.std(returns, ddof=1)) if n > 1 else 0.0,
        "return_sem": float(np.std(returns, ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
        "return_p25": float(np.percentile(returns, 25)) if n else 0.0,
        "return_median": float(np.percentile(returns, 50)) if n else 0.0,
        "return_p75": float(np.percentile(returns, 75)) if n else 0.0,
        "ep_len_mean": float(np.mean(steps)) if n else 0.0,
        "forward_mean": float(np.mean(forwards)) if n else 0.0,
        "forward_std": float(np.std(forwards, ddof=1)) if n > 1 else 0.0,
        "lateral_mean": float(np.mean(laterals)) if n else 0.0,
        "lateral_std": float(np.std(laterals, ddof=1)) if n > 1 else 0.0,
        "reach_time_mean": float(np.mean(valid_reach_time)) if valid_reach_time else None,
        "reach_time_std": float(np.std(valid_reach_time, ddof=1)) if len(valid_reach_time) > 1 else 0.0 if valid_reach_time else None,
    }


def log_eval(
    host,
    *,
    episode: int,
    global_step: int,
    results: List[Dict[str, Any]],
    target_x: Optional[float] = None,
    note: Optional[str] = None,
    **extra,
) -> Dict[str, Any]:
    summary = summarize_eval_results(results, target_x=target_x)
    row = {
        "type": "eval",
        "episode": int(episode),
        "global_step": int(global_step),
        "eval_return_mean": summary["return_mean"],
        "eval_return_sem": summary["return_sem"],
        "eval_return_std": summary["return_std"],
        "eval_return_p25": summary["return_p25"],
        "eval_return_median": summary["return_median"],
        "eval_return_p75": summary["return_p75"],
        "eval_success_rate": summary["success_rate"],
        "eval_ep_len_mean": summary["ep_len_mean"],
        "eval_episodes": summary["num_runs"],
        "forward_mean": summary["forward_mean"],
        "forward_std": summary["forward_std"],
        "lateral_mean": summary["lateral_mean"],
        "lateral_std": summary["lateral_std"],
        "reach_time_mean": summary["reach_time_mean"],
        "reach_time_std": summary["reach_time_std"],
    }
    if note is not None:
        row["note"] = str(note)
    row.update(extra)
    _append_metric(host, row)
    return summary


# =========================
# Post-training eval checkpoint helpers
# =========================

def build_eval_ckpt_dir(seed: int, base_dir: str = "check_points") -> Path:
    p = Path(base_dir) / f"seed_{int(seed)}" / "eval_ckpts"
    p.mkdir(parents=True, exist_ok=True)
    return p


def extract_global_step_from_ckpt_name(filename: str) -> Optional[int]:
    m = re.search(r"ckpt_gs(\d+)\.pth$", str(filename))
    return int(m.group(1)) if m else None


def save_eval_candidate_checkpoint(host, *, seed: int, global_step: int,
                                   base_dir: str = "check_points") -> str:
    ckpt_dir = build_eval_ckpt_dir(seed=seed, base_dir=base_dir)
    ckpt_path = ckpt_dir / f"ckpt_gs{int(global_step):06d}.pth"
    save_checkpoint(host=host, filepath=str(ckpt_path))
    return str(ckpt_path)


def list_eval_candidate_checkpoints(seed: int, *,
                                    base_dir: str = "check_points",
                                    sort_ascending: bool = True) -> List[Path]:
    ckpt_dir = build_eval_ckpt_dir(seed=seed, base_dir=base_dir)
    ckpts = [p for p in ckpt_dir.glob("ckpt_gs*.pth") if extract_global_step_from_ckpt_name(p.name) is not None]
    ckpts.sort(key=lambda p: extract_global_step_from_ckpt_name(p.name) or -1, reverse=not sort_ascending)
    return ckpts
