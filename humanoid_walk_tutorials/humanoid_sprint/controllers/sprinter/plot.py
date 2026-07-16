# -*- coding: utf-8 -*-
# 建议：直接用这份替换你的 plot.py
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Dict, Iterable, Literal, Any
# 用你 helperfunction 里已经有的函数
from helperfunction import (
    plot_eval_curve_from_metrics_csv,   # 我们在 helperfunction 里已提供
    plot_no_noise_convergence, plot_eval_trend_compact        
)

# ========== 可配置 ==========
LOG_DIR      = "runs/walk_exp1"       # metrics.csv 所在目录
METRICS_CSV  = f"{LOG_DIR}/metrics.csv"
PLOTS_DIR    = f"{LOG_DIR}/plots"
EMA_ALPHA    = 0.9
LAST_N       = None                    # 只看最近 N 个评估点；None=全部
# ===========================

Path(PLOTS_DIR).mkdir(parents=True, exist_ok=True)

def _pick_first(df: pd.DataFrame, names):
    for n in names:
        if n in df.columns:
            return n
    return None

def _robust_ylim(*arrays):
    vals = [np.asarray(a)[np.isfinite(a)] for a in arrays if a is not None]
    vals = np.concatenate([v for v in vals if v.size > 0]) if vals else np.array([])
    if vals.size < 5:
        return None
    lo = np.quantile(vals, 0.01)
    hi = np.quantile(vals, 0.99)
    pad = 0.05 * (hi - lo + 1e-9)
    return (lo - pad, hi + pad)

# --------- 新增：loss 曲线（type='update'）---------
def plot_losses_from_metrics_csv(metrics_csv: str, out_png: str, ema_alpha: float = 0.9, last_n=None):
    p = Path(metrics_csv)
    if not p.exists():
        print(f"[loss-plot] file not found: {p}"); return None
    df = pd.read_csv(p)
    df.columns = [c.strip() for c in df.columns]

    # 只取训练更新行
    if "type" in df.columns:
        df = df[df["type"].astype(str).str.lower() == "update"].copy()
    if df.empty:
        print("[loss-plot] no update rows in metrics.csv"); return None

    step_col = _pick_first(df, ["global_step","step","update_idx","update_index"])
    x = pd.to_numeric(df[step_col], errors="coerce").values if step_col else (np.arange(len(df))+1)

    a = pd.to_numeric(df.get("actor_loss", np.nan), errors="coerce").values
    c = pd.to_numeric(df.get("critic_loss", np.nan), errors="coerce").values
    if not np.isfinite(a).any() and not np.isfinite(c).any():
        print("[loss-plot] actor_loss / critic_loss not found"); return None

    if last_n is not None and last_n > 0:
        x = x[-last_n:]; a = a[-last_n:]; c = c[-last_n:]

    a_ema = pd.Series(a).ewm(alpha=ema_alpha, adjust=False).mean().values if np.isfinite(a).any() else None
    c_ema = pd.Series(c).ewm(alpha=ema_alpha, adjust=False).mean().values if np.isfinite(c).any() else None

    ylim = _robust_ylim(a, c, a_ema, c_ema)

    plt.figure(figsize=(10, 4.2), dpi=120)
    if np.isfinite(a).any():
        plt.plot(x, a, linewidth=0.8, alpha=0.45, label="actor_loss (raw)")
        plt.plot(x, a_ema, linewidth=1.8, label=f"actor_loss EMA (α={ema_alpha})")
    if np.isfinite(c).any():
        plt.plot(x, c, linewidth=0.8, alpha=0.45, label="critic_loss (raw)")
        plt.plot(x, c_ema, linewidth=1.8, label=f"critic_loss EMA (α={ema_alpha})")
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Update step"); plt.ylabel("Loss")
    plt.title("Actor / Critic loss (training updates)")
    plt.grid(True, linestyle=":"); plt.legend(); plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png); plt.close()
    print(f"[loss-plot] saved -> {out_png}")
    return out_png

# --------- 新增：成功率与评估平均步长（type='eval'）---------
def plot_eval_aux_from_metrics_csv(metrics_csv: str, out_success_png: str, out_eplen_png: str, ema_alpha: float = 0.9, last_n=None):
    p = Path(metrics_csv)
    if not p.exists():
        print(f"[aux-plot] file not found: {p}"); return None
    df = pd.read_csv(p)
    df.columns = [c.strip() for c in df.columns]

    # 只取评估行
    if "type" in df.columns:
        df = df[df["type"].astype(str).str.lower() == "eval"].copy()
    if df.empty:
        print("[aux-plot] no eval rows in metrics.csv"); return None

    step_col = _pick_first(df, ["global_step","step","update_idx","update_index"])
    x = pd.to_numeric(df[step_col], errors="coerce").values if step_col else (np.arange(len(df))+1)
    succ = pd.to_numeric(df.get("eval_success_rate", np.nan), errors="coerce").values
    elen = pd.to_numeric(df.get("eval_ep_len_mean", np.nan), errors="coerce").values

    if last_n is not None and last_n > 0:
        x = x[-last_n:]; succ = succ[-last_n:]; elen = elen[-last_n:]

    # 成功率
    if np.isfinite(succ).any():
        succ_ema = pd.Series(succ).ewm(alpha=ema_alpha, adjust=False).mean().values
        ylim = _robust_ylim(succ, succ_ema)
        plt.figure(figsize=(10, 3.8), dpi=120)
        plt.plot(x, succ, linewidth=1.0, alpha=0.6, label="success_rate (raw)")
        plt.plot(x, succ_ema, linewidth=1.8, label=f"EMA (α={ema_alpha})")
        if ylim is not None: plt.ylim(*ylim)
        plt.xlabel("Update step"); plt.ylabel("Success rate")
        plt.title("Evaluation success rate")
        plt.grid(True, linestyle=":"); plt.legend(); plt.tight_layout()
        Path(out_success_png).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_success_png); plt.close()
        print(f"[aux-plot] saved -> {out_success_png}")

    # 平均步长
    if np.isfinite(elen).any():
        elen_ema = pd.Series(elen).ewm(alpha=ema_alpha, adjust=False).mean().values
        ylim = _robust_ylim(elen, elen_ema)
        plt.figure(figsize=(10, 3.8), dpi=120)
        plt.plot(x, elen, linewidth=1.0, alpha=0.6, label="ep_len_mean (raw)")
        plt.plot(x, elen_ema, linewidth=1.8, label=f"EMA (α={ema_alpha})")
        if ylim is not None: plt.ylim(*ylim)
        plt.xlabel("Update step"); plt.ylabel("Steps per episode")
        plt.title("Evaluation episode length (mean)")
        plt.grid(True, linestyle=":"); plt.legend(); plt.tight_layout()
        Path(out_eplen_png).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_eplen_png); plt.close()
        print(f"[aux-plot] saved -> {out_eplen_png}")

# === 新增：按每次update聚合 episode reward，取最大N-1，只画每5次update ===
def plot_eval_topk_from_episode_and_update(metrics_csv: str,
                                           out_png: str,
                                           train_every_episodes: int = 10,
                                           k_keep: int = None,
                                           sparse_every: int = 2,
                                           mode: str = "mean"):
    """
    - 从 type='episode' 行和 type='update' 行重算“取最大 N-1 个回报”的评估指标。
    - train_every_episodes: 每多少个 episode 训练一次（N）
    - k_keep: 手动指定取 top-k；默认为 N-1
    - sparse_every: 每多少次 update 画一个点
    - mode: "mean" 或 "sum"
    """
    df = pd.read_csv(metrics_csv)
    df.columns = [c.strip() for c in df.columns]

    df_ep = df[df["type"].astype(str).str.lower() == "episode"].copy()
    df_up = df[df["type"].astype(str).str.lower() == "update"].copy()

    if df_ep.empty or df_up.empty:
        print("[plot-topk] need both episode and update rows"); return None

    df_ep = df_ep.sort_values("episode")
    df_up = df_up.sort_values("global_step")

    N = train_every_episodes
    k_keep = k_keep or max(1, N-1)

    eval_steps = []
    eval_values = []

    last_ep_index = 0
    for update_idx, up_row in enumerate(df_up.itertuples()):
        if update_idx % sparse_every != 0:  # 稀疏化
            last_ep_index += N
            continue
        gs = up_row.global_step
        ep_slice = df_ep.iloc[last_ep_index:last_ep_index + N]  # 取这组episodes
        last_ep_index += N
        if ep_slice.empty:
            continue
        rewards = ep_slice["reward"].values
        k = min(k_keep, len(rewards))
        topk = np.sort(rewards)[-k:]
        val = topk.mean() if mode=="mean" else topk.sum()
        eval_steps.append(gs)
        eval_values.append(val)

    plt.figure(figsize=(10,4.5), dpi=160)
    plt.plot(eval_steps, eval_values, marker="o", linewidth=1.5, label=f"Top-{k_keep} {mode}")
    plt.xlabel("Update step"); plt.ylabel(f"Eval return ({mode})")
    plt.title(f"Evaluation by top-{k_keep} of {N} episodes (every {sparse_every} updates)")
    plt.grid(True, linestyle=":")
    plt.legend()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(out_png); plt.close()
    print(f"[plot-topk] saved -> {out_png}")
    return out_png

def _pick_step_col(df: pd.DataFrame):
    for c in ["global_step", "step", "update_idx", "update_index", "episode"]:
        if c in df.columns:
            return c
    df = df.copy()
    df["idx"] = np.arange(len(df)) + 1
    return "idx"

def _ema(y: pd.Series, alpha: float) -> pd.Series:
    return y.ewm(alpha=alpha, adjust=False).mean()

def plot_convergence_diagnostics(
    csv_path: str,
    out_dir: str = ".",
    *,
    use_ema: bool = True,
    ema_alpha: float = 0.2,           # 轻平滑，突出趋势
    sparse_every: Optional[int] = None, # 如 5/10：稀疏采样点，None 表示全量
    style_title_prefix: str = "Convergence diagnostics"
):
    """
    读取 metrics.csv（写有 update 行的新指标）并输出 3 张图：
      1) policy_stability.png : Actor 输出稳定性（动作方差 / 相邻差）
      2) td_error.png         : TD-error 的均值 / 方差
      3) bellman.png          : Q_current vs Q_target 以及它们的差（Bellman gap）

    依赖的列（缺失会自动跳过）：
      - actor_action_var, actor_action_diff
      - td_error_mean, td_error_std
      - q_current_mean, q_target_mean, bellman_gap
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    # 只用 update 行（训练步）；若没有 type 列，默认全用
    if "type" in df.columns:
        d = df[df["type"].astype(str).str.lower() == "update"].copy()
    else:
        d = df.copy()

    if d.empty:
        raise ValueError("No 'update' rows found in CSV; please log metrics at updates.")

    step_col = _pick_step_col(d)

    # 转数值
    num_cols = [
        step_col,
        "actor_action_var", "actor_action_diff",
        "td_error_mean", "td_error_std", "td_mse",
        "q_current_mean", "q_target_mean", "bellman_gap",
    ]
    for c in num_cols:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    # 稀疏采样
    if sparse_every is not None and sparse_every > 1:
        d = d.iloc[::sparse_every, :].copy()

    x = d[step_col].values

    # ===== 1) Policy stability =====
    has_var  = "actor_action_var"  in d.columns and d["actor_action_var"].notna().any()
    has_diff = "actor_action_diff" in d.columns and d["actor_action_diff"].notna().any()
    if has_var or has_diff:
        plt.figure(figsize=(10, 4.6), dpi=160)
        lines, labels = [], []

        if has_var:
            y = d["actor_action_var"]
            y_plot = _ema(y, ema_alpha) if use_ema else y
            l1, = plt.plot(x, y_plot.values, linewidth=1.8, label="Action variance")
            lines.append(l1); labels.append("Action variance")

        if has_diff:
            y = d["actor_action_diff"]
            y_plot = _ema(y, ema_alpha) if use_ema else y
            l2, = plt.plot(x, y_plot.values, linewidth=1.8, label="Action MSE vs. previous update")
            lines.append(l2); labels.append("Action MSE vs. prev")

        plt.xlabel("Update step")
        plt.ylabel("Magnitude")
        ttl = f"{style_title_prefix}: Policy output stability"
        if use_ema: ttl += f" (EMA α={ema_alpha})"
        plt.title(ttl)
        plt.grid(True, linestyle=":")
        plt.legend(lines, labels, loc="best")
        plt.tight_layout()
        out1 = out_dir / "policy_stability.png"
        plt.savefig(out1); plt.close()
        print(f"[saved] {out1}")
    else:
        print("[skip] policy_stability: missing 'actor_action_var'/'actor_action_diff'")

    # ===== 2) TD-error =====
    has_td_mean = "td_error_mean" in d.columns and d["td_error_mean"].notna().any()
    has_td_std  = "td_error_std"  in d.columns and d["td_error_std"].notna().any()
    if has_td_mean or has_td_std:
        fig, ax1 = plt.subplots(figsize=(10, 4.6), dpi=160)
        lines, labels = [], []

        if has_td_mean:
            y = d["td_error_mean"]
            y_plot = _ema(y, ema_alpha) if use_ema else y
            l1, = ax1.plot(x, y_plot.values, linewidth=1.8, label="TD-error mean")
            lines.append(l1); labels.append("TD-error mean")
        ax1.set_xlabel("Update step")
        ax1.set_ylabel("TD-error mean")
        ax1.grid(True, linestyle=":")

        if has_td_std:
            ax2 = ax1.twinx()
            y = d["td_error_std"]
            y_plot = _ema(y, ema_alpha) if use_ema else y
            l2, = ax2.plot(x, y_plot.values, linewidth=1.8, linestyle="--", label="TD-error std", color="tab:orange")
            lines.append(l2); labels.append("TD-error std")
            ax2.set_ylabel("TD-error std")

        ttl = f"{style_title_prefix}: TD-error"
        if use_ema: ttl += f" (EMA α={ema_alpha})"
        plt.title(ttl)

        # 合并图例
        ax1.legend(lines, labels, loc="best")
        plt.tight_layout()
        out2 = out_dir / "td_error.png"
        plt.savefig(out2); plt.close()
        print(f"[saved] {out2}")
    else:
        print("[skip] td_error: missing 'td_error_mean'/'td_error_std'")

    # ===== 3) Bellman consistency =====
    has_qc = "q_current_mean" in d.columns and d["q_current_mean"].notna().any()
    has_qt = "q_target_mean"  in d.columns and d["q_target_mean"].notna().any()
    has_gap= "bellman_gap"    in d.columns and d["bellman_gap"].notna().any()

    if has_qc or has_qt or has_gap:
        plt.figure(figsize=(10, 4.6), dpi=160)
        lines, labels = [], []

        if has_qc:
            y = d["q_current_mean"]
            y_plot = _ema(y, ema_alpha) if use_ema else y
            l1, = plt.plot(x, y_plot.values, linewidth=1.8, label="E[Q(s,a)] (current)")
            lines.append(l1); labels.append("E[Q] current")

        if has_qt:
            y = d["q_target_mean"]
            y_plot = _ema(y, ema_alpha) if use_ema else y
            l2, = plt.plot(x, y_plot.values, linewidth=1.8, label="E[r + γQ'(s',a')] (target)")
            lines.append(l2); labels.append("E[target]")

        if has_gap:
            y = d["bellman_gap"]
            y_plot = _ema(y, ema_alpha) if use_ema else y
            l3, = plt.plot(x, y_plot.values, linewidth=1.8, linestyle="--", label="Bellman gap")
            lines.append(l3); labels.append("Bellman gap")

        plt.xlabel("Update step")
        plt.ylabel("Value")
        ttl = f"{style_title_prefix}: Bellman consistency"
        if use_ema: ttl += f" (EMA α={ema_alpha})"
        plt.title(ttl)
        plt.grid(True, linestyle=":")
        plt.legend(lines, labels, loc="best")
        plt.tight_layout()
        out3 = out_dir / "bellman.png"
        plt.savefig(out3); plt.close()
        print(f"[saved] {out3}")
    else:
        print("[skip] bellman: missing 'q_current_mean'/'q_target_mean'/'bellman_gap'")

    return True
def main():
    """
    # 1) 学习曲线（评估均值 + 带 + EMA）
    plot_eval_curve_from_metrics_csv(
        metrics_csv=METRICS_CSV,
        out_png=f"{PLOTS_DIR}/eval_learning_curve.png",
        ema_alpha=EMA_ALPHA,
        band="auto",
        last_n=LAST_N,
        title="Evaluation learning curve (noise-free)"
    )



    # 3) 训练健康度：actor/critic loss
    plot_losses_from_metrics_csv(
        metrics_csv=METRICS_CSV,
        out_png=f"{PLOTS_DIR}/losses.png",
        ema_alpha=EMA_ALPHA,
        last_n=LAST_N
    )

    # 4) 成功率与评估平均步长
    plot_eval_aux_from_metrics_csv(
        metrics_csv=METRICS_CSV,
        out_success_png=f"{PLOTS_DIR}/eval_success_rate.png",
        out_eplen_png=f"{PLOTS_DIR}/eval_ep_len_mean.png",
        ema_alpha=EMA_ALPHA,
        last_n=LAST_N
    )
    """
# 路径按你的实际日志文件来
    csv_path = "runs/walk_exp1/metrics.csv"
    out_path = "runs/walk_exp1/plots"

    plot_eval_topk_from_episode_and_update(
        metrics_csv=csv_path,
        out_png="runs/walk_exp1/plots/eval_topk.png",
        train_every_episodes=10,  # 你的N
        sparse_every=1,           # 每5次update画点
        mode="mean"               # 取平均
    )
    # 假设 metrics.csv 在当前目录
    plot_convergence_diagnostics(
        csv_path=csv_path,
        out_dir=out_path,
        use_ema=True,      # 论文图建议开 EMA，曲线更稳
        ema_alpha=0.2,     # 平滑强度：0.1~0.3 较合适
        sparse_every=None  # 如果点太密，可设 5 或 10 稀疏化
    )

if __name__ == "__main__":
    main()
