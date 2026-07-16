# helperfunction_eval.py

import csv
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def prepare_eval_logging(host, log_dir: str) -> None:
    """
    为 eval controller 绑定日志目录。
    不重置已有 metrics.csv，只确保路径存在。
    """
    host.log_dir = Path(log_dir)
    host.log_dir.mkdir(parents=True, exist_ok=True)
    host.metrics_csv = host.log_dir / "metrics.csv"


def append_metric(host, row: Dict[str, Any], filename: str = "metrics.csv") -> None:
    """
    追加一行到 metrics.csv。
    若出现新列，自动扩展表头。
    """
    log_dir = getattr(host, "log_dir", None) or "."
    p = Path(log_dir) / filename
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
        old_fieldnames = r.fieldnames or []
        rows = list(r)

    new_fields = [k for k in row.keys() if k not in old_fieldnames]
    if not new_fields:
        with p.open("a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=old_fieldnames)
            w.writerow(row)
        return

    merged = []
    seen = set()
    for k in list(old_fieldnames) + list(row.keys()):
        if k not in seen:
            merged.append(k)
            seen.add(k)

    tmp = p.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=merged)
        w.writeheader()
        for old in rows:
            w.writerow({k: old.get(k, "") for k in merged})
        w.writerow({k: row.get(k, "") for k in merged})
    tmp.replace(p)


def extract_global_step_from_ckpt_name(filename: str) -> Optional[int]:
    m = re.search(r"ckpt_gs(\d+)\.pth$", filename)
    return int(m.group(1)) if m else None


def build_eval_ckpt_dir(seed: int, base_dir: str = "check_points") -> Path:
    return Path(base_dir) / f"seed_{seed}" / "eval_ckpts"


def load_existing_eval_steps(metrics_csv: Path) -> set:
    """
    从 metrics.csv 中读取已存在的 eval global_step，避免重复评估记录。
    """
    if not metrics_csv.exists():
        return set()

    done = set()
    try:
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