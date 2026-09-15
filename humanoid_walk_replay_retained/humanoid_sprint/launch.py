"""Launch one configured Webots run: python launch.py train|eval [--check|--prepare-only]."""

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import uuid

from configs.runtime import (PROJECT_ROOT, SNAPSHOT_ENV, algorithm_settings, file_hash,
                             load_config, project_path, task_settings, validate)


def experiment_root(config):
    e = config["experiment"]
    return project_path(e["output_root"]) / e["name"] / f"seed_{e['seed']:03d}"


def checkpoints(config):
    v = config["evaluation"]
    items = []
    for path in (experiment_root(config) / "checkpoints/eval_candidates").glob("ckpt_gs*.pth"):
        match = re.fullmatch(r"ckpt_gs(\d+)\.pth", path.name)
        if match:
            step = int(match[1])
            # Use the checkpoint's own frozen training settings, never the live
            # experiment setting: old pure-Critic candidates remain on disk.
            sidecar = path.with_suffix('.config.json')
            if sidecar.exists():
                saved = json.loads(sidecar.read_text(encoding='utf-8'))
                warmup = saved['config']['training'].get('critic_warmup_updates', 0)
                count = saved.get('checkpoint_training_state', {}).get('actor_update_count')
                if count == 0 or (count is None and warmup > 0 and step <= warmup):
                    continue
            items.append((step, path))
    items.sort()
    if v["only_latest"]:
        items = items[-1:]
    return [p for step, p in items if step >= v["start_global_step"] and
            (v["selected_global_step"] is None or step == v["selected_global_step"])]


def resolve_config(config, mode):
    config = validate(copy.deepcopy(config))
    sources, warnings = [], []
    if mode == "eval":
        paths = checkpoints(config)
        if not paths:
            raise ValueError("No matching evaluation Checkpoints; check experiment/seed/selection")
        shared = None
        for path in paths:
            sidecar = path.with_suffix(".config.json")
            if not sidecar.exists():
                if not config["evaluation"]["allow_legacy_without_snapshot"]:
                    raise ValueError(f"No training snapshot: {path}")
                warnings.append(f"Legacy Checkpoint without snapshot: {path.name}; historical conditions unverified")
                sources.append(dict(checkpoint=str(path), sha256=file_hash(path), snapshot=None))
                continue
            saved = json.loads(sidecar.read_text(encoding="utf-8"))
            old = validate(saved["config"])
            signature = dict(environment=old["environment"], rollout_steps=old["training"]["rollout_steps"],
                             webots={k: old["webots"][k] for k in ("basic_time_step_ms", "control_time_step_ms", "initial_translation", "initial_rotation", "world_template")})
            # Template paths differ between sessions; compare the actual immutable content.
            signature["world_hash"] = file_hash(project_path(signature["webots"]["world_template"]))
            comparable = copy.deepcopy(signature)
            comparable["webots"].pop("world_template")
            if shared is not None and shared != comparable and config["evaluation"]["inherit_training_config"]:
                raise ValueError("Selected Checkpoints have different environments; evaluate them separately with selected_global_step")
            shared = comparable
            if config["evaluation"]["inherit_training_config"]:
                config["environment"] = signature["environment"]
                config["training"]["rollout_steps"] = signature["rollout_steps"]
                config["webots"].update(signature["webots"])
            sources.append(dict(checkpoint=str(path), sha256=file_hash(path), snapshot=str(sidecar)))
        if shared is not None and any(s["snapshot"] is None for s in sources):
            raise ValueError("Do not mix legacy Checkpoints and Checkpoints with snapshots in one evaluation selection")
    else:
        resume = experiment_root(config) / "checkpoints/resume.pth"
        policy = config["initialization"]["resume"]
        if policy == "required" and not resume.is_file():
            raise ValueError(f"Resume Checkpoint missing: {resume}")
        if policy == "never" and any((experiment_root(config) / "checkpoints").glob("**/*.pth")):
            raise ValueError("resume=never requires a new experiment name to avoid overwriting existing Checkpoints")
        if policy != "never" and resume.is_file():
            sources.append(dict(checkpoint=str(resume), sha256=file_hash(resume), role="resume"))
        init = config["initialization"]
        if init["use_bc"] or init["reset_to_bc_on_failure"]:
            bc = project_path(init["bc_checkpoint"])
            if not bc.is_file():
                raise ValueError(f"BC Checkpoint missing: {bc}")
            sources.append(dict(checkpoint=str(bc), sha256=file_hash(bc), role="bc"))
    for key in ("executable", "python", "world_template"):
        if not project_path(config["webots"][key]).is_file():
            raise ValueError(f"Missing webots.{key}: {config['webots'][key]}")
    return validate(config), sources, warnings


def absolute_resources(text, template):
    def replace(match):
        value = match[1]
        if value.startswith(("http:", "https:", "webots:", "file:")):
            return match[0]
        if Path(value).suffix.lower() in (".proto", ".png", ".jpg", ".jpeg", ".obj", ".stl"):
            return '"' + (template.parent / value).resolve().as_posix() + '"'
        return match[0]
    return re.sub(r'"([^"\n]+)"', replace, text)


def render_world(template, config, controller="configured_run"):
    text = absolute_resources(template.read_text(encoding="utf-8"), template)
    text, count = re.subn(r"(\bbasicTimeStep\s+)\S+", lambda m: m[1] + str(config["webots"]["basic_time_step_ms"]), text)
    if count != 1:
        raise ValueError("World template must contain exactly one basicTimeStep")
    match = re.search(r"DEF NAO Nao\s*\{[^{}]*\}", text)
    if not match:
        raise ValueError("Expected a flat DEF NAO Nao block in world template")
    block = match[0]
    for field, values in (("translation", config["webots"]["initial_translation"]), ("rotation", config["webots"]["initial_rotation"])):
        block, count = re.subn(rf"(?m)^(\s*{field}\s+)[^\r\n]+", lambda m: m[1] + " ".join(map(str, values)), block)
        if count != 1:
            raise ValueError(f"Missing/duplicate NAO {field}")
    block, count = re.subn(r'\bcontroller\s+"[^"]*"', 'controller "' + controller + '"', block)
    if count != 1:
        raise ValueError("Missing/duplicate NAO Controller")
    return text[:match.start()] + block + text[match.end():]


def validate_models(payload):
    controllers = PROJECT_ROOT / "controllers"
    sys.path[:0] = [str(controllers / "common"), str(controllers)]
    import torch
    from algorithms import load_algorithm
    from checkpointing import load_actor_weights
    from models import BoundedActor
    from tasks import load_task
    config = payload["config"]
    e, t = config["environment"], config["training"]
    task = load_task(e["task_module"], e["task_class"], task_settings(config, payload["mode"]))
    algorithm = None
    if payload["mode"] == "train":
        algorithm = load_algorithm(t["algorithm_module"], t["algorithm_class"], algorithm_settings(config, "cpu", task.task_id))
    for source in payload["sources"]:
        path = source["checkpoint"]
        if source.get("role") == "resume":
            algorithm.load_checkpoint(path)
        else:
            from configs.runtime import create_actor
            load_actor_weights(create_actor(config), path, torch.device("cpu"), expected_task_id=task.task_id)


def prepare_run(config, mode, destination=None):
    config, sources, warnings = resolve_config(config, mode)
    template = project_path(config["webots"]["world_template"])
    rendered = render_world(template, config)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid.uuid4().hex[:6]
    folder = Path(destination) if destination else experiment_root(config) / "runs" / (mode + "_" + run_id)
    folder.mkdir(parents=True, exist_ok=False)
    original = folder / "world_template.wbt"
    original.write_text(absolute_resources(template.read_text(encoding="utf-8"), template), encoding="utf-8")
    config["webots"]["world_template"] = str(original.resolve())
    payload = dict(schema_version=1, mode=mode, run_id=run_id, config=config, sources=sources, warnings=warnings,
                   source_world_sha256=file_hash(template), comparison_status="pending_evaluation", status="prepared")
    payload["reproducibility_protocol"] = ("seed_before_network_init_v1" if mode == "train"
                                           else "shared_webots_world_reload_v3")
    snapshot = folder / "effective_config.json"
    snapshot.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    # Webots/Robot Window resources must stay below Windows MAX_PATH even
    # when experiment names and evaluation batch directories are long.
    # Snapshots and results remain in the experiment; only runtime files move.
    world_dir = PROJECT_ROOT / ".webots_runs" / run_id / "worlds"
    world_dir.mkdir(parents=True)
    world = world_dir / "run.wbt"
    world.write_text(rendered, encoding="utf-8")
    controller_dir = world_dir.parent / "controllers/configured_run"
    controller_dir.mkdir(parents=True)
    entry = PROJECT_ROOT / "controllers" / ("sprinter/sprinter.py" if mode == "train" else "eval_controller/eval_controller.py")
    wrapper = ("import os, sys, runpy, json\nfrom pathlib import Path\n"
               f"os.environ[{SNAPSHOT_ENV!r}] = {str(snapshot.resolve())!r}\n"
               f"sys.path.insert(0, {str(entry.parent)!r})\n"
               f"os.chdir({str(entry.parent)!r})\n"
               f"status = Path({str(folder.resolve() / 'run_status.json')!r})\n"
               "status.write_text(json.dumps({'status': 'running'}), encoding='utf-8')\n"
               "try:\n"
               f"    runpy.run_path({str(entry)!r}, run_name='__main__')\n"
               "except BaseException as error:\n"
               "    status.write_text(json.dumps({'status': 'failed', 'error': str(error)}), encoding='utf-8')\n"
               "    raise\n"
               "else:\n"
               "    status.write_text(json.dumps({'status': 'completed'}), encoding='utf-8')\n")
    (controller_dir / "configured_run.py").write_text(wrapper, encoding="utf-8")
    (controller_dir / "runtime.ini").write_text("[python]\nCOMMAND = " + str(project_path(config["webots"]["python"])) + "\n", encoding="utf-8")
    # Preserve Robot Window resource lookup without changing any source project files.
    plugins = PROJECT_ROOT / "plugins"
    if plugins.exists():
        shutil.copytree(plugins, world_dir.parent / "plugins")
    return snapshot, world, payload


def prepare_evaluation(config):
    """Freeze each model once; load each world in one Webots application."""
    config, sources, warnings = resolve_config(config, "eval")
    batch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid.uuid4().hex[:6]
    folder = experiment_root(config) / "runs" / ("eval_batch_" + batch_id)
    folder.mkdir(parents=True, exist_ok=False)
    manifest = dict(batch_id=batch_id, protocol="shared_webots_world_reload_v3",
                    status="prepared", warnings=warnings, runs=[])
    for source in sources:
        child = copy.deepcopy(config)
        step = int(re.fullmatch(r"ckpt_gs(\d+)\.pth", Path(source["checkpoint"]).name)[1])
        child["evaluation"].update(selected_global_step=step, only_latest=False, episodes=1)
        snapshot, world, payload = prepare_run(child, "eval", destination=folder / f"gs{step:06d}")
        if payload["sources"] != [source]:
            raise RuntimeError("Checkpoint changed while preparing evaluation; prepare again")
        manifest["runs"].append(dict(global_step=step, snapshot=str(snapshot.resolve()),
                                     world=str(world.resolve()), checkpoint=source["checkpoint"],
                                     sha256=source["sha256"], status="prepared"))
    path = folder / "batch_status.json"
    for index, run in enumerate(manifest["runs"]):
        snapshot = Path(run["snapshot"])
        payload = json.loads(snapshot.read_text(encoding="utf-8"))
        payload.update(evaluation_batch=str(path.resolve()), evaluation_batch_index=index)
        snapshot.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def run_evaluation_batch(path):
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not manifest["runs"] or manifest["status"] != "prepared":
        raise RuntimeError("Expected a new, nonempty evaluation batch")
    try:
        for run in manifest["runs"]:
            if file_hash(run["checkpoint"]) != run["sha256"]:
                raise RuntimeError("Checkpoint changed since evaluation preparation")
        first = manifest["runs"][0]
        payload = json.loads(Path(first["snapshot"]).read_text(encoding="utf-8"))
        executable = project_path(payload["config"]["webots"]["executable"])
        environment = os.environ.copy()
        environment["PATH"] = str(executable.parent) + os.pathsep + environment.get("PATH", "")
        manifest["status"] = "running"
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        with (path.parent / "webots.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                [str(executable), "--batch", "--minimize", "--mode=fast", "--stdout", "--stderr", first["world"]],
                cwd=str(executable.parent), env=environment, stdout=log, stderr=subprocess.STDOUT)
            (path.parent / "webots_process.json").write_text(
                json.dumps(dict(webots_pid=process.pid, launches=1)), encoding="utf-8")
            returncode = process.wait()
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if returncode or manifest["status"] != "completed":
            raise RuntimeError(f"Evaluation interrupted or failed; inspect {path.parent / 'webots.log'}")
        for run in manifest["runs"]:
            result = json.loads((Path(run["snapshot"]).parent / "evaluation_result.json").read_text(encoding="utf-8"))
            payload = json.loads(Path(run["snapshot"]).read_text(encoding="utf-8"))
            if (run["status"] != "completed" or result["eval_run_id"] != payload["run_id"]
                    or result["checkpoint_sha256"] != run["sha256"]):
                raise RuntimeError("Incomplete or mismatched evaluation result")
    except BaseException as error:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest.update(status="failed", error=str(error))
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        raise
    print(f"Evaluation complete: {path}")


def print_summary(config, mode, sources, warnings):
    t, i = config["training"], config["initialization"]
    resume = any(s.get("role") == "resume" for s in sources)
    print(f"Mode: {mode}; experiment: {config['experiment']['name']}; seed: {config['experiment']['seed']}")
    print(f"Output: {experiment_root(config)}")
    print(f"Physics/control: {config['webots']['basic_time_step_ms']}/{config['webots']['control_time_step_ms']} ms")
    if mode == "train":
        print(f"Resume: {resume}; startup BC: {i['use_bc'] and (not resume or i['bc_on_resume'])}; failure BC reset: {i['reset_to_bc_on_failure']}")
        continuous = t['sampling_mode'] == 'continuous'
        cutoff = i.get('bc_reset_max_env_steps' if continuous else 'bc_reset_max_rollouts', 0)
        if i['reset_to_bc_on_failure'] and cutoff:
            print(f"BC recovery allowed through {'env step' if continuous else 'rollout'} {cutoff}; later resets preserve Actor and Replay Buffer")
        print(f"Effective initial LR: Actor={i['resume_actor_lr'] if resume else t['actor_lr']}, Critic={i['resume_critic_lr'] if resume else t['critic_lr']}")
        if t.get('lr_switch_env_steps', 0):
            print(f"Cumulative LR schedule overrides resume LR: steps 1..{t['lr_switch_env_steps']} "
                  f"Actor={t['actor_lr']}, Critic={t['critic_lr']}; later "
                  f"Actor={t['actor_lr_after_switch']}, Critic={t['critic_lr_after_switch']}")
        print(f"Training Gaussian Action noise: std={t['action_noise_std_rad']} rad (evaluation: disabled)")
        if t['noise_switch_env_steps']:
            print(f"Noise switches after env step {t['noise_switch_env_steps']} to std={t['action_noise_std_after_switch']} rad")
        print(f"Critic-only warmup: {t['critic_warmup_updates']} update rounds (within rollout budget)")
        if t['sampling_mode'] == 'continuous':
            print(f"Critic-only interval: {t['critic_warmup_update_every_steps']} env steps; joint interval: {t['update_every_steps']}")
        if continuous:
            print(f"Continuous interactions={t['total_env_steps']}; update every {t['update_every_steps']} steps; log every {t['log_every_steps']} steps; buffer threshold={t['min_buffer_for_train']}")
        else:
            print(f"Rollouts={t['rollouts']} x {t['rollout_steps']}; update interval={t['train_every_rollouts']}; warmup={t['min_buffer_for_train']}")
    print(f"Previous: {config['experiment']['previous_experiment']}; baseline: {config['experiment']['baseline_id']}")
    for source in sources:
        print(f"Model: {source['checkpoint']}")
    for warning in warnings:
        print(f"WARNING: {warning}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "eval"))
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true", help="Validate only; do not write a run or launch Webots")
    group.add_argument("--prepare-only", action="store_true", help="Generate a runnable world without launching")
    parser.add_argument("--validate-models", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.validate_models:
        validate_models(json.loads(Path(args.validate_models).read_text(encoding="utf-8")))
        return
    config, sources, warnings = resolve_config(load_config(), args.mode)
    print_summary(config, args.mode, sources, warnings)
    render_world(project_path(config["webots"]["world_template"]), config)
    with tempfile.TemporaryDirectory() as temporary:
        check = Path(temporary) / "models.json"
        check.write_text(json.dumps(dict(config=config, mode=args.mode, sources=sources)), encoding="utf-8")
        subprocess.run([str(project_path(config["webots"]["python"])), str(Path(__file__).resolve()), args.mode,
                        "--validate-models", str(check)], check=True)
    if args.check:
        print("Configuration, world template and model contracts validated; nothing launched.")
        return
    if args.mode == "eval":
        batch = prepare_evaluation(config)
        print(f"Evaluation manifest: {batch}")
        if not args.prepare_only:
            run_evaluation_batch(batch)
        return
    snapshot, world, payload = prepare_run(config, args.mode)
    print(f"Configuration snapshot: {snapshot}\nWorld: {world}")
    if not args.prepare_only:
        executable = project_path(payload["config"]["webots"]["executable"])
        environment = os.environ.copy()
        environment["PATH"] = str(executable.parent) + os.pathsep + environment.get("PATH", "")
        # World reloads must discard runtime world changes without a modal save prompt.
        subprocess.Popen([str(executable), "--batch", "--mode=fast", str(world.resolve())],
                         cwd=str(executable.parent), env=environment)


if __name__ == "__main__":
    main()
