"""Prepare the logged prefix of a DDPG model before resuming a step checkpoint."""
import csv
import gzip
import json
from pathlib import Path
import os
import shutil


def copy_prefix(source, output, field, step):
    count = falls = 0
    previous = -1
    with source.open(encoding='utf-8-sig', newline='') as src, output.open('x', newline='', encoding='utf-8') as dst:
        rows = csv.DictReader(src)
        if not rows.fieldnames or field not in rows.fieldnames:
            raise ValueError(f'Missing {field} in {source}')
        writer = csv.DictWriter(dst, rows.fieldnames)
        writer.writeheader()
        for row in rows:
            index = int(row[field])
            if index > step:
                break
            if index <= previous:
                raise ValueError(f'Out-of-order {field} in {source}')
            previous = index
            writer.writerow(row)
            count += 1
            falls += row.get('fell', '').lower() in ('true', '1')
    return count, falls


def prepare_resume(model, session, checkpoint):
    origin = Path(checkpoint['path']).parent.parent
    step = checkpoint['step']
    staged = session/'staging'
    staged.mkdir()
    # Some model-only historical checkpoints precede their finalized CSV rows.
    counts = (0, 0)
    for name, field in [('training.csv', 'step'), ('training_episodes.csv', 'end_step')]:
        if (origin/name).is_file():
            summary = copy_prefix(origin/name, staged/name, field, step)
            if field == 'end_step':
                counts = summary
    trace = 'transition_trace.jsonl.gz'
    if (origin/trace).is_file():
        with gzip.open(origin/trace, 'rt', encoding='utf-8') as src, gzip.open(staged/trace, 'wt', encoding='utf-8') as dst:
            for line in src:
                index = int(json.loads(line)['step'])
                if index > step:
                    break
                dst.write(line)
                if index == step:
                    break
    shutil.copyfile(session/'effective_config.json', staged/'effective_config.json')
    (staged/'resume.json').write_text(json.dumps(dict(checkpoint=checkpoint['path'], step=step,
        completed_episodes=counts[0], falls=counts[1]), indent=2), encoding='utf-8')

    archive = session/'previous'
    archive.mkdir()
    old_names = {p.name for p in staged.iterdir()} | {
        'training.csv', 'training_episodes.csv', trace, 'world_reload_state.pth', 'result.json',
        'progress.json', 'status.json', 'training_failure.json', 'report_trace.jsonl', 'report_trace.jsonl.gz'}
    originals = [model/name for name in sorted(old_names) if (model/name).is_file()]
    originals += [p for p in (model/'checkpoints').glob('step_*.pth')
                  if p.stem[5:].isdigit() and int(p.stem[5:]) > step]
    originals += [p for p in (model/'checkpoints').glob('continuation_*.pth')
                  if p.stem[13:].isdigit() and int(p.stem[13:]) > step]
    moved, published = [], []
    try:
        for p in originals:
            backup = archive/p.relative_to(model)
            backup.parent.mkdir(parents=True, exist_ok=True)
            os.replace(p, backup)
            moved.append((p, backup))
        for p in staged.iterdir():
            target = model/p.name
            os.replace(p, target)
            published.append(target)
    except BaseException:
        for target in reversed(published):
            os.replace(target, staged/target.name)
        for target, backup in reversed(moved):
            os.replace(backup, target)
        raise
