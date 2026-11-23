"""Build the offline SAC-Isaac tracking report from finalized environment steps."""
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

AXES = ('vx', 'vy', 'wz')
WINDOW_SIZE = 1000
EARLY_STEPS = 25_000
LATE_STEPS = 10_000


@dataclass
class CommandSegment:
    start: int
    episode: int
    task: str
    command: tuple
    slot: int
    end: int = 0
    count: int = 0
    sums: list = field(default_factory=lambda: [0., 0., 0.])

    def add(self, step, actual):
        self.end = step
        self.count += 1
        for i, value in enumerate(actual):
            self.sums[i] += value

    def record(self):
        return [self.start, self.end, self.count, self.episode, self.task, *self.command,
                *(value / self.count for value in self.sums)]


@dataclass
class Window:
    count: int = 0
    errors: list = field(default_factory=lambda: [0., 0., 0.])
    planar: float = 0.
    falls: int = 0
    reward: float = 0.
    coverage: Counter = field(default_factory=Counter)

    def add(self, target, actual, fell, reward, label):
        errors = [abs(a - t) for a, t in zip(actual, target)]
        self.count += 1
        for i, error in enumerate(errors):
            self.errors[i] += error
        self.planar += math.hypot(errors[0], errors[1])
        self.reward += reward
        self.falls += fell
        self.coverage[label] += 1

    def record(self, start):
        return dict(start=start, end=start+self.count-1, n=self.count, falls=self.falls,
                    reward=self.reward/self.count, xy=self.planar/self.count,
                    coverage=self.coverage, **dict(zip(AXES, (e/self.count for e in self.errors))))


def coverage_label(task, target, random_forward, yaw_ramp):
    def interval(value, width):
        low = math.floor(round(value / width, 6)) * width
        return f'[{low:.2f}, {low+width:.2f})'
    x, y, yaw = target
    vx = interval(x, .01) if random_forward and task == 'walk' else f'{x:.3f}'
    wz = interval(yaw, .02) if yaw_ramp and task == 'turn' else f'{yaw:.3f}'
    return f'{task}   {vx} / {y:.3f} / {wz}'


def build_report_data(rows, cfg, total):
    training = cfg['training']
    walk = training.get('walk_commands', {})
    omni = training.get('omni_curriculum', {})
    random_forward = ('random_vx_range' in walk or
                      cfg.get('isaac', {}).get('experiment_variant') == 'walk_lateral_random_vx')
    yaw_ramp = 'yaw_ramp' in training or ('turn_ramp_steps' in omni and 'turn_speed_steps' not in omni)
    active, last_clock = {}, {}
    segments = defaultdict(list)
    windows = defaultdict(Window)
    early, late = Window(), Window()
    compare = total >= EARLY_STEPS + LATE_STEPS
    seen = bytearray(total)
    has_turn = False
    for row in rows:
        step = int(row['step'])
        if not 1 <= step <= total or seen[step-1]:
            raise ValueError(f'Invalid or duplicate global step {step}')
        seen[step-1] = 1
        robot = int(row.get('env_id', row.get('rank', 0)))
        clock = int(row.get('env_step', row.get('local_step', step)))
        if robot in last_clock and clock != last_clock[robot]+1:
            raise ValueError(f'Discontinuous command clock for robot {robot} at step {step}')
        last_clock[robot] = clock
        episode, task = int(row['episode']), row['motion_phase']
        target = tuple(float(row[axis+'_cmd']) for axis in AXES)
        actual = tuple(float(row['actual_'+axis]) for axis in AXES)
        reward = float(row['reward'])
        if not all(map(math.isfinite, (*target, *actual, reward))):
            raise ValueError(f'Nonfinite tracking measurement at step {step}')
        has_turn |= task == 'turn' or target[2] != 0.
        hold = walk.get('hold_steps') or omni.get(task+'_speed_steps')
        slot = (clock-1)//hold if hold else 0
        segment = active.get(robot)
        if segment is None or (episode, task, target, slot) != (
                segment.episode, segment.task, segment.command, segment.slot):
            if segment is not None:
                segments[robot].append(segment.record())
            segment = active[robot] = CommandSegment(step, episode, task, target, slot)
        segment.add(step, actual)
        fell = row['fell'].lower() in ('true', '1')
        if fell or row['truncated'].lower() in ('true', '1'):
            segments[robot].append(active.pop(robot).record())
        label = coverage_label(task, target, random_forward, yaw_ramp)
        windows[(step-1)//WINDOW_SIZE].add(target, actual, fell, reward, label)
        if compare and step <= EARLY_STEPS:
            early.add(target, actual, fell, reward, label)
        elif compare and step > total - LATE_STEPS:
            late.add(target, actual, fell, reward, label)
    if not total or 0 in seen:
        raise ValueError('The completed run requires one CSV row for every global environment step')
    for robot, segment in active.items():
        segments[robot].append(segment.record())
    change = dict(early_steps=EARLY_STEPS, late_steps=LATE_STEPS, metrics={})
    if compare:
        first = early.record(1)
        last = late.record(total-LATE_STEPS+1)
        for axis in ('vx', 'vy', 'xy', 'wz'):
            if axis == 'wz' and not has_turn:
                continue
            before, after = first[axis], last[axis]
            change['metrics'][axis] = dict(early=before, late=after,
                reduction_pct=100 * (before-after) / before if before else None)
    return dict(experiment=cfg.get('experiment_name', cfg.get('experiment', 'SAC-Isaac')),
                total=total, turn=has_turn, robots=dict(sorted(segments.items())), change=change,
                windows=[windows[i].record(i*WINDOW_SIZE+1) for i in sorted(windows)])


def write_report(folder, output=None):
    folder = Path(folder)
    result = json.loads((folder/'result.json').read_text(encoding='utf-8-sig'))
    if result['status'] != 'completed':
        raise ValueError('Select a completed training run')
    snapshot = json.loads((folder/'effective_config.json').read_text(encoding='utf-8-sig'))
    total = result.get('env_steps', result.get('step'))
    if type(total) is not int or total <= 0:
        raise ValueError('Missing completed environment-step count')
    with (folder/'training.csv').open(encoding='utf-8-sig', newline='') as stream:
        data = build_report_data(csv.DictReader(stream), snapshot.get('config', snapshot), total)
    data['run'] = folder.name
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(',', ':')).replace('<', r'\u003c')
    page = Path(__file__).with_name('tracking_view.html').read_text(encoding='utf-8')
    output = Path(output) if output is not None else folder/'report'/'index.html'
    output.parent.mkdir(exist_ok=True)
    staging = output.with_suffix('.tmp')
    staging.write_text(page.replace('__PAYLOAD__', payload), encoding='utf-8')
    staging.replace(output)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    print(write_report(parser.parse_args().run))
