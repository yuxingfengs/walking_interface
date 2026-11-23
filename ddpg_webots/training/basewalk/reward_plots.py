"""Plot every logged control-step reward by its source motion stage."""
import csv
import json
import math
from pathlib import Path


def plot_rewards(folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    folder = Path(folder)
    source = folder/'training.csv'
    if not source.exists():
        return None
    groups = {stage: [] for stage in ('turn', 'stand', 'walk')}
    batch_q = []
    stage_q = {stage: [] for stage in groups}
    skipped = 0
    with source.open(newline='', encoding='utf-8-sig') as stream:
        for row in csv.DictReader(stream):
            try:
                step, reward = int(row['step']), float(row['reward'])
                stage = row['motion_phase']
                if stage not in groups or not math.isfinite(reward):
                    raise ValueError('Invalid reward/stage')
                groups[stage].append((step, reward))
                action_q = row.get('q_executed', '')
                if action_q and math.isfinite(float(action_q)):
                    stage_q[stage].append((step,float(action_q)))
                raw_q = row.get('q_mean', '')
                if raw_q:
                    q = float(raw_q)
                    if math.isfinite(q):
                        batch_q.append((step, q))
            except (KeyError, ValueError, TypeError):
                skipped += 1
    resets = {}
    recovery_path = folder/'recoveries.json'
    if recovery_path.exists():
        for event in json.loads(recovery_path.read_text(encoding='utf-8')):
            resets[int(event['step'])] = event.get('reason', 'BC reset')
    episodes = folder/'training_episodes.csv'
    if episodes.exists():
        with episodes.open(newline='', encoding='utf-8-sig') as stream:
            for row in csv.DictReader(stream):
                if row.get('bc_recovery'):
                    resets.setdefault(int(row['end_step']), row['bc_recovery'])
    output = folder/'reward_plots' 
    output.mkdir(exist_ok=True)
    for stage, rows in groups.items():
        action_q=stage_q[stage]
        import numpy as np
        fig, zoom = plt.subplots(figsize=(12, 6), constrained_layout=True)
        central_ranges = []
        full_fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
        fig.suptitle(f'{stage.title()} Reward and Q - every logged sample')
        if rows:
            steps, rewards = zip(*rows)
            # Linear around typical rewards; logarithmic compression of extremes.
            threshold = max(float(np.quantile(np.abs(rewards), .95)), .001)
            ax.set_yscale('symlog', linthresh=threshold, linscale=2.)
            for axis in (ax, zoom):
                axis.scatter(steps, rewards, s=5, alpha=.7, linewidths=0,
                             color='tab:blue', label=f'{stage.title()} Reward')
            central_ranges.append(np.quantile(rewards, [.10, .90]))
        else:
            for axis in (ax, zoom):
                axis.text(.5, .5, 'No logged steps in this stage',
                          transform=axis.transAxes, ha='center', va='center')
        if action_q:
            qsteps, qvalues = zip(*action_q)
            central_ranges.append(np.quantile(qvalues, [.10, .90]))
            for axis in (zoom, ax):
                axis.scatter(qsteps, qvalues, s=7, alpha=.8, color='tab:orange',
                             linewidths=0, label='Executed-action Q(s,a)')
        if not action_q:
            zoom.text(.01,.94,'No executed-action Q recorded (historical logs cannot be reconstructed)',transform=zoom.transAxes,fontsize=8)
        if central_ranges:
            low = min(bounds[0] for bounds in central_ranges)
            high = max(bounds[1] for bounds in central_ranges)
            pad = max((high-low)*.03, .0001)
            lower, upper = float(low-pad), float(high+pad)
            zoom.set_ylim(lower, upper)
            outside_reward = sum(not lower <= value <= upper for _, value in rows)
            outside_q = sum(not lower <= value <= upper for _, value in action_q)
            zoom.set_title(f'Shared scale: central 10-90% of each series + 3% margin; '
                           f'outside view: Reward {outside_reward}, Q {outside_q}', fontsize=9)
            ax.set_yscale('symlog', linthresh=max(abs(lower), abs(upper), .001), linscale=2.)
        ax.set_title('All Reward and Q points; extreme values compressed (symlog)', fontsize=10)
        for axis in (zoom, ax):
            axis.set_ylabel('Reward / Q (same numerical scale)')
            axis.set_xlabel('Global environment step')
            axis.grid(alpha=.2)
        for index, (step, reason) in enumerate(sorted(resets.items())):
            for axis in (zoom, ax):
                axis.axvline(step, color='crimson', linestyle='--', linewidth=1.,
                             label='BC reset' if index == 0 else None)
                axis.text(step, .98-(index%3)*.08, f'BC reset {step}',
                          transform=axis.get_xaxis_transform(), rotation=90,
                          ha='right', va='top', fontsize=8, color='crimson')
        if not resets:
            zoom.text(.01, .02, 'No recorded BC reset', transform=zoom.transAxes, fontsize=9)
        for axis in (zoom, ax):
            handles, labels = axis.get_legend_handles_labels()
            if handles:
                axis.legend(handles, labels, loc='lower right', fontsize=8)
        fig.savefig(output/f'{stage}_reward.png', dpi=160)
        plt.close(fig)
        full_fig.savefig(output/f'{stage}_full_reward.png', dpi=160)
        plt.close(full_fig)
        with (output/f'{stage}_reward.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(('step','reward','q_executed'))
            qmap=dict(action_q)
            writer.writerows((step,reward,qmap.get(step,'')) for step,reward in rows)
    with (output/'mixed_batch_q.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(('step', 'q_mean'))
        writer.writerows(batch_q)
    batch_fig,batch_ax=plt.subplots(figsize=(12,4),constrained_layout=True)
    if batch_q:batch_ax.scatter(*zip(*batch_q),s=5,color='tab:orange')
    batch_ax.set(xlabel='Global environment step',ylabel='Mixed-batch mean Q',title='Replay batch diagnostic (not current-action Q)')
    batch_fig.savefig(output/'mixed_batch_q.png',dpi=160);plt.close(batch_fig)
    (output/'summary.json').write_text(json.dumps(dict(
        counts={key: len(value) for key, value in groups.items()}, skipped_rows=skipped,
        source=str(source), reward='training.csv reward after configured cycle fall backfill',
        q_samples=sum(map(len,stage_q.values())), q_by_stage={k:len(v) for k,v in stage_q.items()}, q='Source-state executed-action Q before optimizer update, grouped by source command',
        detail_axis='shared linear axis; union of each series 10-90 percentiles plus 3 percent margin',
        full_reward_axis='shared symlog; linear threshold covers detail limits, linscale=2',
        bc_resets=[dict(step=k,reason=v) for k,v in sorted(resets.items())],
        smoothing=False, downsampling=False), indent=2), encoding='utf-8')
    (output/'index.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>Reward and Q</title>'
        '<style>body{max-width:1200px;margin:24px auto;font-family:sans-serif}img{width:100%}</style>'
        '<h1>Reward and Q</h1><p>Default: shared linear axis, central 10-90% of each series plus 3% margin. Outliers remain in CSV and full-range figures. Blue: Reward; orange: executed-action Q(s,a); both use the same numerical axis; red dashed: recorded BC reset. Same global-step coordinates; missing Q is not interpolated. No smoothing. Q is grouped by the executed source stage. Historical missing Q is never replaced with batch Q. Reward is immediate; Q estimates discounted return, so equal values are not expected.</p>'
        + '<h2>Mixed batch Q (separate)</h2><img src="mixed_batch_q.png" alt="Mixed batch Q">'
        + ''.join(f'<h2>{stage.title()}</h2><img src="{stage}_reward.png" alt="{stage} Reward and Q"><p><a href="{stage}_full_reward.png">Full Reward and Q range including falls</a></p>' for stage in groups),
        encoding='utf-8')
    return output


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('run_folder', type=Path)
    print(plot_rewards(parser.parse_args().run_folder))
