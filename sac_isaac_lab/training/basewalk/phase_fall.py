"""Finalize a gait cycle before exposing its rewards to Replay sampling."""
import copy
import math


def fall_penalty_at(progress, magnitude):
    """Legacy S0 curve, normalized to the actual collected segment length."""
    if progress < .8:
        return -.02 * magnitude
    return -magnitude * (.002 + .998 * math.expm1(10 * (progress - .8)) / math.expm1(2))


def cycle_yaw_adjustments(cfg, commands, rates, durations, yaw_rewards):
    """Return per-step score adjustments for a completed single-command Turn."""
    mode = cfg['reward'].get('biped_tracking', {}).get('cycle_yaw_scoring')
    if mode is None or not commands:
        return None
    command = commands[0]
    if not command or any(value != command for value in commands):
        return None
    rate = math.fsum(v * dt for v, dt in zip(rates, durations)) / math.fsum(durations)
    error = abs(rate - command)
    if mode == 'absolute_error':
        penalty = -cfg['reward']['biped_tracking']['cycle_yaw_error_weight'] * error / abs(command)
        return [penalty] * len(commands)
    if mode == 'gaussian':
        params = cfg['reward']['biped_tracking']
        cycle_width = params.get('cycle_yaw_gaussian_width_rad_s',
                                 params['turn_yaw_gaussian_width_rad_s'])
        if not math.isfinite(cycle_width) or cycle_width <= 0:
            raise ValueError('Cycle yaw Gaussian width must be positive and finite')
        score = params['turn_yaw_gaussian_weight'] * math.exp(
            -(error / cycle_width) ** 2)
        redistribution = params.get('cycle_yaw_redistribution_scale')
        if redistribution is not None:
            if not math.isfinite(redistribution) or redistribution < 0:
                raise ValueError('Cycle yaw redistribution scale must be finite and nonnegative')
            gamma = cfg['training']['gamma']
            directed_rates = [math.copysign(1., command) * value for value in rates]
            discounts = [gamma ** index for index in range(len(rates))]
            baseline = math.fsum(w * value for w, value in zip(discounts, directed_rates)) / math.fsum(discounts)
            return [score + redistribution * (value - baseline) - reward
                    for value, reward in zip(directed_rates, yaw_rewards)]
        step_weight = params.get('cycle_yaw_step_weight', 0.)
        if not math.isfinite(step_weight) or step_weight < 0:
            raise ValueError('Cycle yaw step weight must be finite and nonnegative')
        step_source = params.get('cycle_yaw_step_source', 'ema')
        if step_source == 'raw_gyro':
            width = params['turn_yaw_gaussian_width_rad_s']
            peak = params['turn_yaw_gaussian_weight']
            step_scores = [peak * math.exp(-((value - command) / width) ** 2) for value in rates]
        elif step_source == 'ema':
            step_scores = yaw_rewards
        else:
            raise ValueError(f'Unknown cycle yaw step source: {step_source}')
        return [score + step_weight * step_score - reward
                for step_score, reward in zip(step_scores, yaw_rewards)]
    raise ValueError(f'Unknown cycle yaw scoring mode: {mode}')


class PhaseFallBuffer:
    def __init__(self, magnitude, tracking_config):
        if not math.isfinite(magnitude) or magnitude < 0:
            raise ValueError('Fall magnitude must be finite and nonnegative')
        self.magnitude = magnitude
        self.pending = []
        self.tracking_config = tracking_config

    def append(self, obs, action, reward, nxt, terminal, kwargs, row, console):
        # Robot adapters may reuse arrays; retain the original source/next states.
        transition = copy.deepcopy((obs, action, reward, nxt, terminal, kwargs))
        self.pending.append((transition, row, console))

    def flush(self, replay, writer, fell=False, complete_cycle=False, after_replay=None, archive_cycle=None):
        adjustment = 0.
        count = len(self.pending)
        yaw_deltas = None
        if complete_cycle and not fell and self.tracking_config['reward'].get('biped_tracking', {}).get('cycle_yaw_scoring'):
            rows = [row for _, row, _ in self.pending]
            if all(row['motion_phase'] == 'turn' for row in rows):
                durations = [row['_cycle_dt'] for row in rows]
                source = self.tracking_config['reward']['biped_tracking'].get('cycle_yaw_rate_source', 'body_gyro_z')
                if source == 'heading_delta':
                    rates = [row['_cycle_actual'][2] / dt for row, dt in zip(rows, durations)]
                elif source == 'body_gyro_z':
                    rates = [row['sensor_gyro_z'] for row in rows]
                else:
                    raise ValueError(f'Unknown cycle yaw rate source: {source}')
                yaw_deltas = cycle_yaw_adjustments(
                    self.tracking_config, [row['wz_cmd'] if 'wz_cmd' in row else row['reward_wz_cmd'] for row in rows],
                    rates, durations,
                    [row['reward_tracking_yaw'] for row in rows])
        archive = [] if archive_cycle is not None and complete_cycle and not fell else None
        for index, (transition, row, console) in enumerate(self.pending):
            obs, action, reward, nxt, terminal, kwargs = transition
            for key in ('_cycle_actual','_cycle_target','_cycle_dt'):
                row.pop(key, None)
            if yaw_deltas is not None:
                delta = yaw_deltas[index]
                reward += delta
                adjustment += delta
                row['reward'] = reward
                row['reward_tracking_yaw'] += delta
                row['reward_turn_weighted'] += delta
            if console is not None:
                console[0] = row['reward_turn_weighted']
                console[1] = row['reward_walk_weighted']
                console[3] = reward
            if fell:
                penalty = fall_penalty_at((index + 1) / count, self.magnitude)
                delta = penalty - float(row.get('reward_fall') or 0.)
                reward += delta
                adjustment += delta
                row.update(reward=reward, reward_fall=penalty)
                if console is not None:
                    console[2] = penalty
                    console[3] = reward
            insert = replay if callable(replay) else replay.add
            if self.tracking_config.get('outcome_replay') and not self.tracking_config.get('bc_rollout', {}).get('collection_mode'):
                kwargs['cycle_outcome'] = 'failure' if fell else ('success' if complete_cycle else 'incomplete')
                row['replay_destination'] = kwargs['cycle_outcome']
            insert(obs, action, reward, nxt, terminal, **kwargs)
            if archive is not None:
                archive.append((obs, action, reward, nxt, terminal, row))
            if after_replay is None:
                writer.writerow(row)
        if archive:
            archive_cycle(archive)
        if after_replay is not None and self.pending:
            after_replay(self.pending[-1][1])
            for _, row, _ in self.pending:
                writer.writerow(row)
        self.pending.clear()
        return adjustment
