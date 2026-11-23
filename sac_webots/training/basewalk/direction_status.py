"""Display-only direction diagnostics; never fed to the policy or Reward."""
import json
import math
from pathlib import Path


def wrap_degrees(value):
    return math.degrees(math.atan2(math.sin(math.radians(value)), math.cos(math.radians(value))))


def direction_metrics(position, origin, initial_yaw, yaw, target_angle,
                      target_body_angle=0.):
    dx, dy = position[0] - origin[0], position[1] - origin[1]
    actual = error = None
    # Net displacement is ill-defined at reset and unstable over tiny distances.
    if math.hypot(dx, dy) >= .05:
        actual = wrap_degrees(math.degrees(math.atan2(dy, dx) - initial_yaw))
        error = wrap_degrees(actual - target_angle)
    body_yaw = wrap_degrees(math.degrees(yaw-initial_yaw))
    body_yaw_error = wrap_degrees(target_body_angle-body_yaw)
    return dict(target_angle_deg=wrap_degrees(target_angle), displacement_angle_deg=actual,
                direction_error_deg=error, body_yaw_deg=body_yaw,
                heading_error_deg=body_yaw_error)


def format_direction(state):
    if state.get('fixed_cycle_commands'):
        vx,vy,wz=state['active_command']
        return (f"[command] stage={state['phase']} vx={vx:+.4f} vy={vy:+.4f} wz={wz:+.4f} "
                f"cycles={state['command_cycles']}/{state['command_cycle_limit']} "
                f"global_step={state.get('step',0)} phase={state['gait_phase']:.3f}")
    def angle(key):
        value = state.get(key)
        return 'N/A' if value is None else f'{value:+.2f}deg'
    gait = state.get('gait_phase')
    gait_text = 'N/A' if gait is None else f'{float(gait):.3f}'
    return (f"[direction] target={angle('target_angle_deg')} "
            f"heading_error={angle('heading_error_deg')} "
            f"heading={angle('body_yaw_deg')} "
            f"stage={state.get('phase', 'unknown')} "
            f"global_step={state.get('step', 0)} phase={gait_text}")


def format_reward_breakdown(state):
    """Current environment reward, before retrospective gait-cycle fall backfill."""
    terms = state.get('instant_reward_terms')
    if not terms:
        return None
    if 'tracking_xy' in terms:
        return f"[reward provisional-step; Replay uses cycle-integrated reward] total={state['instant_reward']:+.6f} tracking_xy={terms['tracking_xy']:+.6f} tracking_yaw={terms['tracking_yaw']:+.6f} fall={terms.get('fall', 0.):+.6f}"
    walk = float(terms.get('walk_weight', 0.))
    turn = float(terms.get('turn_weight', 0.)) + float(terms.get('stand_weight', 0.))
    components = {}
    if turn:
        for label, key in (('progress', 'turn_heading_progress'),
                           ('translation', 'turn_linear_velocity'),
                           ('heading', 'turn_heading_error'),
                           ('yaw_rate', 'turn_yaw_rate_error'),
                           ('hold', 'turn_hold')):
            components[label] = turn * float(terms.get(key, 0.))
    if walk:
        for label, key in (('forward', 'walk_forward_progress'),
                           ('lateral', 'walk_lateral_speed'),
                           ('heading', 'walk_heading_error'),
                           ('yaw_rate', 'walk_yaw_rate_error'),
                           ('speed', 'walk_speed_error')):
            components[label] = components.get(label, 0.) + walk * float(terms.get(key, 0.))
    components['fall'] = float(terms.get('fall', 0.))
    total = float(state['instant_reward'])
    remainder = total - sum(components.values())
    if abs(remainder) > 1e-9:
        components['other'] = remainder
    details = ' '.join(f'{key}={value:+.6f}' for key, value in components.items())
    return f'[reward step={state.get("step", 0)} raw] total={total:+.6f} {details}'


class DirectionConsole:
    """Forward every transition and the latest angles into the launcher console."""
    def __init__(self, folder):
        self.folder = Path(folder)
        self.offset = 0
        self.last_state = None

    def poll(self):
        events = self.folder/'direction_events.jsonl'
        if events.exists():
            with events.open(encoding='utf-8') as handle:
                handle.seek(self.offset)
                while True:
                    position = handle.tell()
                    line = handle.readline()
                    if not line.endswith('\n'):
                        self.offset = position
                        break
                    state = json.loads(line)
                    print(format_direction(state), flush=True)
                    breakdown = format_reward_breakdown(state)
                    if breakdown:
                        print(breakdown, flush=True)
                    self.offset = handle.tell()
        path = self.folder/'direction_status.json'
        if path.exists():
            try:
                state = json.loads(path.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                return  # A subsequent poll retries a transient Windows reader lock.
            signature = (state.get('step'), state.get('direction_deg'), state.get('phase'))
            if signature != self.last_state:
                print(format_direction(state), flush=True)
                breakdown = format_reward_breakdown(state)
                if breakdown:
                    print(breakdown, flush=True)
                self.last_state = signature
