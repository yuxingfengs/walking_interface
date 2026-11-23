"""Canonical experiment003 logs with reversible aliases and State chains."""
import json

OBSOLETE = set('''reward_turn_translation reward_turn_linear_velocity reward_walk_forward_progress
reward_walk_heading_progress reward_turn_heading_progress reward_turn_heading_error reward_turn_yaw_rate_error
reward_walk_heading_error reward_walk_cross_track reward_survival reward_height reward_action_rate
reward_joint_velocity replay_online_mixed replay_expert_turn replay_expert_stand replay_expert_walk
replay_online_turn replay_expert_turn_stand_walk walk_path_m online_turn_fifo_count reward_stand_excess_translation
reward_command_calibration_offset reward_turn_hold stand_hold_steps reward_walk_speed_error
reward_walk_lateral_speed reward_walk_yaw_rate_error reward_stand_total'''.split())
ALIASES = dict(reward_vx_cmd='vx_cmd',reward_vy_cmd='vy_cmd',reward_wz_cmd='wz_cmd',
    next_critic_body_vx='actual_vx',next_critic_body_vy='actual_vy',actor_yaw_rate='actual_wz')
HEAVY = {'source_observation','next_observation','executed_action','reward_gravity','reward_gyro_xy'}
DERIVED = {'reward_velocity_tracking','reward_turn_total','reward_walk_total','reward_turn_weighted','reward_walk_weighted'}


def fields(names):
    return [k for k in names if k not in OBSOLETE|set(ALIASES)|HEAVY|DERIVED|{'vx','vy','wz'}]


def canonical(row, previous):
    out = {k:v for k,v in row.items() if k not in OBSOLETE|set(ALIASES)|DERIVED|{'vx','vy','wz'} and v is not None and v != ''}
    # Source is exactly the prior next-State inside a continuous segment.
    if previous is not None and out.get('source_observation') == previous:
        out.pop('source_observation',None)
    return out


def restore(row, previous):
    out = dict(row)
    if 'next_observation' not in out:
        raise ValueError('Compact log is missing next Observation')
    if 'source_observation' not in out:
        if previous is None:
            raise ValueError('Compact log starts without source Observation')
        out['source_observation'] = previous
    nxt = out['next_observation']
    out.setdefault('executed_action',nxt[34:46])
    out.setdefault('reward_gravity',nxt[:3]);out.setdefault('reward_gyro_xy',nxt[3:5])
    for alias,key in ALIASES.items():
        if key in out:out.setdefault(alias,out[key])
    return out


def records(lines):
    previous = None
    for line in lines:
        if not line.strip():continue
        raw=json.loads(line)
        if raw.get('log_schema') == 'compact_transition_v3':
            raw=restore(raw,previous)
        previous=raw.get('next_observation')
        yield raw
