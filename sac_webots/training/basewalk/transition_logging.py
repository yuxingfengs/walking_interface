"""Write final, cycle-backfilled transition rows without changing legacy CSV schemas."""
import csv
import json


class TransitionWriter:
    def __init__(self, csv_handle, trace_handle, fields, config=None):
        self.compact = (config or {}).get('log_layout') == 'compact_transition_v3'
        self.previous_next = None
        if self.compact:
            from .compact_logs import fields as compact_fields
            fields = compact_fields(fields)
        self.csv = csv.DictWriter(csv_handle, fieldnames=fields, extrasaction='ignore')
        self.trace = trace_handle

    def writeheader(self):
        self.csv.writeheader()

    def writerow(self, row):
        # PhaseFallBuffer has already finalized reward/fall before calling this method.
        detail = {k: v for k, v in row.items() if not k.startswith('_')}
        # Only after PhaseFallBuffer finalization: report sees the exact Replay Reward.
        detail.update(time_s=int(row['step'])*float(row['cycle_dt']),
            command_vx=row['reward_vx_cmd'],command_vy=row['reward_vy_cmd'],command_wz=row['reward_wz_cmd'],
            reward_input_vx=row['reward_vx'],reward_input_vy=row['reward_vy'],reward_input_wz=row['reward_wz'])
        for key in ('source_observation', 'executed_action', 'next_observation', 'reward_gravity', 'reward_gyro_xy'):
            if isinstance(detail.get(key), str):
                detail[key] = json.loads(detail[key])
        if self.compact:
            from .compact_logs import canonical
            # These aliases were added for old report formats; canonical names
            # are retained once and restored by the reader as needed.
            for key in ('command_vx','command_vy','command_wz','reward_input_vx','reward_input_vy','reward_input_wz'):
                detail.pop(key,None)
            nxt = detail.get('next_observation')
            detail = canonical(detail,self.previous_next)
            for key in ('executed_action','reward_gravity','reward_gyro_xy'):
                detail.pop(key,None)
            self.previous_next = nxt
            detail['log_schema'] = 'compact_transition_v3'
        encoded = json.dumps(detail, separators=(',', ':'), allow_nan=False,
                             default=lambda v: v.item())
        self.csv.writerow(row)
        self.trace.write(encoded+'\n')
