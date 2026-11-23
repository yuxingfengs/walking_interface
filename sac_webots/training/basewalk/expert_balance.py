"""Bounded posture feedback candidate; not a full footstep/ZMP controller."""
import math
import numpy as np
def stabilize(target,gravity,gyro,settings):
    q=np.asarray(target,dtype=float).copy()
    pitch=math.asin(float(np.clip(gravity[0],-1,1)))
    roll=math.atan2(-float(gravity[1]),-float(gravity[2]))
    dp=float(np.clip(settings['balance_pitch_gain']*pitch+settings.get('balance_pitch_d',.08)*gyro[1],-.18,.18))
    dr=float(np.clip(settings.get('balance_roll_gain',0.)*roll+settings.get('balance_roll_d',0.)*gyro[0],-.12,.12))
    q[[4,10]]+=dp;q[[5,11]]+=dr
    return q.astype(np.float32)
