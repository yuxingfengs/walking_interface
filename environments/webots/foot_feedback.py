"""Copy Webots force vectors, including ctypes pointer return values."""
import numpy as np


def read_foot_forces(sensors):
    rows = []
    for sensor in sensors:
        values = sensor.getValues()
        # Some Webots versions return POINTER(c_double), not a Python list.
        # Read exactly three components; never infer a pointer's buffer shape.
        rows.append([float(values[i]) for i in range(3)])
    forces = np.asarray(rows, dtype=float)
    if forces.shape != (2, 3) or not np.isfinite(forces).all():
        raise RuntimeError('Invalid LFsr/RFsr force feedback')
    return forces
