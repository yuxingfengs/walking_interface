"""V28 data timing and support contract checks, shared by BC and online loading."""
import numpy as np
from .network_inputs import observation_dim


def validate_arrays(a,cfg):
    n=len(a['executed_actions'])
    width = observation_dim(cfg)
    for name,shape in [('observations',(n,width)),('next_observations',(n,width)),
                       ('executed_actions',(n,12)),('support_normal_loads',(n,2)),('next_support_normal_loads',(n,2))]:
        if name not in a or a[name].shape!=shape or not np.isfinite(a[name]).all():
            raise ValueError('Fresh V28 array required: '+name)
    if n==0:raise ValueError('Empty trajectory')
    if not np.allclose(a['next_observations'][:,34:46],a['executed_actions'],atol=1e-6):
        raise ValueError('Previous Action must be executed Action, not clean BC target')
    if not np.allclose(a['observations'][0,34:46],0.,atol=1e-6):
        raise ValueError('First State must reset previous Action to zero')
    if n>1 and not np.allclose(a['observations'][1:],a['next_observations'][:-1],atol=1e-6):
        raise ValueError('State chain has timing discontinuity')
    epsilon=cfg['support_observation']['epsilon']
    for prefix in ('','next_'):
        loads=a[prefix+'support_normal_loads']
        if np.any(loads<0):raise ValueError('Normal loads must be nonnegative')
        support=(loads[:,0]-loads[:,1])/(loads.sum(axis=1)+epsilon)
        if not np.allclose(support,a[prefix+'observations'][:,49],atol=1e-6):
            raise ValueError('Support State must correspond to same-time measured loads')
