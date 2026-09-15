"""DDPG learning absolute executed actions, with a frozen BC Actor backbone."""
from algorithms.ddpg import DDPGAlgorithm
from residual_actor import ResidualActor


class ResidualDDPGAlgorithm(DDPGAlgorithm):
    algorithm_id='ddpg_bc_residual_heading_v1'

    def __init__(self, *, residual_limit_rad=.03, **kwargs):
        self.residual_limit_rad=float(residual_limit_rad)
        super().__init__(**kwargs)

    def create_actor(self,state_dim,action_dim):
        return ResidualActor(state_dim,256,action_dim,self.residual_limit_rad)

    def limit_exploration(self,observation,action):
        return self.actor.limit_action(observation.values,action).squeeze(0).detach()
