"""Actor factory. Architecture is part of the saved model contract."""


def create_actor(config):
    kind=config['actor_type']
    if kind=='film_gait_correction':
        raise ValueError('V28 50D supports concat_mlp only')
    elif kind=='concat_mlp':
        from .concat_actor import Actor
    else:
        raise ValueError('Unknown Actor architecture: '+str(kind))
    return Actor(config)
