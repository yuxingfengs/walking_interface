"""Fixed-cycle schedule for the original 001 through 003 Isaac tasks."""


def command_schedule(cfg, state=None):
    from basewalk.fixed_cycle import FixedCycleStages
    training = cfg['training']
    return FixedCycleStages(cfg, cycle_limit=training['online_fixed_cycle_limit'],
                            turn_only=training.get('online_turn_only', True),
                            fixed_turn_sign=training['online_fixed_turn_sign'], state=state)
