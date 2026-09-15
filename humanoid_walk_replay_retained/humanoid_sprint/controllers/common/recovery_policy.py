"""The initial BC load is separate from the finite BC recovery window."""


def within_bc_reset_window(initialization, rollout_number):
    """rollout_number is one-based and includes rollouts before a resume."""
    cutoff = initialization.get('bc_reset_max_rollouts', 0)
    return cutoff == 0 or rollout_number <= cutoff
