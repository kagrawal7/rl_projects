"""Public Monte Carlo control algorithms."""

from rl_suite._algorithms._mc._off_policy._off_policy_control import _OffPolicyAgent


class OffPolicyMonteCarlo(_OffPolicyAgent):
    """Weighted importance-sampling off-policy Monte Carlo control agent."""


OffPolicyMC = OffPolicyMonteCarlo

IMPLEMENTED_ALGOS = (OffPolicyMonteCarlo,)


__all__ = ["IMPLEMENTED_ALGOS", "OffPolicyMC", "OffPolicyMonteCarlo"]
