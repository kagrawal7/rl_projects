"""Tabular RL agents from the FrozenLake and CartPole notebooks."""

from __future__ import annotations

from .cartpole_mc import AbstractAgent as CartPoleMCAbstractAgent
from .cartpole_mc import Agent1
from .cartpole_mc import Agent2
from .cartpole_mc import Agent3
from .cartpole_td import AgentA
from .cartpole_td import AgentB
from .cartpole_td import AgentC
from .cartpole_td import TDAgent
from .frozen_lake import AbstractAgent as FrozenLakeAbstractAgent
from .frozen_lake import DeterministicPolicyIteration
from .frozen_lake import GPIAgent as FrozenLakeGPIAgent
from .frozen_lake import RandomPolicyAgent
from .frozen_lake import RandomSelectionAgent
from .frozen_lake import StochasticPolicyGPI
from .frozen_lake import ValueIterClass

__all__ = [
    "Agent1",
    "Agent2",
    "Agent3",
    "AgentA",
    "AgentB",
    "AgentC",
    "CartPoleMCAbstractAgent",
    "DeterministicPolicyIteration",
    "FrozenLakeAbstractAgent",
    "FrozenLakeGPIAgent",
    "RandomPolicyAgent",
    "RandomSelectionAgent",
    "StochasticPolicyGPI",
    "TDAgent",
    "ValueIterClass",
]
