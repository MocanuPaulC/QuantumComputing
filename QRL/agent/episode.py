from collections import deque
import random

from QRL.agent.percept import Percept
from QRL.environment.environment import Environment


class Episode:
    """
    A collection of Percepts forms an Episode. A Percept is added per step/time t.
    The Percept contains the state, action, reward and next_state.
    This class is INCOMPLETE
    """

    def __init__(self, env: Environment, γ=0.9) -> None:
        self._env = env
        self._percepts: [Percept] = deque()
        self.γ = γ

    def add(self, percept: Percept):
        self._percepts.append(percept)

    def percepts(self, n: int):
        """ Return n final percepts from this Episode """
        return list(self._percepts)[-n:]

        pass

    def all_percepts(self):
        """ Return all percepts from this Episode """
        return list(self._percepts)

    def compute_returns(self) -> None:
        """ For EACH Percept in the Episode, calculate its discounted return Gt"""
        G = 0  # Initialize the return for the last percept
        for percept in reversed(self._percepts):
            # Update the return: Gt = Rt + γ * Gt+1
            G = percept.reward + self.γ * G
            # Store the return in the percept
            percept.return_ = G

    def sample(self, batch_size: int):
        """ Sample and return a random batch of Percepts from this Episode """
        return random.sample(self._percepts, batch_size)

    @property
    def size(self):
        return len(self._percepts)
