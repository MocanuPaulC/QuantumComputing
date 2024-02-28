from abc import ABC, abstractmethod

import numpy as np

from QRL.agent.episode import Episode
from QRL.environment.environment import Environment


class LearningStrategy(ABC):
    """
    Implementations of this class represent a Learning Method
    This class is INCOMPLETE
    """
    env: Environment

    def __init__(self, environment: Environment,α, λ, γ, t_max,ϵ_min=0.0005, ϵ_max=0.5) -> None:
        self.env = environment
        self.α = α  # learning rate (given)
        self.λ = λ  # exponential decay rate used for exploration/exploitation (given)
        self.γ = γ  # discount rate for exploration (given)
        self.ϵ_max = ϵ_max  # Exploration probability at start (given)
        self.ϵ_min = ϵ_min  # Minimum exploration probability (given)

        self.ϵ = self.ϵ_max  # (decaying) probability of selecting random action according to ϵ-soft policy
        self.t_max = t_max  # upper limit voor episode
        self.t = 0  # episode time step
        self.τ = 0  # overall time step

    @abstractmethod
    def next_action(self, state):
        pass

    @abstractmethod
    def learn(self, episode: Episode):
        # at this point subclasses insert their implementation
        # see for example be\kdg\rl\learning\tabular\tabular_learning.py
        self.t += 1

    @abstractmethod
    def on_episode_start(self):
        """
        Implements all necessary initialization that needs to be done at the start of new Episode
        Each subclass learning algorithm should decide what to do here
        """
        self.t = 0

    def done(self):
        return self.t > self.t_max

    def decay(self):
        self.ϵ = self.ϵ_min + (self.ϵ_max - self.ϵ_min) * np.exp(-self.λ * self.τ)

    def on_episode_end(self):
        self.τ += 1

    @abstractmethod
    def get_loss(self):
        pass
