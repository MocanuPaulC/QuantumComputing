import numpy as np

from QRL.agent.percept import Percept
from QRL.environment.environment import Environment


class MarkovDecisionProcess:
    """
    This class is INCOMPLETE.
    Making abstraction of Environment using a mathematical model (MDP) for the environment (Environment)
    You don't need this class for implementing the algorithms, because we'll be using model-free algorithms,
    so we'll need to learn the model from experience.
    Implementing this can give more insight into Reinforcement Learning, though.
    """

    def __init__(self, environment: Environment) -> None:
        self.env = environment  # (given)
        self.n_actions = self.env.n_actions  # (given)
        self.n_states = self.env.state_size  # (given)

        # state-action-next state reward model (learned)
        self._reward_model = np.zeros((self.n_states, self.n_actions))

        # how often state s and action a occurred (learned)
        self.n_sa = np.zeros((self.n_states, self.n_actions))

        # how often has state t followed a state s after action a (learned)
        self.n_tsa = np.zeros((self.n_states, self.n_states, self.n_actions))

        # Markov Decision Process transition model (learned)
        self.P = np.zeros((self.n_states, self.n_states, self.n_actions))

        # Update count
        self.n = 0

    def update(self, percept: Percept) -> None:
        self.n += 1
        self.update_reward(percept)
        self.update_counts(percept)
        self.update_transition_model(percept)

    def p(self, tsa) -> float:
        return self.P[tsa]

    def reward(self, state: int, action: int) -> float:
        return self._reward_model[state, action]

    def update_reward(self, p: Percept) -> None:
        # TODO: COMPLETE THE CODE
        pass

    def update_counts(self, percept: Percept) -> None:
        # TODO: COMPLETE THE CODE
        pass

    def update_transition_model(self, percept: Percept) -> None:
        # TODO: COMPLETE THE CODE
        pass
