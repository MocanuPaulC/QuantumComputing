from abc import ABC, abstractmethod


class Environment(ABC):
    """
    Abstract Environment
    Superclass for all kinds of environments
    """

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def render(self):
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    @property
    @abstractmethod
    def action_space(self):
        pass

    @property
    @abstractmethod
    def observation_space(self):
        pass

    @property
    @abstractmethod
    def n_actions(self):
        pass

    @property
    @abstractmethod
    def state_size(self):
        pass

    @property
    @abstractmethod
    def isdiscrete(self) -> bool:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass
