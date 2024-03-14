from abc import ABC

import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from gymnasium.wrappers import TimeLimit

from QRL.environment.environment import Environment


class OpenAIGym(Environment, ABC):
    """
    Superclass for all kinds of OpenAI environments
    Wrapper for all OpenAI Environments
    """

    def __init__(self, name: str, render=False, **kwargs) -> None:
        super().__init__()
        self._name = name
        render_mode = 'human' if render else None
        self._env = gym.make(name, render_mode=render_mode, **kwargs)

    def reset(self):
        return self._env.reset()

    def step(self, action):
        return self._env.step(action)

    def render(self):
        if self._env.render_mode:
            self._env.render()

    def close(self) -> None:
        self._env.close()

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def n_actions(self):
        return self._env.action_space.n

    @property
    def state_size(self):
        if self.isdiscrete:
            return self._env.observation_space.n
        else:
            return self._env.observation_space.shape[0]

    @property
    def isdiscrete(self) -> bool:
        return hasattr(self._env.observation_space, 'n')

    @property
    def name(self) -> str:
        return self._name

    @property
    def map_shape(self) -> str:
        return self._env.unwrapped.desc.shape

    @property
    def map(self) -> str:
        return str(self.map_arr).replace("b'F'", '⬜️').replace("b'G'", '🔶').replace("b'H'", '🚫').replace("b'S'", '🟧')

    @property
    def map_arr(self):
        return self._env.unwrapped.desc


class FrozenLakeEnvironment(OpenAIGym):

    def __init__(self, render=False, is_slippery=True, random=False, size=4) -> None:
        if size not in [4, 8] and not random:
            raise ValueError("Size must be 4 or 8 or random must be True")
        super().__init__(name="FrozenLake-v1", render=render, is_slippery=is_slippery,
                         desc=generate_random_map(size) if random else None,
                         map_name=None if random else '4x4' if size == 4 else '8x8')


class CartPoleEnvironment(OpenAIGym):

    def __init__(self, render=False) -> None:
        super().__init__(name="CartPole-v1", render=render)


class CliffWalkingEnvironment(OpenAIGym):

    def __init__(self, render=False) -> None:
        super().__init__(name="CliffWalking-v0", render=render)
