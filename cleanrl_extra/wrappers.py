# cleanrl_extra/wrappers.py
import gymnasium as gym
import numpy as np


class PartialObsWrapper(gym.ObservationWrapper):
    """
    Drops qvel (velocity) features from the MuJoCo state → partially
    observable locomotion tasks (Hopper, Walker2d, HalfCheetah, Ant).
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        dim        = env.observation_space.shape[0]
        self._mask = np.concatenate([np.ones(dim // 2),
                                     np.zeros(dim - dim // 2)]).astype(bool)
        low, high  = env.observation_space.low[self._mask], \
                     env.observation_space.high[self._mask]
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)

    def observation(self, obs):
        return obs[self._mask]
