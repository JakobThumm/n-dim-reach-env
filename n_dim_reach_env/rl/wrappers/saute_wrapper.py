"""This file describes a wrapper that adds the cost of the environment to the observation.

The reward is set to the maximum negative reward if the cumultaive cost in the episode exceeds the cost threshold.
"""
import gym
import numpy as np
from copy import copy


class SauteWrapper(gym.Wrapper):
    """This wrapper adds the cost of the environment to the observation and adapts the reward."""

    def __init__(self,
                 env: gym.Env,
                 cost_threshold: float = 1.0,
                 min_step_reward: float = -1.0):
        """Initialize the wrapper.

        Args:
            env (gym.Env): Environment to wrap.
            cost_threshold (float): Threshold for the cost.
            min_step_reward (float): Minimum possible reward for a step.
        """
        super().__init__(env)
        self.cost_threshold = cost_threshold
        self.cumulative_cost = 0.0
        self.min_step_reward = min_step_reward
        if isinstance(self.env.observation_space, gym.spaces.Dict):
            assert "observation" in self.env.observation_space.spaces
            self._observation_space = copy(self.env.observation_space)
            obs_space = self._observation_space.spaces["observation"]
            self._observation_space.spaces["observation"] = gym.spaces.Box(
                low=np.append(obs_space.low, 0),
                high=np.append(obs_space.high, cost_threshold),
                dtype=obs_space.dtype)
            self.dict_space = True
        elif isinstance(self.env.observation_space, gym.spaces.Box):
            obs_space = self.env.observation_space
            self._observation_space = gym.spaces.Box(
                low=np.append(obs_space.low, 0),
                high=np.append(obs_space.high, cost_threshold),
                dtype=obs_space.dtype)
            self.dict_space = False
        else:
            raise NotImplementedError

    def reset(self, **kwargs):
        """Reset the environment."""
        observation = self.env.reset(**kwargs)
        self.cumulative_cost = 0.0
        return self.observation(observation)

    def step(self, action):
        """Step the environment."""
        observation, reward, done, info = self.env.step(action)
        if "cost" in info:
            self.cumulative_cost += info["cost"]
        info["dense_reward"] = reward
        return self.observation(observation), self.reward(reward, info["cost"]), done, info

    def reward(self, reward, cost):
        """Adapt the reward to minimal reward if cost threshold is reached."""
        return reward #* 0.99999**(self.cumulative_cost)
        """
        if self.cumulative_cost >= self.cost_threshold:
            return self.min_step_reward
        else:
            return reward
        """

    def observation(self, observation):
        """Add the cost to the observation."""
        if self.dict_space:
            observation["observation"] = np.append(observation["observation"], self.cumulative_cost)
        else:
            observation = np.append(observation, self.cumulative_cost)
        return observation
