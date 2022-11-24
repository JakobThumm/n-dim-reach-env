"""This file describes a wrapper that adds the cost of the environment to the observation.

The reward is set to the maximum negative reward if the cumultaive cost in the episode exceeds the cost threshold.
"""
import gym
import math
import numpy as np
from copy import copy
from typing import Union, List, Dict


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
        self.mode = "standard"
        if isinstance(self.env.observation_space, gym.spaces.Dict):
            assert "observation" in self.env.observation_space.spaces
            self._observation_space = copy(self.env.observation_space)
            # Extend the observation space by one dimension (cumm cost)
            obs_space = self._observation_space.spaces["observation"]
            self._observation_space.spaces["observation"] = gym.spaces.Box(
                low=np.append(obs_space.low, 0),
                high=np.append(obs_space.high, cost_threshold),
                dtype=obs_space.dtype)
            # Extend the achieved goal space by one dimension (cumm cost)
            obs_space = self._observation_space.spaces["achieved_goal"]
            self._observation_space.spaces["achieved_goal"] = gym.spaces.Box(
                low=np.append(obs_space.low, 0),
                high=np.append(obs_space.high, cost_threshold),
                dtype=obs_space.dtype)
            # Extend the desired goal space by one dimension (cumm cost)
            obs_space = self._observation_space.spaces["desired_goal"]
            self._observation_space.spaces["desired_goal"] = gym.spaces.Box(
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
        if "dense_reward" not in info:
            info["dense_reward"] = reward
        return self.observation(observation), self.reward(reward, info["cost"]), done, info

    def reward(self, reward, cost):
        """Adapt the reward to minimal reward if cost threshold is reached."""
        if self.mode == "exponential":
            reward = math.pow(0.1, (self.cumulative_cost/self.cost_threshold)) * reward
            return reward
        else:
            if self.cumulative_cost >= self.cost_threshold:
                return self.min_step_reward
            else:
                return reward

    def compute_reward(
        self,
        achieved_goal: Union[List[np.ndarray], np.ndarray],
        desired_goal: Union[List[np.ndarray], np.ndarray],
        info: Union[np.ndarray, List[Dict], Dict]
    ) -> Union[np.ndarray, List[float], float]:
        """Compute the step reward.

        This externalizes the reward function and makes it dependent
        on an a desired goal and the one that was achieved.
        If you wish to include additional rewards that are independent
        of the goal, you can include the necessary values to derive
        it in info and compute it accordingly.
        Args:
            achieved_goal (object): the goal that was achieved during execution
            desired_goal (object): the desired goal that we asked the agent to
                attempt to achieve.
            info (dict): an info dictionary with additional information
        Returns:
            float: The reward that corresponds to the provided achieved goal
                w.r.t. to the desired goal.
                Note that the following should always hold true:
                ob, reward, done, info = env.step()
                assert reward == env.compute_reward(
                    ob['achieved_goal'], ob['goal'], info)
        """
        reward = self.env.compute_reward(achieved_goal[..., :-1], desired_goal[..., :-1], info)
        cumulative_cost = achieved_goal[..., -1]
        if self.mode == "exponential":
            reward = np.power(np.full(cumulative_cost.shape, 0.1), (cumulative_cost/self.cost_threshold)) * reward
            return reward
        else:
            reward = (cumulative_cost < self.cost_threshold) * reward + (cumulative_cost >= self.cost_threshold) * self.min_step_reward
            return reward

    def observation(self, observation):
        """Add the cost to the observation."""
        if self.dict_space:
            observation["observation"] = np.append(observation["observation"], self.cumulative_cost)
            observation["achieved_goal"] = np.append(observation["achieved_goal"], self.cumulative_cost)
            observation["desired_goal"] = np.append(observation["desired_goal"], self.cumulative_cost)
        else:
            observation = np.append(observation, self.cumulative_cost)
        return observation
