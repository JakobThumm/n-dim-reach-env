"""This file defines functions to handle dict observation spaces.

Author: Jakob Thumm
Date: 4.11.2022
"""

import numpy as np
from typing import Optional, Dict, List
import gym
from gym import spaces
from copy import copy


def single_obs(
    obs_dict: Dict,
    obs_fn: Optional[callable] = None
) -> np.ndarray:
    """Convert an observation dictionary to a concatenated vector.

    Args:
        obs_dict (Dict): {'observation', 'achieved_goal', 'desired_goal'} dictionary
        obs_fn (Optional, Callable): Optional additional function to convert the observation.
    Returns:
        vec [observation, desired_goal]
    """
    observation = np.concatenate((obs_dict["observation"], obs_dict["desired_goal"]), axis=-1)
    if obs_fn is not None:
        obs_dict = copy(obs_dict)
        obs_dict["observation"] = observation
        return obs_fn(obs_dict)
    else:
        return observation


def goal_dist(
    obs_dict: Dict,
    obs_fn: Optional[callable] = None
) -> np.ndarray:
    """Convert an observation dictionary to a concatenated vector.

    The goal information is saved as the distance between the achieved and desired goal.

    Args:
        obs_dict: {'observation', 'achieved_goal', 'desired_goal'} dictionary
        obs_fn (Optional, Callable): Optional additional function to convert the observation.
    Returns:
        vec [observation, goal_dist]
    """
    observation = np.concatenate((
        obs_dict["observation"],
        np.linalg.norm(obs_dict["achieved_goal"] - obs_dict["desired_goal"], axis=-1)),  axis=-1)
    if obs_fn is not None:
        obs_dict = copy(obs_dict)
        obs_dict["observation"] = observation
        return obs_fn(obs_dict)
    else:
        return observation


def goal_lidar(
    obs_dict: Dict,
    lidar_num_bins: int = 16,
    lidar_max_dist: Optional[float] = 3,
    lidar_exp_gain: Optional[float] = None,
    lidar_alias: bool = True,
    obs_fn: Optional[callable] = None
) -> np.ndarray:
    """Convert an observation dictionary to a concatenated vector.

    The goal information is saved as lidar observations between the achieved and desired goal.
    This represents the pseudo lidar from safety-gym.

    Args:
        obs_dict: {'observation', 'achieved_goal', 'desired_goal'} dictionary
        lidar_num_bins (int): Number of bins for the lidar.
        lidar_max_dist (float): Maximum distance for the lidar.
        lidar_exp_gain (float): Exponential gain for the lidar (only used if lidar_max_dist is None).
        lidar_alias (bool): Whether to use the lidar aliasing.
        obs_fn (Optional, Callable): Optional additional function to convert the observation.
    Returns:
        vec [observation, goal_lidar]
    """
    assert lidar_max_dist is not None or lidar_exp_gain is not None,\
        "Either lidar_max_dist or lidar_exp_gain must be set."
    goal_pos = obs_dict["desired_goal"][..., :2]
    ego_pos = obs_dict["achieved_goal"][..., :2]
    cos_phi = obs_dict["achieved_goal"][..., 2]
    sin_phi = -obs_dict["achieved_goal"][..., 3]
    # New function
    direction_vec_ego = np.stack([cos_phi, sin_phi], axis=-1)
    direction_vec_goal = goal_pos - ego_pos
    dot = np.sum(direction_vec_ego * direction_vec_goal, axis=-1)
    det = direction_vec_ego[..., 0] * direction_vec_goal[..., 1] -\
        direction_vec_ego[..., 1] * direction_vec_goal[..., 0]
    angle = np.arctan2(det, dot)
    angle = np.mod(angle + 2 * np.pi, 2 * np.pi)
    dist = np.linalg.norm(direction_vec_goal, axis=-1)
    # Old function
    """
    ego_mat = np.array([[cos_phi, -sin_phi, 0], [sin_phi, cos_phi, 0], [0, 0, 1]])
    robot_3vec = np.concatenate((ego_pos, [0]))
    robot_mat = ego_mat
    pos_3vec = np.concatenate([goal_pos, [0]])  # Add a zero z-coordinate
    world_3vec = pos_3vec - robot_3vec
    ego_xy = np.matmul(world_3vec, robot_mat)[:2]  # only take XY coordinates
    z = np.complex(ego_xy[0], ego_xy[1])  # X, Y as real, imaginary components
    dist_old = np.abs(z)
    angle_old = np.angle(z) % (np.pi * 2)
    """
    bin_size = (np.pi * 2) / lidar_num_bins
    bin = np.floor(angle / bin_size).astype(int)
    bin_angle = bin_size * bin
    if lidar_max_dist is None:
        sensor = np.exp(-lidar_exp_gain * dist)
    else:
        sensor = np.clip(dist, 0, lidar_max_dist) / lidar_max_dist
    if len(goal_pos.shape) == 1:
        lidar_obs = np.zeros(lidar_num_bins)
        lidar_obs[bin] = sensor
    elif len(goal_pos.shape) == 2:
        lidar_obs = np.zeros((goal_pos.shape[0], lidar_num_bins))
        lidar_obs[np.arange(bin.shape[0]).astype(np.intp), bin.astype(np.intp)] = sensor
    else:
        raise ValueError("Invalid shape of the desired observation.")

    # Aliasing
    if lidar_alias:
        alias = (angle - bin_angle) / bin_size
        eps = 1e-5
        assert np.all(-eps <= alias) and np.all(alias <= 1+eps),\
            f'bad alias {alias}, dist {dist}, angle {angle}, bin {bin}'
        alias = np.clip(alias, 0, 1)
        bin_plus = (bin + 1) % lidar_num_bins
        bin_minus = (bin - 1) % lidar_num_bins
        if len(goal_pos.shape) == 1:
            lidar_obs[bin_plus] = max(lidar_obs[bin_plus], alias * sensor)
            lidar_obs[bin_minus] = max(lidar_obs[bin_minus], (1 - alias) * sensor)
        elif len(goal_pos.shape) == 2:
            lidar_obs[np.arange(bin.shape[0]).astype(np.intp), bin_plus.astype(np.intp)] = alias * sensor
            lidar_obs[np.arange(bin.shape[0]).astype(np.intp), bin_minus.astype(np.intp)] = (1 - alias) * sensor

    observation = np.concatenate((obs_dict["observation"], lidar_obs),  axis=-1)
    if obs_fn is not None:
        obs_dict = copy(obs_dict)
        obs_dict["observation"] = observation
        return obs_fn(obs_dict)
    else:
        return observation


def saute_obs(
    obs_dict: Dict,
    obs_fn: Optional[callable] = None
) -> np.ndarray:
    """Add the max cost of the saute wrapper to the observation.

    ma_cost is assumed to be in the last position of the desired_goal observation.

    Args:
        obs_dict: {'observation', 'achieved_goal', 'desired_goal'} dictionary
        obs_fn (Optional, Callable): Optional additional function to convert the observation.
    Returns:
        vec [observation, max_cost]
    """
    observation = np.concatenate(
        (obs_dict["observation"], (obs_dict["desired_goal"][..., -1])[..., np.newaxis]),
        axis=-1
    )
    if obs_fn is not None:
        obs_dict = copy(obs_dict)
        obs_dict["observation"] = observation
        return obs_fn(obs_dict)
    else:
        return observation


def get_observation_space(
    env: gym.Env,
    goal_observation_type: List[str] = ["default"],
    lidar_num_bins: int = 16
) -> spaces.Box:
    """Get the observation space.

    Args:
        env (gym.Env): Environment.
        goal_observation_type (List[str]): Type of goal observation.
            Can be 'default', 'lidar', 'dist', or 'saute'. Order is important.
        lidar_num_bins (int): Number of bins for the lidar.

    Returns:
        spaces.Box: Observation space.
    """
    # If the observation space is of type dict,
    # change the observation space. DroQ cannot handle dicts right now.
    if isinstance(env.observation_space, spaces.Dict):
        observation_space = copy(env.observation_space)
        lows = np.array(env.observation_space.spaces["observation"].low)
        highs = np.array(env.observation_space.spaces["observation"].high)
        for type in goal_observation_type:
            if type == "default":
                lows = np.append(lows, env.observation_space.spaces["desired_goal"].low)
                highs = np.append(highs, env.observation_space.spaces["desired_goal"].high)
            elif type == "lidar":
                lows = np.append(lows, np.full(lidar_num_bins, 0))
                highs = np.append(highs, np.full(lidar_num_bins, 1))
            elif type == "dist":
                lows = np.append(lows, np.full(1, 0))
                highs = np.append(highs, np.full(1, 1))
            elif type == "saute":
                lows = np.append(lows, np.full(1, 0))
                highs = np.append(highs, np.full(1, env.obs_limit_cost))
            else:
                raise ValueError(f"Unknown goal_observation_type {goal_observation_type}")
        observation_space = spaces.Box(low=lows, high=highs)
    else:
        observation_space = env.observation_space
    return observation_space


def has_dict_obs(env: gym.Env) -> bool:
    """Check if the environment has a dict observation space.

    Args:
        env (gym.Env): Environment.

    Returns:
        bool: True if the observation space is a dict.
    """
    return isinstance(env.observation_space, spaces.Dict)
