"""Normalize observations to have zero mean and unit variance."""

import numpy as np
import gym
from n_dim_reach_env.rl.data.dataset import DatasetDict
from flax.core import frozen_dict


def normalize_observation(
    observation: np.ndarray,
    observation_space: gym.spaces.Box
) -> np.ndarray:
    """Normalize an observation to have zero mean and unit variance.

    Args:
        observation (np.ndarray): Observation to normalize.
        observation_space (gym.spaces.Box): Observation space.

    Returns:
        np.ndarray: Normalized observation.
    """
    return (observation - observation_space.low) / (
        observation_space.high - observation_space.low
    )


def normalize_batch(
    batch: DatasetDict,
    observation_space: gym.spaces.Box
) -> DatasetDict:
    """Normalize a training batch with observations to have zero mean and unit variance.

    Args:
        batch (DatasetDict): Observations to normalize.
        observation_space (gym.spaces.Box): Observation space.
    Returns:
        DatasetDict: Normalized batch.
    """
    batch = batch.unfreeze()
    batch["observations"] = normalize_observation(batch["observations"], observation_space)
    batch["next_observations"] = normalize_observation(batch["next_observations"], observation_space)
    return frozen_dict.freeze(batch)
