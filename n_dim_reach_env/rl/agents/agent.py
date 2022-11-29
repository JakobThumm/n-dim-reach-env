"""This file describes a base agent in JAX+FLAX."""

from functools import partial

import jax
import numpy as np
from flax import struct
from flax.training.train_state import TrainState

from n_dim_reach_env.rl.types import PRNGKey


@partial(jax.jit, static_argnames='apply_fn')
def _sample_actions(rng, apply_fn, params,
                    observations: np.ndarray) -> np.ndarray:
    key, rng = jax.random.split(rng)
    dist = apply_fn({'params': params}, observations)
    return dist.sample(seed=key), rng


@partial(jax.jit, static_argnames='apply_fn')
def _eval_actions(apply_fn, params, observations: np.ndarray) -> np.ndarray:
    dist = apply_fn({'params': params}, observations)
    return dist.mode()


class Agent(struct.PyTreeNode):
    """A base class for agents."""

    actor: TrainState
    rng: PRNGKey

    def eval_actions(self, observations: np.ndarray) -> np.ndarray:
        """Sample actions from the policy without noise."""
        if hasattr(self, "use_feature_extractor") and self.use_feature_extractor:
            features = self.feature_extractor.apply_fn({'params': self.feature_extractor.params}, observations)
        else:
            features = observations
        actions = _eval_actions(self.actor.apply_fn, self.actor.params, features)
        return np.asarray(actions)

    def sample_actions(self, observations: np.ndarray) -> np.ndarray:
        """Samples actions from the policy with noise."""
        if hasattr(self, "use_feature_extractor") and self.use_feature_extractor:
            features = self.feature_extractor.apply_fn({'params': self.feature_extractor.params}, observations)
        else:
            features = observations
        actions, new_rng = _sample_actions(self.rng, self.actor.apply_fn, self.actor.params, features)
        return np.asarray(actions), self.replace(rng=new_rng)
