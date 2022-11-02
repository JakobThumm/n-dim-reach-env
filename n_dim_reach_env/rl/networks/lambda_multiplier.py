"""This file defines the lambda multiplier network."""

import flax.linen as nn
import jax.numpy as jnp

from n_dim_reach_env.rl.networks.common import default_init


class LambdaMultiplier(nn.Module):
    """The lambda multiplier network."""

    base_cls: nn.Module

    @nn.compact
    def __call__(self, observations: jnp.ndarray, *args,
                 **kwargs) -> jnp.ndarray:
        """Forward pass of the lambda multiplier network."""
        outputs = self.base_cls()(observations, *args, **kwargs)

        value = nn.Dense(1, kernel_init=default_init())(outputs)

        return nn.softplus(value)
