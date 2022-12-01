import functools
from typing import Optional

import tensorflow_probability

from n_dim_reach_env.rl.distributions.tanh_transformed import TanhTransformedDistribution

tfp = tensorflow_probability.substrates.jax
tfd = tfp.distributions

import flax.linen as nn
import jax.numpy as jnp

from n_dim_reach_env.rl.networks.common import default_init


class NormalFixed(nn.Module):
    base_cls: nn.Module
    action_dim: int
    stds: jnp.array
    squash_tanh: bool = False

    @nn.compact
    def __call__(self, inputs, *args, **kwargs) -> tfd.Distribution:
        x = self.base_cls()(inputs, *args, **kwargs)

        means = nn.Dense(self.action_dim, kernel_init=default_init())(x)

        distribution = tfd.MultivariateNormalDiag(loc=means,
                                                  scale_diag=self.stds)

        if self.squash_tanh:
            return TanhTransformedDistribution(distribution)
        else:
            return distribution


TanhNormalFixed = functools.partial(NormalFixed, squash_tanh=True)
