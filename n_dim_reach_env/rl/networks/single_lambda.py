"""Network class for non-state dependent lambda network.

Author: Jakob Thumm
Date: 11.11.2022
"""

import flax.linen as nn
import jax.numpy as jnp


class SingleLambda(nn.Module):
    """Network class for non-state dependent lambda network."""

    initial_lambda: float = 100.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        """Forward pass of the lambda network."""
        lam = self.param('lam', init_fn=lambda key: jnp.full((), self.initial_lambda))
        return nn.softplus(lam)
