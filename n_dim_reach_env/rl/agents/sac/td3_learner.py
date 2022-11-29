"""TD3 algorithm.

Author: Jakob Thumm
Date: 2022-11-02
"""

from functools import partial
from typing import Dict, Optional, Sequence, Tuple

import gym
import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.training.train_state import TrainState

from n_dim_reach_env.rl.agents.agent import Agent
from n_dim_reach_env.rl.networks.temperature import Temperature
from n_dim_reach_env.rl.data.dataset import DatasetDict
from n_dim_reach_env.rl.distributions import TanhNormal
from n_dim_reach_env.rl.networks import MLP, StateActionValue
from n_dim_reach_env.rl.networks.common import soft_target_update


class TD3Learner(Agent):
    r"""The Actor-Critic using TD3.

    Our network architecture is:
        - feature extractor: MLP
        - 2 Q-functions
        - 2 target Q-functions
        - 1 actor

    Args:
        critic_1: The first critic network.
        critic_2: The second critic network.
        target_critic_1: The first target critic network.
        target_critic_2: The second target critic network.
        temp: The temperature network.
        tau: The soft target update coefficient.
        discount: The discount factor.
        target_entropy: The target entropy.
        sampled_backup: Whether to use sampled backups.
    """

    feature_extractor: TrainState
    actor: TrainState
    critic_1: TrainState
    critic_2: TrainState
    target_critic_1: TrainState
    target_critic_2: TrainState
    temp: TrainState
    tau: float
    discount: float
    target_entropy: float
    sampled_backup: bool = struct.field(pytree_node=False)

    @classmethod
    def create(cls,
               seed: int,
               observation_space: gym.Space,
               action_space: gym.Space,
               actor_lr: float = 3e-4,
               critic_lr: float = 3e-4,
               feature_extractor_lr: float = 3e-4,
               temp_lr: float = 3e-4,
               feature_extractor_dims: Sequence[int] = (256, 256),
               network_dims: Sequence[int] = (64, 64),
               discount: float = 0.99,
               tau: float = 0.005,
               target_entropy: Optional[float] = None,
               init_temperature: float = 1.0,
               sampled_backup: bool = True):
        r"""Create the TD3 agent and its optimizers.

        Args:
            seed (int): The random seed.
            observation_space (gym.Space): The observation space.
            action_space (gym.Space): The action space.
            actor_lr (float, optional): The learning rate for the actor. Defaults to 3e-4.
            critic_lr (float, optional): The learning rate for the critic. Defaults to 3e-4.
            feature_extractor_lr (float, optional): The learning rate for the feature extractor. Defaults to 3e-4.
            temp_lr (float, optional): The learning rate for the temperature. Defaults to 3e-4.
            feature_extractor_dims (Sequence[int], optional): The dimensions of the feature extractor.
                Defaults to (256, 256).
            network_dims (Sequence[int], optional): The dimensions of the policy and critic networks.
                Defaults to (64, 64).
            discount (float, optional): The discount factor. Defaults to 0.99.
            tau (float, optional): The soft target update coefficient. Defaults to 0.005.
            target_entropy (Optional[float], optional): The target entropy. Defaults to None.
            init_temperature (float, optional): The initial temperature. Defaults to 1.0.
            sampled_backup (bool, optional): Whether to use sampled backups. Defaults to True.
        """
        action_dim = action_space.shape[-1]
        observations = observation_space.sample()
        actions = action_space.sample()

        if target_entropy is None:
            target_entropy = -action_dim / 2

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, feature_extractor_key, critic_1_key, critic_2_key, temp_key = jax.random.split(rng, 6)
        # Feature extractor
        feature_extractor_def = MLP(hidden_dims=feature_extractor_dims,
                                    activate_final=True)
        feature_extractor_params = feature_extractor_def.init(feature_extractor_key, observations)['params']
        feature_extractor = TrainState.create(apply_fn=feature_extractor_def.apply,
                                              params=feature_extractor_params,
                                              tx=optax.adam(learning_rate=feature_extractor_lr))
        features = feature_extractor.apply_fn({'params': feature_extractor_params}, observations)
        # Actor
        actor_base_cls = partial(MLP,
                                 hidden_dims=network_dims,
                                 activate_final=True)
        actor_def = TanhNormal(actor_base_cls, action_dim)
        actor_params = actor_def.init(actor_key, features)['params']
        actor = TrainState.create(apply_fn=actor_def.apply,
                                  params=actor_params,
                                  tx=optax.adam(learning_rate=actor_lr))
        # Critic 1
        critic_1_base_cls = partial(MLP,
                                    hidden_dims=network_dims,
                                    activate_final=False)
        critic_1_def = StateActionValue(critic_1_base_cls)
        critic_1_params = critic_1_def.init(critic_1_key, features, actions)['params']
        critic_1 = TrainState.create(apply_fn=critic_1_def.apply,
                                     params=critic_1_params,
                                     tx=optax.adam(learning_rate=critic_lr))
        # Target critic 1
        target_critic_1_def = StateActionValue(critic_1_base_cls)
        target_critic_1 = TrainState.create(apply_fn=target_critic_1_def.apply,
                                            params=critic_1_params,
                                            tx=optax.GradientTransformation(
                                              lambda _: None, lambda _: None))
        # Critic 2
        critic_2_base_cls = partial(MLP,
                                    hidden_dims=network_dims,
                                    activate_final=False)
        critic_2_def = StateActionValue(critic_2_base_cls)
        critic_2_params = critic_2_def.init(critic_2_key, features, actions)['params']
        critic_2 = TrainState.create(apply_fn=critic_2_def.apply,
                                     params=critic_2_params,
                                     tx=optax.adam(learning_rate=critic_lr))
        # Target critic 1
        target_critic_2_def = StateActionValue(critic_2_base_cls)
        target_critic_2 = TrainState.create(apply_fn=target_critic_2_def.apply,
                                            params=critic_2_params,
                                            tx=optax.GradientTransformation(
                                              lambda _: None, lambda _: None))
        # Temperature
        temp_def = Temperature(init_temperature)
        temp_params = temp_def.init(temp_key)['params']
        temp = TrainState.create(apply_fn=temp_def.apply,
                                 params=temp_params,
                                 tx=optax.adam(learning_rate=temp_lr))
        return cls(rng=rng,
                   feature_extractor=feature_extractor,
                   actor=actor,
                   critic_1=critic_1,
                   critic_2=critic_2,
                   target_critic_1=target_critic_1,
                   target_critic_2=target_critic_2,
                   temp=temp,
                   tau=tau,
                   discount=discount,
                   target_entropy=target_entropy,
                   sampled_backup=sampled_backup)

    @staticmethod
    def update_actor(agent,
                     batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        r"""Update the actor.

        The actor loss uses the critic 1 value.

        Loss = \alpha * log(\pi(a|s)) - Q_1(s, \pi(a|s))
        """
        rng, critic_1_key, feature_extractor_key, dist_key = jax.random.split(agent.rng, 4)

        def feature_extractor_loss_fn(feature_extractor_params) -> jnp.ndarray:
            # This action
            features = agent.feature_extractor.apply_fn({'params': feature_extractor_params},
                                                        batch['observations'],
                                                        True,
                                                        rngs={'dropout': feature_extractor_key})
            dist = agent.actor.apply_fn({'params': agent.actor.params}, features)
            actions = dist.sample(seed=dist_key)
            log_probs = dist.log_prob(actions)
            alpha = agent.temp.apply_fn({'params': agent.temp.params})
            q_1 = agent.critic_1.apply_fn({'params': agent.critic_1.params},
                                          features,
                                          actions,
                                          True,
                                          rngs={'dropout': critic_1_key})
            feat_loss = (alpha * log_probs - q_1).mean()
            return feat_loss

        def actor_loss_fn(actor_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            features = agent.feature_extractor.apply_fn({'params': agent.feature_extractor.params},
                                                        batch['observations'],
                                                        True,
                                                        rngs={'dropout': feature_extractor_key})
            dist = agent.actor.apply_fn({'params': actor_params}, features)
            actions = dist.sample(seed=dist_key)
            log_probs = dist.log_prob(actions)
            alpha = agent.temp.apply_fn({'params': agent.temp.params})
            q_1 = agent.critic_1.apply_fn({'params': agent.critic_1.params},
                                          features,
                                          actions,
                                          True,
                                          rngs={'dropout': critic_1_key})
            actor_loss = (alpha * log_probs - q_1).mean()
            return actor_loss, {
                'actor_loss': actor_loss,
                'entropy': -log_probs.mean(),
            }

        feature_grads = jax.grad(feature_extractor_loss_fn)(agent.feature_extractor.params)
        actor_grads, actor_info = jax.grad(actor_loss_fn, has_aux=True)(agent.actor.params)

        actor = agent.actor.apply_gradients(grads=actor_grads)
        feature_extractor = agent.feature_extractor.apply_gradients(grads=feature_grads)

        agent = agent.replace(
            actor=actor,
            feature_extractor=feature_extractor,
            rng=rng
        )
        return agent, actor_info

    @staticmethod
    def update_temperature(agent,
                           entropy: float) -> Tuple[Agent, Dict[str, float]]:
        r"""Update the temperature.

        Loss = \alpha * (target_entropy - entropy)
        """
        def temperature_loss_fn(temp_params):
            temperature = agent.temp.apply_fn({'params': temp_params})
            temp_loss = temperature * (entropy - agent.target_entropy).mean()
            return temp_loss, {
                'temperature': temperature,
                'temperature_loss': temp_loss
            }

        grads, temp_info = jax.grad(temperature_loss_fn,
                                    has_aux=True)(agent.temp.params)
        temp = agent.temp.apply_gradients(grads=grads)

        agent = agent.replace(temp=temp)

        return agent, temp_info

    @staticmethod
    def update_critic(
        agent,
        batch: DatasetDict,
        update_targets: bool = True
    ) -> Tuple[TrainState, Dict[str, float]]:
        r"""Update the critic(s).

        The target for updating the critics is taken from the MINIMUM over target critics.
        The minimum is used to avoid overapproximation of the true value function.

        y = r + gamma * mask * min_{i=1..2}(Q^{i}_{target}(s', \pi(s')),
            where mask = 1 if not done else 0 and M=N if not specified otherwise.
        if sampled_backup:
            y -= gamma * mask * \alpha * log(\pi(a|s'))
        Loss = 0.5 * (Q(s, a) - y)^2
        """
        if agent.sampled_backup:
            rng, critic_1_key, critic_2_key, feature_extractor_key, dist_key = jax.random.split(agent.rng, 5)
        else:
            rng, critic_1_key, critic_2_key, feature_extractor_key = jax.random.split(agent.rng, 4)

        next_features = agent.feature_extractor.apply_fn({'params': agent.feature_extractor.params},
                                                         batch['next_observations'],
                                                         True,
                                                         rngs={'dropout': feature_extractor_key})
        next_dist = agent.actor.apply_fn({'params': agent.actor.params},
                                         next_features)
        if agent.sampled_backup:
            next_actions = next_dist.sample(seed=dist_key)
        else:
            next_actions = next_dist.mode()
        next_q_1 = agent.target_critic_1.apply_fn({'params': agent.target_critic_1.params},
                                                  next_features,
                                                  next_actions,
                                                  True,
                                                  rngs={'dropout': critic_1_key})
        next_q_2 = agent.target_critic_2.apply_fn({'params': agent.target_critic_2.params},
                                                  next_features,
                                                  next_actions,
                                                  True,
                                                  rngs={'dropout': critic_2_key})
        next_q = jnp.min(jnp.stack([next_q_1, next_q_2]), axis=0)
        y = batch['rewards'] + agent.discount * batch['masks'] * next_q
        if agent.sampled_backup:
            next_log_probs = next_dist.log_prob(next_actions)
            alpha = agent.temp.apply_fn({'params': agent.temp.params})
            target_backup = - agent.discount * batch['masks'] * alpha * next_log_probs
            y += target_backup

        def feature_extractor_loss_fn(feature_extractor_params) -> jnp.ndarray:
            # This action
            features = agent.feature_extractor.apply_fn({'params': feature_extractor_params},
                                                        batch['observations'],
                                                        True,
                                                        rngs={'dropout': feature_extractor_key})
            q_1 = agent.critic_1.apply_fn({'params': agent.critic_1.params},
                                          features,
                                          batch['actions'],
                                          True,
                                          rngs={'dropout': critic_1_key})
            q_2 = agent.critic_2.apply_fn({'params': agent.critic_2.params},
                                          features,
                                          batch['actions'],
                                          True,
                                          rngs={'dropout': critic_2_key})
            feat_loss = (1/2 * ((y - q_1)**2 + (y - q_2)**2)).mean()
            return feat_loss

        def critic_loss_fn(
            critic_1_params,
            critic_2_params
        ) -> Tuple[jnp.ndarray, Dict[str, float]]:
            # This action
            features = agent.feature_extractor.apply_fn({'params': agent.feature_extractor.params},
                                                        batch['observations'],
                                                        True,
                                                        rngs={'dropout': feature_extractor_key})
            q_1 = agent.critic_1.apply_fn({'params': critic_1_params},
                                          features,
                                          batch['actions'],
                                          True,
                                          rngs={'dropout': critic_1_key})
            q_2 = agent.critic_2.apply_fn({'params': critic_2_params},
                                          features,
                                          batch['actions'],
                                          True,
                                          rngs={'dropout': critic_2_key})
            critic_loss = (1/2 * ((y - q_1)**2 + (y - q_2)**2)).mean()
            return critic_loss, {'critic_loss': critic_loss,
                                 'q_1': q_1.mean(),
                                 'q_2': q_2.mean(),
                                 'batch_reward': batch['rewards'].mean()}

        feature_grads = jax.grad(feature_extractor_loss_fn)(agent.feature_extractor.params)
        critic_grads, info = jax.grad(critic_loss_fn, has_aux=True)(
            agent.critic_1.params,
            agent.critic_2.params,
        )
        feature_extractor = agent.feature_extractor.apply_gradients(grads=feature_grads)
        critic_1 = agent.critic_1.apply_gradients(grads=critic_grads)
        critic_2 = agent.critic_2.apply_gradients(grads=critic_grads)
        if update_targets:
            target_critic_1_params = soft_target_update(critic_1.params,
                                                        agent.target_critic_1.params,
                                                        agent.tau)
            target_critic_2_params = soft_target_update(critic_2.params,
                                                        agent.target_critic_2.params,
                                                        agent.tau)
            target_critic_1 = agent.target_critic_1.replace(
                params=target_critic_1_params)

            target_critic_2 = agent.target_critic_2.replace(
                params=target_critic_2_params)
            new_agent = agent.replace(
                feature_extractor=feature_extractor,
                critic_1=critic_1,
                critic_2=critic_2,
                target_critic_1=target_critic_1,
                target_critic_2=target_critic_2,
                rng=rng
            )
        else:
            new_agent = agent.replace(
                feature_extractor=feature_extractor,
                critic_1=critic_1,
                critic_2=critic_2,
                rng=rng
            )
        return new_agent, info

    @partial(jax.jit, static_argnames=['utd_ratio', 'update_actor', 'update_targets'])
    def update(self,
               batch: DatasetDict,
               utd_ratio: int,
               update_actor: bool = True,
               update_targets: bool = True) -> Tuple[Agent, Dict[str, float]]:
        """Update the agent."""
        critic_info = {}
        actor_info = {}
        temp_info = {}

        new_agent = self
        for i in range(utd_ratio):

            def slice(x):
                assert x.shape[0] % utd_ratio == 0
                batch_size = x.shape[0] // utd_ratio
                return x[batch_size * i:batch_size * (i + 1)]

            mini_batch = jax.tree_util.tree_map(slice, batch)
            update_critic_target = update_targets and i == utd_ratio - 1
            new_agent, critic_info = self.update_critic(new_agent, mini_batch, update_critic_target)

        if update_actor:
            new_agent, actor_info = self.update_actor(new_agent, mini_batch)
            new_agent, temp_info = self.update_temperature(new_agent, actor_info['entropy'])

        return new_agent, {**actor_info, **critic_info, **temp_info}
