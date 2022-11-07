"""Feasible Actor-Critic (FAC).

From: https://arxiv.org/pdf/2105.10682.pdf
FAC is a constrained RL algorithm that learns state-dependent lambda values.
FAC is based on SAC, but with a state-dependent constraint function.

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
from n_dim_reach_env.rl.agents.sac.temperature import Temperature
from n_dim_reach_env.rl.data.dataset import DatasetDict
from n_dim_reach_env.rl.distributions import TanhNormal
from n_dim_reach_env.rl.networks import MLP, Ensemble, StateActionValue, LambdaMultiplier
from n_dim_reach_env.rl.networks.common import soft_target_update


class FACLearner(Agent):
    r"""The Feasible Actor-Critic (FAC) learner.

    Args:
        critic: The critic network.
        target_critic: The target critic network.
        cost_critic: The constraint critic network.
        target_cost_critic: The target constraint critic network.
        temp: The temperature network.
        lam: The lambda multiplier network.
        tau: The soft target update coefficient.
        cost_discount: The discount factor for the constraint critic.
        discount: The discount factor.
        target_entropy: The target entropy.
        num_qs: The number of Q networks in the critic and cost critic ensembles.
        num_min_qs: See M in RedQ https://arxiv.org/abs/2101.05982
        sampled_backup: Whether to use sampled backups.
        delta: The cost ratio threshold for constraint violation. Qc(s,a) \leq \delta
    """

    critic: TrainState
    target_critic: TrainState
    cost_critic: TrainState
    target_cost_critic: TrainState
    temp: TrainState
    lam: TrainState
    tau: float
    discount: float
    cost_discount: float
    target_entropy: float
    num_qs: int = struct.field(pytree_node=False)
    num_min_qs: Optional[int] = struct.field(
        pytree_node=False)  # See M in RedQ https://arxiv.org/abs/2101.05982
    sampled_backup: bool = struct.field(pytree_node=False)
    delta: float = 0.1

    @classmethod
    def create(cls,
               seed: int,
               observation_space: gym.Space,
               action_space: gym.Space,
               delta: float = 10,
               actor_lr: float = 3e-4,
               critic_lr: float = 3e-4,
               cost_critic_lr: float = 3e-4,
               lambda_lr: float = 3e-4,
               temp_lr: float = 3e-4,
               hidden_dims: Sequence[int] = (256, 256),
               cost_discount: float = 0.99,
               discount: float = 0.99,
               tau: float = 0.005,
               num_qs: int = 2,
               num_min_qs: Optional[int] = None,
               critic_dropout_rate: Optional[float] = None,
               critic_layer_norm: bool = False,
               target_entropy: Optional[float] = None,
               init_temperature: float = 1.0,
               sampled_backup: bool = True):
        r"""Create the FAC agent and its optimizers.

        Args:
            seed (int): The random seed.
            observation_space (gym.Space): The observation space.
            action_space (gym.Space): The action space.
            delta (float, optional): The cost ratio threshold for constraint violation. Qc(s,a) \leq \delta.
                Defaults to 10.
            actor_lr (float, optional): The learning rate for the actor. Defaults to 3e-4.
            critic_lr (float, optional): The learning rate for the critic. Defaults to 3e-4.
            cost_critic_lr (float, optional): The learning rate for the cost critic. Defaults to 3e-4.
            lambda_lr (float, optional): The learning rate for the lambda multiplier. Defaults to 3e-4.
            temp_lr (float, optional): The learning rate for the temperature. Defaults to 3e-4.
            hidden_dims (Sequence[int], optional): The hidden dimensions of the actor and critic networks.
                Defaults to (256, 256).
            cost_discount (float, optional): The discount factor for the constraint critic. Defaults to 0.99.
            discount (float, optional): The discount factor. Defaults to 0.99.
            tau (float, optional): The soft target update coefficient. Defaults to 0.005.
            num_qs (int, optional): The number of Q networks in the critic and cost critic ensembles. Defaults to 2.
            num_min_qs (Optional[int], optional): See M in RedQ https://arxiv.org/abs/2101.05982. Defaults to None.
            critic_dropout_rate (Optional[float], optional): The dropout rate for the critic networks. Defaults to None.
            critic_layer_norm (bool, optional): Whether to use layer normalization in the critic networks.
                Defaults to False.
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
        rng, actor_key, critic_key, cost_critic_key, temp_key, lambda_key = jax.random.split(rng, 6)
        # Actor
        actor_base_cls = partial(MLP,
                                 hidden_dims=hidden_dims,
                                 activate_final=True)
        actor_def = TanhNormal(actor_base_cls, action_dim)
        actor_params = actor_def.init(actor_key, observations)['params']
        actor = TrainState.create(apply_fn=actor_def.apply,
                                  params=actor_params,
                                  tx=optax.adam(learning_rate=actor_lr))
        # Critic(s)
        critic_base_cls = partial(MLP,
                                  hidden_dims=hidden_dims,
                                  activate_final=True,
                                  dropout_rate=critic_dropout_rate,
                                  use_layer_norm=critic_layer_norm)
        critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
        critic_def = Ensemble(critic_cls, num=num_qs)
        critic_params = critic_def.init(critic_key, observations,
                                        actions)['params']
        critic = TrainState.create(apply_fn=critic_def.apply,
                                   params=critic_params,
                                   tx=optax.adam(learning_rate=critic_lr))
        # Target critic(s)
        target_critic_def = Ensemble(critic_cls, num=num_min_qs or num_qs)
        target_critic = TrainState.create(apply_fn=target_critic_def.apply,
                                          params=critic_params,
                                          tx=optax.GradientTransformation(
                                              lambda _: None, lambda _: None))
        # Cost critic(s)
        cost_critic_base_cls = partial(MLP,
                                       hidden_dims=hidden_dims,
                                       activate_final=True,
                                       dropout_rate=critic_dropout_rate,
                                       use_layer_norm=critic_layer_norm)
        cost_critic_cls = partial(StateActionValue, base_cls=cost_critic_base_cls)
        cost_critic_def = Ensemble(cost_critic_cls, num=num_qs)
        cost_critic_params = cost_critic_def.init(cost_critic_key, observations,
                                                  actions)['params']
        cost_critic = TrainState.create(apply_fn=cost_critic_def.apply,
                                        params=cost_critic_params,
                                        tx=optax.adam(learning_rate=cost_critic_lr))
        # Cost target critic(s)
        target_cost_critic_def = Ensemble(cost_critic_cls, num=num_min_qs or num_qs)
        target_cost_critic = TrainState.create(apply_fn=target_cost_critic_def.apply,
                                               params=cost_critic_params,
                                               tx=optax.GradientTransformation(
                                                   lambda _: None, lambda _: None))
        # Temperature
        temp_def = Temperature(init_temperature)
        temp_params = temp_def.init(temp_key)['params']
        temp = TrainState.create(apply_fn=temp_def.apply,
                                 params=temp_params,
                                 tx=optax.adam(learning_rate=temp_lr))
        # Lambda
        lambda_base_cls = partial(MLP,
                                  hidden_dims=hidden_dims,
                                  activate_final=True,
                                  dropout_rate=0.0,
                                  use_layer_norm=False)
        lambda_def = LambdaMultiplier(base_cls=lambda_base_cls)
        lambda_params = lambda_def.init(lambda_key, observations)['params']
        lam = TrainState.create(apply_fn=lambda_def.apply,
                                params=lambda_params,
                                tx=optax.adam(learning_rate=lambda_lr))
        return cls(rng=rng,
                   actor=actor,
                   critic=critic,
                   target_critic=target_critic,
                   cost_critic=cost_critic,
                   target_cost_critic=target_cost_critic,
                   temp=temp,
                   lam=lam,
                   target_entropy=target_entropy,
                   tau=tau,
                   discount=discount,
                   cost_discount=cost_discount,
                   num_qs=num_qs,
                   num_min_qs=num_min_qs,
                   sampled_backup=sampled_backup,
                   delta=delta)

    @staticmethod
    def update_actor(agent,
                     batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        r"""Update the actor.

        Loss = \alpha * log(\pi(a|s)) - Q(s, \pi(a|s)) + \lambda(s) * (Q_c(s, \pi(a|s)) - d)
        """
        rng, action_key, critic_key, cost_critic_key, lambda_key = jax.random.split(agent.rng, 5)

        def actor_loss_fn(
                actor_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            dist = agent.actor.apply_fn({'params': actor_params},
                                        batch['observations'])
            actions = dist.sample(seed=action_key)
            log_probs = dist.log_prob(actions)
            alpha = agent.temp.apply_fn({'params': agent.temp.params})
            qs = agent.critic.apply_fn({'params': agent.critic.params},
                                       batch['observations'],
                                       actions,
                                       True,
                                       rngs={'dropout': critic_key})  # training=True
            q = qs.min(axis=0)
            qcs = agent.cost_critic.apply_fn({'params': agent.cost_critic.params},
                                             batch['observations'],
                                             actions,
                                             True,
                                             rngs={'dropout': cost_critic_key})
            qc = qcs.min(axis=0)
            lambda_val = agent.lam.apply_fn({'params': agent.lam.params},
                                            batch['observations'],
                                            True,
                                            rngs={'dropout': lambda_key})
            actor_loss = (alpha * log_probs - q + lambda_val * (qc - agent.delta)).mean()
            return actor_loss, {
                'actor_loss': actor_loss,
                'entropy': -log_probs.mean(),
                'alpha': alpha.mean(),
            }

        grads, actor_info = jax.grad(actor_loss_fn,
                                     has_aux=True)(agent.actor.params)
        actor = agent.actor.apply_gradients(grads=grads)

        agent = agent.replace(actor=actor, rng=rng)

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
            agent, batch: DatasetDict) -> Tuple[TrainState, Dict[str, float]]:
        r"""Update the critic(s).

        Q_target = r + gamma * mask * Q(s', \pi(s')),
            where mask = 1 if not done else 0.
        if sampled_backup:
            Q_target -= gamma * mask * \alpha * log(\pi(a|s'))
        Loss = 0.5 * (Q(s, a) - Q_target)^2
        """
        dist = agent.actor.apply_fn({'params': agent.actor.params},
                                    batch['next_observations'])

        rng = agent.rng

        if agent.sampled_backup:
            key, rng = jax.random.split(rng)
            next_actions = dist.sample(seed=key)
        else:
            next_actions = dist.mode()

        key2, rng = jax.random.split(rng)

        if agent.num_min_qs is None:
            target_params = agent.target_critic.params
        else:
            all_indx = jnp.arange(0, agent.num_qs)
            rng, key = jax.random.split(rng)
            indx = jax.random.choice(key,
                                     a=all_indx,
                                     shape=(agent.num_min_qs, ),
                                     replace=False)
            target_params = jax.tree_util.tree_map(lambda param: param[indx],
                                                   agent.target_critic.params)

        next_qs = agent.target_critic.apply_fn({'params': target_params},
                                               batch['next_observations'],
                                               next_actions,
                                               True,
                                               rngs={'dropout':
                                                     key2})  # training=True
        next_q = next_qs.min(axis=0)

        target_q = batch['rewards'] + agent.discount * batch['masks'] * next_q

        if agent.sampled_backup:
            next_log_probs = dist.log_prob(next_actions)
            alpha = agent.temp.apply_fn({'params': agent.temp.params})
            target_q -= agent.discount * batch['masks'] * alpha * next_log_probs

        key3, rng = jax.random.split(rng)

        def critic_loss_fn(
                critic_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            qs = agent.critic.apply_fn({'params': critic_params},
                                       batch['observations'],
                                       batch['actions'],
                                       True,
                                       rngs={'dropout': key3})  # training=True
            critic_loss = (0.5 * (qs - target_q)**2).mean()
            return critic_loss, {'critic_loss': critic_loss, 'q': qs.mean()}

        grads, info = jax.grad(critic_loss_fn,
                               has_aux=True)(agent.critic.params)
        critic = agent.critic.apply_gradients(grads=grads)

        target_critic_params = soft_target_update(critic.params,
                                                  agent.target_critic.params,
                                                  agent.tau)
        target_critic = agent.target_critic.replace(
            params=target_critic_params)

        new_agent = agent.replace(critic=critic,
                                  target_critic=target_critic,
                                  rng=rng)

        return new_agent, info

    @staticmethod
    def update_cost_critic(
            agent, batch: DatasetDict) -> Tuple[TrainState, Dict[str, float]]:
        r"""Update the cost critic(s).

        Qc_target = c + gamma_c * mask * Qc(s', \pi(s')),
            where mask = 1 if not done else 0.
        Loss = 0.5 * (Qc(s, a) - Qc_target)^2
        """
        dist = agent.actor.apply_fn({'params': agent.actor.params},
                                    batch['next_observations'])

        rng = agent.rng

        if agent.sampled_backup:
            key, rng = jax.random.split(rng)
            next_actions = dist.sample(seed=key)
        else:
            next_actions = dist.mode()

        key2, rng = jax.random.split(rng)

        if agent.num_min_qs is None:
            target_params = agent.target_cost_critic.params
        else:
            all_indx = jnp.arange(0, agent.num_qs)
            rng, key = jax.random.split(rng)
            indx = jax.random.choice(key,
                                     a=all_indx,
                                     shape=(agent.num_min_qs, ),
                                     replace=False)
            target_params = jax.tree_util.tree_map(lambda param: param[indx],
                                                   agent.target_cost_critic.params)

        next_qcs = agent.target_cost_critic.apply_fn({'params': target_params},
                                                     batch['next_observations'],
                                                     next_actions,
                                                     True,
                                                     rngs={'dropout': key2})  # training=True
        next_qc = next_qcs.min(axis=0)

        target_qc = batch['costs'] + agent.cost_discount * batch['masks'] * next_qc

        key3, rng = jax.random.split(rng)

        def cost_critic_loss_fn(
                cost_critic_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            qcs = agent.cost_critic.apply_fn({'params': cost_critic_params},
                                             batch['observations'],
                                             batch['actions'],
                                             True,
                                             rngs={'dropout': key3})  # training=True
            cost_critic_loss = (0.5 * (qcs - target_qc)**2).mean()
            return cost_critic_loss, {'cost_critic_loss': cost_critic_loss,
                                      'qc': qcs.mean(),
                                      'batch_costs': batch['costs'].mean()}

        grads, info = jax.grad(cost_critic_loss_fn,
                               has_aux=True)(agent.cost_critic.params)
        cost_critic = agent.cost_critic.apply_gradients(grads=grads)

        target_cost_critic_params = soft_target_update(cost_critic.params,
                                                       agent.target_cost_critic.params,
                                                       agent.tau)
        target_cost_critic = agent.target_cost_critic.replace(
            params=target_cost_critic_params)

        new_agent = agent.replace(cost_critic=cost_critic,
                                  target_cost_critic=target_cost_critic,
                                  rng=rng)
        return new_agent, info

    @staticmethod
    def update_lambda(agent,
                      batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        r"""Update the lambda multiplicator.

        Loss = - lambda(s) * (Qc(s, \pi(s)) - delta)
        """
        rng, action_key, cost_critic_key, cost_critic_dropout_key, lambda_key = jax.random.split(agent.rng, 5)
        dist = agent.actor.apply_fn({'params': agent.actor.params},
                                    batch['observations'])
        if agent.sampled_backup:
            actions = dist.sample(seed=action_key)
        else:
            actions = dist.mode()

        if agent.num_min_qs is None:
            target_params = agent.target_cost_critic.params
        else:
            all_indx = jnp.arange(0, agent.num_qs)
            indx = jax.random.choice(cost_critic_key,
                                     a=all_indx,
                                     shape=(agent.num_min_qs, ),
                                     replace=False)
            target_params = jax.tree_util.tree_map(lambda param: param[indx],
                                                   agent.target_cost_critic.params)
        qcs = agent.target_cost_critic.apply_fn({'params': target_params},
                                                batch['observations'],
                                                actions,
                                                True,
                                                rngs={'dropout': cost_critic_dropout_key})  # training=True
        qc = qcs.min(axis=0)

        def lambda_loss_fn(
                lambda_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            lambda_val = agent.lam.apply_fn({'params': lambda_params},
                                            batch['observations'],
                                            True,
                                            rngs={'dropout': lambda_key})
            lambda_loss = (-lambda_val * (qc - agent.delta)).mean()
            return lambda_loss, {
                'lambda_loss': lambda_loss,
                'lambda_val': lambda_val.mean()
            }

        grads, lambda_info = jax.grad(lambda_loss_fn,
                                      has_aux=True)(agent.lam.params)
        lam = agent.lam.apply_gradients(grads=grads)

        agent = agent.replace(lam=lam, rng=rng)

        return agent, lambda_info

    @partial(jax.jit,
             static_argnames=['utd_ratio', 'update_actor', 'update_lambda'])
    def update(self,
               batch: DatasetDict,
               utd_ratio: int,
               update_actor: bool = True,
               update_lambda: bool = True) -> Tuple[Agent, Dict[str, float]]:
        """Update the agent."""
        critic_info = {}
        cost_critic_info = {}
        actor_info = {}
        temp_info = {}
        lambda_info = {}

        new_agent = self
        for i in range(utd_ratio):

            def slice(x):
                assert x.shape[0] % utd_ratio == 0
                batch_size = x.shape[0] // utd_ratio
                return x[batch_size * i:batch_size * (i + 1)]

            mini_batch = jax.tree_util.tree_map(slice, batch)
            new_agent, critic_info = self.update_critic(new_agent, mini_batch)
            new_agent, cost_critic_info = self.update_cost_critic(new_agent, mini_batch)

        if update_actor:
            new_agent, actor_info = self.update_actor(new_agent, mini_batch)
            new_agent, temp_info = self.update_temperature(new_agent,
                                                           actor_info['entropy'])
        if update_lambda:
            new_agent, lambda_info = self.update_lambda(new_agent, mini_batch)

        return new_agent, {**actor_info, **critic_info, **cost_critic_info, **temp_info, **lambda_info}
