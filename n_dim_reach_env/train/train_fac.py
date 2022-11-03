#!/usr/bin/env python
"""This file describes the training functionality for FAC with HER.

Owner:
    Jakob Thumm (JT)

Contributors:

Changelog:
    3.11.22 JT Created File.
"""
from copy import copy
import gym  # noqa: F401
import numpy as np

import hydra
from hydra.core.config_store import ConfigStore

from gym.wrappers import TimeLimit
from gym import spaces

import safety_gym  # noqa: F401
from n_dim_reach_env.rl.optimization.optimize_hyperparameters import optimize_hyperparameters  # noqa: F401
# from n_dim_reach_env.wrappers.speed_action_wrapper import SpeedActionWrapper
from n_dim_reach_env.conf.config_fac import FACTrainingConfig, EnvConfig

from n_dim_reach_env.rl.train_fac import train_fac

cs = ConfigStore.instance()
cs.store(name="fac_config", node=FACTrainingConfig)


def create_env(env_args: EnvConfig) -> gym.Env:
    """Create the environment.

    Args:
        env_args (EnvConfig): Environment arguments.

    Returns:
        gym.Env: Environment.
    """
    env = gym.make(env_args.id)
    env = TimeLimit(env, env_args.max_ep_len)
    return env


def get_observation_space(env: gym.Env) -> spaces.Box:
    """Get the observation space.

    Args:
        env (gym.Env): Environment.

    Returns:
        spaces.Box: Observation space.
    """
    # If the observation space is of type dict,
    # change the observation space. DroQ cannot handle dicts right now.
    if isinstance(env.observation_space, spaces.Dict):
        observation_space = copy(env.observation_space)
        lows = None
        highs = None
        for key in env.observation_space.spaces:
            if key != "achieved_goal":
                if lows is None:
                    lows = np.array(env.observation_space.spaces[key].low)
                    highs = np.array(env.observation_space.spaces[key].high)
                else:
                    lows = np.append(lows, env.observation_space.spaces[key].low)
                    highs = np.append(highs, env.observation_space.spaces[key].high)
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


@hydra.main(config_path="../conf", config_name="conf_fac")
def main(cfg: FACTrainingConfig):
    """Train a FAC agent on the safety gym environment."""
    print(cfg)
    agent_kwargs = {
        "actor_lr": cfg.fac.actor_lr,
        "critic_lr": cfg.fac.critic_lr,
        "cost_critic_lr": cfg.fac.cost_critic_lr,
        "temp_lr": cfg.fac.temp_lr,
        "lambda_lr": cfg.fac.lambda_lr,
        "hidden_dims": cfg.fac.hidden_dims,
        "discount": cfg.fac.discount,
        "cost_discount": cfg.fac.cost_discount,
        "tau": cfg.fac.tau,
        "num_qs": cfg.fac.num_qs,
        "num_min_qs": cfg.fac.num_min_qs,
        "critic_dropout_rate": cfg.fac.critic_dropout_rate,
        "critic_layer_norm": cfg.fac.critic_layer_norm,
        "target_entropy": cfg.fac.target_entropy,
        "init_temperature": cfg.fac.init_temperature,
        "sampled_backup": cfg.fac.sampled_backup
    }
    learn_args = {
        "seed": cfg.env.seed,
        "delta": cfg.env.delta,
        "agent_kwargs": agent_kwargs,
        "max_ep_len": cfg.env.max_ep_len,
        "max_steps": cfg.train.max_steps,
        "start_steps": cfg.fac.start_steps,
        "squash_output": cfg.fac.squash_output,
        "use_her": cfg.fac.use_her,
        "n_her_samples": cfg.fac.n_her_samples,
        "goal_selection_strategy": cfg.fac.goal_selection_strategy,
        "handle_timeout_termination": cfg.fac.handle_timeout_termination,
        "utd_ratio": cfg.fac.utd_ratio,
        "batch_size": cfg.fac.batch_size,
        "buffer_size": cfg.fac.buffer_size,
        "eval_interval": cfg.train.eval_interval,
        "eval_episodes": cfg.train.eval_episodes,
        "load_checkpoint": cfg.train.load_checkpoint,
        "load_from_folder": cfg.train.load_from_folder,
        "use_tqdm": cfg.train.tqdm,
        "use_wandb": cfg.train.use_wandb,
        "wandb_project": cfg.train.wandb_project,
        "wandb_cfg": cfg,
        "wandb_sync_tensorboard": True,
        "wandb_monitor_gym": True,
        "wandb_save_code": False,
    }
    if not cfg.optimize.optimize:
        env = create_env(cfg.env)
        eval_env = create_env(cfg.env)
        observation_space = get_observation_space(env)
        dict_obs = has_dict_obs(env)
        train_fac(
            env=env,
            eval_env=eval_env,
            observation_space=observation_space,
            dict_obs=dict_obs,
            **learn_args
        )
    else:
        learn_args["use_wandb"] = False
        learn_args["load_checkpoint"] = -1
        learn_args["load_from_folder"] = None
        env_args = copy(cfg.env)
        # del env_args.max_ep_len
        optimize_hyperparameters(
            env_fn=create_env,
            obs_space_fn=get_observation_space,
            has_dict_obs_fn=has_dict_obs,
            env_args=env_args,
            learn_args=learn_args,
            tuning_params=cfg.optimize.tuning_params,
            n_trials=cfg.optimize.n_trials,
            n_startup_trials=cfg.optimize.n_startup_trials,
            n_timesteps=cfg.optimize.n_timesteps,
            n_jobs=cfg.optimize.n_jobs,
            sampler_method=cfg.optimize.sampler_method,
            pruner_method=cfg.optimize.pruner_method,
            n_warmup_steps=cfg.optimize.n_warmup_steps,
            upper_threshold=cfg.optimize.upper_threshold,
            n_eval_episodes=cfg.optimize.n_eval_episodes,
            n_evaluations=cfg.optimize.n_evaluations,
            seed=cfg.optimize.seed,
            use_prior=cfg.optimize.use_prior,
            verbose=cfg.verbose,
        )


if __name__ == "__main__":
    main()
