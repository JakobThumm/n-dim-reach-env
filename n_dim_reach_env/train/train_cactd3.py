#!/usr/bin/env python
"""This file describes the training functionality for CAC TD3 with HER.

Owner:
    Jakob Thumm (JT)

Contributors:

Changelog:
    3.11.22 JT Created File.
"""
from copy import copy
import gym  # noqa: F401

import hydra
from hydra.core.config_store import ConfigStore

from gym.wrappers import TimeLimit

import safety_gym  # noqa: F401
from n_dim_reach_env.rl.optimization.optimize_hyperparameters import optimize_hyperparameters  # noqa: F401
# from n_dim_reach_env.wrappers.speed_action_wrapper import SpeedActionWrapper
from n_dim_reach_env.conf.config_cactd3 import CACTD3TrainingConfig, EnvConfig

from n_dim_reach_env.rl.train_cactd3 import train_cactd3

cs = ConfigStore.instance()
cs.store(name="cactd3_config", node=CACTD3TrainingConfig)


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


@hydra.main(config_path="../conf", config_name="conf_cactd3")
def main(cfg: CACTD3TrainingConfig):
    """Train a CACTD3 agent on the safety gym environment."""
    print(cfg)
    agent_kwargs = {
        "actor_lr": cfg.cactd3.actor_lr,
        "critic_lr": cfg.cactd3.critic_lr,
        "cost_critic_lr": cfg.cactd3.cost_critic_lr,
        "temp_lr": cfg.cactd3.temp_lr,
        "lambda_lr": cfg.cactd3.lambda_lr,
        "hidden_dims": cfg.cactd3.hidden_dims,
        "discount": cfg.cactd3.discount,
        "cost_discount": cfg.cactd3.cost_discount,
        "tau": cfg.cactd3.tau,
        "num_qs": cfg.cactd3.num_qs,
        "num_min_qs": cfg.cactd3.num_min_qs,
        "critic_dropout_rate": cfg.cactd3.critic_dropout_rate,
        "critic_layer_norm": cfg.cactd3.critic_layer_norm,
        "target_entropy": cfg.cactd3.target_entropy,
        "init_temperature": cfg.cactd3.init_temperature,
        "sampled_backup": cfg.cactd3.sampled_backup,
        "state_dependent_lambda": cfg.cactd3.state_dependent_lambda,
        "init_lambda": cfg.cactd3.init_lambda,
        "lambda_regularization": cfg.cactd3.lambda_regularization,
    }
    learn_args = {
        "seed": cfg.env.seed,
        "delta": cfg.env.delta,
        "agent_kwargs": agent_kwargs,
        "max_ep_len": cfg.env.max_ep_len,
        "max_steps": cfg.train.max_steps,
        "start_steps": cfg.cactd3.start_steps,
        "squash_output": cfg.cactd3.squash_output,
        "use_her": cfg.cactd3.use_her,
        "n_her_samples": cfg.cactd3.n_her_samples,
        "goal_selection_strategy": cfg.cactd3.goal_selection_strategy,
        "handle_timeout_termination": cfg.cactd3.handle_timeout_termination,
        "utd_ratio": cfg.cactd3.utd_ratio,
        "batch_size": cfg.cactd3.batch_size,
        "update_lambda_every": cfg.cactd3.update_lambda_every,
        "update_cost_target_every": cfg.cactd3.update_cost_target_every,
        "buffer_size": cfg.cactd3.buffer_size,
        "eval_interval": cfg.train.eval_interval,
        "eval_episodes": cfg.train.eval_episodes,
        "load_checkpoint": cfg.train.load_checkpoint,
        "load_from_folder": cfg.train.load_from_folder,
        "logging_keys": cfg.train.logging_keys,
        "use_tqdm": cfg.train.tqdm,
        "train_logging_interval": cfg.train.train_logging_interval,
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
        train_cactd3(
            env=env,
            eval_env=eval_env,
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
