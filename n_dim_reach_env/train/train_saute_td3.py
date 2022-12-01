#!/usr/bin/env python
"""This file describes the training functionality for DroQ with HER in Saute RL.

Owner:
    Jakob Thumm (JT)

Contributors:

Changelog:
    16.11.22 JT Created File.
"""
from copy import copy
import gym  # noqa: F401

import hydra
from hydra.core.config_store import ConfigStore

import safety_gym  # noqa: F401
from n_dim_reach_env.rl.optimization.optimize_hyperparameters import optimize_hyperparameters  # noqa: F401
# from n_dim_reach_env.wrappers.speed_action_wrapper import SpeedActionWrapper
from n_dim_reach_env.conf.config_td3 import TD3TrainingConfig

from n_dim_reach_env.rl.train_ac import train_ac
from n_dim_reach_env.train.train_saute_droq import create_env

cs = ConfigStore.instance()
cs.store(name="td3_config", node=TD3TrainingConfig)


@hydra.main(config_path="../conf", config_name="conf_td3")
def main(cfg: TD3TrainingConfig):
    """Train a TD3 agent on the safety gym environment with Saute wrapper and HER."""
    print(cfg)
    agent_kwargs = {
        "actor_lr": cfg.td3.actor_lr,
        "critic_lr": cfg.td3.critic_lr,
        "feature_extractor_lr": cfg.td3.feature_extractor_lr,
        "feature_extractor_dims": cfg.td3.feature_extractor_dims,
        "network_dims": cfg.td3.network_dims,
        "discount": cfg.td3.discount,
        "tau": cfg.td3.tau,
        "action_noise_std": cfg.td3.action_noise_std,
        "action_noise_clip": cfg.td3.action_noise_clip
    }
    learn_args = {
        "seed": cfg.env.seed,
        "agent_kwargs": agent_kwargs,
        "max_ep_len": cfg.env.max_ep_len,
        "max_steps": cfg.train.max_steps,
        "start_steps": cfg.td3.start_steps,
        "squash_output": cfg.td3.squash_output,
        "use_her": cfg.td3.use_her,
        "n_her_samples": cfg.td3.n_her_samples,
        "goal_selection_strategy": cfg.td3.goal_selection_strategy,
        "handle_timeout_termination": cfg.td3.handle_timeout_termination,
        "utd_ratio": cfg.td3.utd_ratio,
        "batch_size": cfg.td3.batch_size,
        "buffer_size": cfg.td3.buffer_size,
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
        train_ac(
            env=env,
            eval_env=eval_env,
            alg="td3",
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
            alg="td3",
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
