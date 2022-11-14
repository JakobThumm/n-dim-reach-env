#!/usr/bin/env python
"""This file describes the plotting of the Q function of DroQ.

Owner:
    Jakob Thumm (JT)

Contributors:

Changelog:
    15.10.22 JT Created File.
"""
import os
import gym  # noqa: F401
import numpy as np
import jax
# import pickle
import matplotlib.pyplot as plt

from flax.training import checkpoints

import hydra
from hydra.core.config_store import ConfigStore

from n_dim_reach_env.conf.config_fac import FACTrainingConfig

from n_dim_reach_env.train.train_fac import create_env
from n_dim_reach_env.rl.util.action_scaling import unscale_action
from n_dim_reach_env.rl.util.dict_conversion import\
    single_obs, goal_dist, goal_lidar, get_observation_space, has_dict_obs

from n_dim_reach_env.rl.agents import FACLearner
# from n_dim_reach_env.rl.data import ReplayBuffer
# from n_dim_reach_env.rl.data.her_replay_buffer import HEReplayBuffer

cs = ConfigStore.instance()
cs.store(name="fac_config", node=FACTrainingConfig)

PLOT_DIMENSIONS = [3, 5]
N_PLOTS = 5
N_POINTS = 20
N_ACTIONS = 16


@hydra.main(config_path="../conf", config_name="conf_fac")
def main(cfg: FACTrainingConfig):
    """Plot the Q function of the DroQ Agent."""
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".80"
    print(jax.devices())
    # << Training >>
    env = create_env(cfg.env)
    observation_space = get_observation_space(env)
    # has_dict = has_dict_obs(env)
    action_space = env.action_space
    assert cfg.train.load_from_folder is not None
    chkpt_dir = cfg.train.load_from_folder + 'saved/checkpoints/'
    buffer_dir = cfg.train.load_from_folder + 'saved/buffers/'
    d = os.listdir(chkpt_dir)[0]
    assert os.path.isdir(chkpt_dir + d)
    chkpt_dir = chkpt_dir + d
    buffer_dir = buffer_dir + d

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
        "sampled_backup": cfg.fac.sampled_backup,
        "state_dependent_lambda": cfg.fac.state_dependent_lambda,
        "init_lambda": cfg.fac.init_lambda
    }
    agent = FACLearner.create(
        seed=cfg.env.seed,
        observation_space=observation_space,
        action_space=env.action_space,
        delta=cfg.env.delta,
        **agent_kwargs)
    if cfg.train.load_checkpoint == -1:
        last_checkpoint = checkpoints.latest_checkpoint(chkpt_dir)
    else:
        last_checkpoint = chkpt_dir + '/checkpoint_' + str(cfg.train.load_checkpoint)
    # start_i = int(last_checkpoint.split('_')[-1])
    agent = checkpoints.restore_checkpoint(last_checkpoint, agent)
    # with open(os.path.join(buffer_dir, f'buffer_{start_i}'), 'rb') as f:
    #     replay_buffer = pickle.load(f)
    if hasattr(env, 'observe_goal_lidar') and env.observe_goal_lidar:
        dict_to_obs_fn = goal_lidar
        observation_space = get_observation_space(env, "lidar")
    elif hasattr(env, 'observe_goal_dist') and env.observe_goal_dist:
        dict_to_obs_fn = goal_dist
        observation_space = get_observation_space(env, "dist")
    else:
        dict_to_obs_fn = single_obs
        observation_space = get_observation_space(env)
    dict_obs = has_dict_obs(env)
    ##### EVAL #####
    
    observation, done = env.reset(), False
    cost_sum = 0
    np.set_printoptions(precision=2)
    for i in range(10):
        observation, done = env.reset(), False
        env.render("human")
        while not done:
            if dict_obs:
                action_observation = dict_to_obs_fn(observation)
            else:
                action_observation = observation
            agent_action = agent.eval_actions(action_observation)
            action = unscale_action(agent_action,
                                    env.action_space.low,
                                    env.action_space.high,
                                    cfg.fac.squash_output)
            qc_val = agent.cost_critic.apply_fn(
                {'params': agent.cost_critic.params},
                action_observation,
                agent_action,
                False
            )._value
            observation, reward, done, infos = env.step(action)
            env.render("human")
            hazards = action_observation[22:38]
            print("Cost critic: {:.2f}, cost: {:.0f}".format(#, action: {}, a: {}, v: {}, g: {}, m: {}, max hazard: {:.3f}".format(
                qc_val,
                infos['cost'],
                # action,
                # action_observation[0:3],
                # action_observation[57:60],
                # action_observation[19:22],
                # action_observation[38:41],
                # np.max(hazards)
            ))
            if "cost" in infos:
                cost_sum += infos["cost"]
        print("Cost sum: {:.0f}".format(cost_sum))
    

    # Test the Qc network for actions
    observation_faulty_no_cost = np.array([  
         0.34, -12.34,   9.81,   0.42,   0.  ,   0.  ,   0.  ,   0.  ,
         0.  ,   0.  ,   0.  ,   0.  ,   0.  ,   0.  ,   0.  ,   0.  ,
         0.  ,   0.25,   0.67,   0.  ,   0.  ,  -0.48,   0.86,   0.7 ,
         0.61,   0.62,   0.74,   0.47,   0.  ,   0.  ,   0.  ,   0.  ,
         0.  ,   0.  ,   0.  ,   0.  ,   0.  ,   0.28,   0.34,  -0.37,
         0.  ,   0.  ,   0.  ,   0.  ,   0.  ,   0.  ,   0.  ,   0.  ,
         0.  ,   0.  ,   0.01,   0.67,   0.66,   0.  ,   0.  ,   0.  ,
         0.  ,   1.33,  -0.16,   0.  ])
    observation_medium_cost = np.array([
        -0.88, -2.72,  9.81,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
        0.09,  0.45,  0.35,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
        0.  ,  0.  ,  0.  ,  2.27,  0.  ,  0.6 ,  0.72,  0.82,  0.7 ,
        0.44,  0.8 ,  0.93,  0.13,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
        0.  ,  0.  ,  0.42, -0.27,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
        0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.22,  0.56,  0.34,  0.  ,
        0.  ,  0.  ,  0.  , -0.99,  0.04,  0.  ])
    observation_high_cost = np.array([
        -3.81e-01, -1.69e+00,  9.81e+00,  0.00e+00,  0.00e+00,  0.00e+00,
        0.00e+00,  0.00e+00,  0.00e+00,  0.00e+00,  3.14e-01,  6.23e-01,
        3.08e-01,  0.00e+00,  0.00e+00,  0.00e+00,  0.00e+00,  0.00e+00,
        0.00e+00,  0.00e+00,  0.00e+00, -2.38e+00,  0.00e+00,  0.00e+00,
        0.00e+00,  0.00e+00,  0.00e+00,  0.00e+00,  0.00e+00,  3.55e-02,
        8.48e-01,  8.12e-01,  4.04e-01,  6.99e-01,  9.39e-01,  3.46e-01,
        8.58e-01,  8.47e-01, -4.65e-01, -1.84e-01,  0.00e+00,  0.00e+00,
        0.00e+00,  0.00e+00,  0.00e+00,  0.00e+00,  0.00e+00,  0.00e+00,
        0.00e+00,  0.00e+00,  0.00e+00,  7.12e-01,  7.16e-01,  4.17e-03,
        0.00e+00,  0.00e+00,  0.00e+00, -1.25e+00, -2.92e-01,  0.00e+00])
    observation = observation_high_cost
    observation[3:19] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.67, 0.42])
    # observation[57:60] = observation_high_cost[57:60]
    agent_action = agent.eval_actions(observation)
    # Test critic for actions
    actions = np.linspace(-1, 1, N_POINTS)
    all_actions = np.tile(actions, (N_POINTS**2, 1))
    a_i = np.repeat(actions, N_POINTS)
    a_j = np.tile(actions, N_POINTS)
    all_actions = np.concatenate([a_i[:, np.newaxis], a_j[:, np.newaxis]], axis=1)
    all_observations = np.tile(observation, (N_POINTS**2, 1))
    key, rng = jax.random.split(agent.rng)
    # key2, rng = jax.random.split(rng)
    qc_vals = agent.cost_critic.apply_fn(
            {'params': agent.cost_critic.params},
            all_observations,
            all_actions,
            False
        )._value
    qc_val_agent = agent.cost_critic.apply_fn(
            {'params': agent.cost_critic.params},
            observation,
            agent_action,
            False
        )._value
    # Test critic for observations
    obs_size = observation.shape[0]
    action = np.array([0.0, 0.0])
    factors = np.ones((obs_size, ))
    offsets = np.zeros((obs_size, ))
    N_LIDAR = 16
    N_SENS = 3
    # accelerometer 0:3
    factors[0:N_SENS] = 10
    factors[2] = 0
    # goal lidar 3:19
    factors[N_SENS:N_SENS+N_LIDAR] = 0.5
    offsets[N_SENS:N_SENS+N_LIDAR] = 1
    # gyro 19:22
    factors[N_SENS+N_LIDAR:2*N_SENS+N_LIDAR] = 1
    factors[2*N_SENS+N_LIDAR-1] = 0
    # hazards lidar 22:38
    factors[2*N_SENS+N_LIDAR:2*N_SENS+2*N_LIDAR] = 0.5
    offsets[2*N_SENS+N_LIDAR:2*N_SENS+2*N_LIDAR] = 1
    # magneto 38:41
    factors[2*N_SENS+2*N_LIDAR:3*N_SENS+2*N_LIDAR] = 0
    offsets[2*N_SENS+2*N_LIDAR:3*N_SENS+2*N_LIDAR] = 0
    # vases lidar 41:57
    factors[3*N_SENS+2*N_LIDAR:3*N_SENS+3*N_LIDAR] = 0.5
    offsets[3*N_SENS+2*N_LIDAR:3*N_SENS+3*N_LIDAR] = 1
    # velocity 57:60
    factors[3*N_SENS+3*N_LIDAR:4*N_SENS+3*N_LIDAR] = 2
    new_obs = np.linspace(-1, 1, N_POINTS)
    all_observations = np.tile(observation, (N_POINTS*obs_size, 1))
    for i in range(obs_size):
        all_observations[i*N_POINTS:(i+1)*N_POINTS, i] = (new_obs+offsets[i])*factors[i]
    all_actions = np.tile(action, (N_POINTS*obs_size, 1))
    qc_vals = agent.cost_critic.apply_fn(
            {'params': agent.cost_critic.params},
            all_observations,
            all_actions,
            False
        )._value
    for i in range(obs_size):
        vals = qc_vals[i*N_POINTS:(i+1)*N_POINTS]
        print("Mean, std Qc for obs {}: {:.2f}, {:.2f}".format(i, np.mean(vals), np.std(vals)))
    stop=0


if __name__ == "__main__":
    main()
