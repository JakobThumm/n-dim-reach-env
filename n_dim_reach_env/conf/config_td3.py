"""Defines the dataclasses of the config files."""
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class TD3Config:
    """TD3 + HER config."""

    actor_lr: float
    critic_lr: float
    feature_extractor_lr: float
    feature_extractor_dims: List[int]
    network_dims: List[int]
    discount: float
    tau: float
    action_noise_std: float
    action_noise_clip: float
    buffer_size: int
    use_her: bool
    n_her_samples: int
    goal_selection_strategy: str
    handle_timeout_termination: bool
    start_steps: int
    batch_size: int
    utd_ratio: int
    squash_output: bool


@dataclass
class EnvConfig:
    """Environment config."""

    id: str
    n_dim: int
    max_action: float
    goal_distance: float
    done_on_collision: bool
    randomize_env: bool
    collision_reward: float
    goal_reward: float
    step_reward: float
    reward_shaping: bool
    render_mode: bool
    seed: int
    max_ep_len: int
    replace_type: int
    n_resamples: int
    punishment: int


@dataclass
class TrainConfig:
    """Training settings."""

    max_steps: int
    eval_interval: int
    eval_episodes: int
    tqdm: bool
    use_wandb: bool
    wandb_project: str
    load_checkpoint: int
    load_from_folder: str
    logging_keys: Dict


@dataclass
class OptimizeConfig:
    """Optimization settings."""

    optimize: bool
    tuning_params: List[str]
    n_trials: int
    n_startup_trials: int
    n_timesteps: int
    n_jobs: int
    sampler_method: str
    pruner_method: str
    n_warmup_steps: int
    upper_threshold: float
    n_eval_episodes: int
    n_evaluations: int
    seed: int
    use_prior: bool


@dataclass
class TD3TrainingConfig:
    """Training config."""

    td3: TD3Config
    env: EnvConfig
    train: TrainConfig
    optimize: OptimizeConfig
    verbose: bool
