"""Defines the dataclasses of the config files."""
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class FACConfig:
    """FAC + HER config."""

    actor_lr: float
    critic_lr: float
    cost_critic_lr: float
    temp_lr: float
    lambda_lr: float
    hidden_dims: List[int]
    discount: float
    cost_discount: float
    tau: float
    num_qs: int
    num_min_qs: int
    critic_dropout_rate: float
    critic_layer_norm: bool
    target_entropy: float
    init_temperature: float
    sampled_backup: bool
    update_lambda_every: int
    buffer_size: int
    use_her: bool
    n_her_samples: int
    goal_selection_strategy: str
    handle_timeout_termination: bool
    start_steps: int
    batch_size: int
    utd_ratio: int
    squash_output: bool
    state_dependent_lambda: bool
    init_lambda: float
    lambda_regularization: float


@dataclass
class EnvConfig:
    """Environment config."""

    id: str
    render_mode: str
    seed: int
    max_ep_len: int
    delta: float


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
    train_logging_interval: int


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
class FACTrainingConfig:
    """Training config."""

    fac: FACConfig
    env: EnvConfig
    train: TrainConfig
    optimize: OptimizeConfig
    verbose: bool
