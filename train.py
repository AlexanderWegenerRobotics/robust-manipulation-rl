import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

import numpy as np
from pathlib import Path
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor

from src.common.utils         import load_yaml
from src.environment.gym_wrapper import ManipulationEnv


def make_env(config: dict, log_dir: str, render: bool = False) -> ManipulationEnv:
    """Construct a monitored training environment."""
    env = ManipulationEnv(
        config      = config,
        log_dir     = log_dir,
        render_mode = 'human' if render else None,
    )
    return Monitor(env, log_dir)


def train(config_path: str = "config.yaml"):
    config       = load_yaml(config_path)
    train_cfg    = config['training']

    log_dir      = Path("logs/sac")
    ckpt_dir     = Path("checkpoints/sac")
    log_dir.mkdir(parents=True,  exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_env = make_env(config, log_dir=str(log_dir))
    eval_env  = make_env(config, log_dir=str(log_dir / "eval"))

    check_env(train_env, warn=True)

    model = SAC(
        policy             = "MlpPolicy",
        env                = train_env,
        learning_rate      = train_cfg['learning_rate'],
        buffer_size        = train_cfg['buffer_size'],
        batch_size         = train_cfg['batch_size'],
        gamma              = train_cfg['gamma'],
        tau                = train_cfg['tau'],
        train_freq         = train_cfg['train_freq'],
        gradient_steps     = train_cfg['gradient_steps'],
        learning_starts    = train_cfg['learning_starts'],
        verbose            = 1,
        seed               = train_cfg['seed'],
        tensorboard_log    = str(log_dir / "tb"),
    )

    checkpoint_cb = CheckpointCallback(
        save_freq       = 10_000,
        save_path       = str(ckpt_dir),
        name_prefix     = "sac_manipulation",
        save_replay_buffer = True,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = str(ckpt_dir / "best"),
        log_path             = str(log_dir / "eval"),
        eval_freq            = 10_000,
        n_eval_episodes      = 5,
        deterministic        = True,
    )

    model.learn(
        total_timesteps = train_cfg['total_timesteps'],
        callback        = [checkpoint_cb, eval_cb],
        progress_bar    = True,
    )

    final_path = ckpt_dir / "sac_manipulation_final"
    model.save(str(final_path))
    print(f"Training complete. Model saved to {final_path}.")
    train_env.close()
    eval_env.close()


if __name__ == '__main__':
    train()
