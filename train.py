import sys, os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from pathlib import Path
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from src.common.utils            import load_yaml
from src.policy.gym_wrapper      import ManipulationEnv


class InfoLoggingCallback(BaseCallback):
    """Log per-step info dict means (phi, shape, place_dist, etc.) to TensorBoard."""

    def __init__(self, keys: list[str], log_every: int = 1000):
        super().__init__()
        self._keys      = keys
        self._log_every = log_every
        self._buffers   = {k: [] for k in keys}

    def _on_step(self) -> bool:
        for info in self.locals.get('infos', []):
            for k in self._keys:
                if k in info:
                    self._buffers[k].append(float(info[k]))

        if self.num_timesteps % self._log_every == 0:
            for k, buf in self._buffers.items():
                if buf:
                    self.logger.record(f'info/{k}_mean', float(np.mean(buf)))
                    self._buffers[k] = []
        return True


def _make_env_fn(config: dict, log_dir: str):
    def _init():
        env = ManipulationEnv(config, log_dir=None)
        return Monitor(env, log_dir)
    return _init


def _make_eval_env(config: dict, log_dir: str) -> Monitor:
    """Construct a single monitored eval environment with logging enabled."""
    env = ManipulationEnv(config, log_dir=log_dir)
    return Monitor(env, log_dir)


def train(config_path: str = "config.yaml"):
    config    = load_yaml(config_path)
    train_cfg = config['training']
    n_envs    = train_cfg.get('n_envs', 8)

    log_dir  = Path("logs/sac")
    ckpt_dir = Path("checkpoints/sac")
    log_dir.mkdir(parents=True,  exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_env = SubprocVecEnv(
        [_make_env_fn(config, str(log_dir)) for _ in range(n_envs)],
        start_method = 'spawn',
    )
    eval_env = _make_eval_env(config, str(log_dir / "eval"))

    check_env(eval_env, warn=True)

    model = SAC(
        policy          = "MlpPolicy",
        env             = train_env,
        learning_rate   = train_cfg['learning_rate'],
        buffer_size     = train_cfg['buffer_size'],
        batch_size      = train_cfg['batch_size'],
        gamma           = train_cfg['gamma'],
        tau             = train_cfg['tau'],
        train_freq      = train_cfg['train_freq'],
        gradient_steps  = train_cfg['gradient_steps'],
        learning_starts = train_cfg['learning_starts'],
        verbose         = 1,
        seed            = train_cfg['seed'],
        tensorboard_log = str(log_dir / "tb"),
        device          = "cuda",
    )

    checkpoint_cb = CheckpointCallback(
        save_freq          = 10_000,
        save_path          = str(ckpt_dir),
        name_prefix        = "sac_manipulation",
        save_replay_buffer = True,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = str(ckpt_dir / "best"),
        log_path             = str(log_dir / "eval"),
        eval_freq            = 25_000,
        n_eval_episodes      = 5,
        deterministic        = True,
    )

    info_cb = InfoLoggingCallback(
        keys      = ['phi', 'shape', 'reg', 'success_bonus',
                     'place_dist', 'obj_height', 'grasped', 'success'],
        log_every = 1000,
    )

    try:
        model.learn(
            total_timesteps = train_cfg['total_timesteps'],
            callback        = [checkpoint_cb, eval_cb, info_cb],
            progress_bar    = True,
        )
        final_path = ckpt_dir / "sac_manipulation_final"
        model.save(str(final_path))
        print(f"Training complete. Model saved to {final_path}.")
    except KeyboardInterrupt:
        print("Training interrupted — saving current model.")
        model.save(str(ckpt_dir / "sac_manipulation_interrupted"))
    finally:
        train_env.close()
        eval_env.close()


if __name__ == '__main__':
    train()