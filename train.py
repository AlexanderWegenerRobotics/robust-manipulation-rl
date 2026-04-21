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

from src.common.utils       import load_yaml
from src.policy.gym_wrapper import ManipulationEnv


class InfoLoggingCallback(BaseCallback):
    def __init__(self, keys: list[str], log_every: int = 1000):
        super().__init__()
        self._keys      = keys
        self._log_every = log_every
        self._buffers   = {k: [] for k in keys}

    def _on_step(self) -> bool:
        """Aggregate selected info fields and periodically write their means."""
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
    """Construct a single monitored evaluation environment with logging enabled."""
    env = ManipulationEnv(config, log_dir=log_dir)
    return Monitor(env, log_dir)


def _make_model(config: dict, train_env, log_dir: Path):
    """Construct a fresh SAC model from config."""
    train_cfg = config['training']
    return SAC(
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


def _load_or_create_model(config: dict, train_env, log_dir: Path):
    """Load a saved SAC checkpoint if configured, otherwise create a fresh model."""
    train_cfg         = config['training']
    resume_model_path = train_cfg.get('resume_model_path', None)

    if resume_model_path:
        print(f"Loading model from {resume_model_path}")
        model = SAC.load(resume_model_path, env=train_env, device="cuda")
        return model

    return _make_model(config, train_env, log_dir)


def _try_load_replay_buffer(model: SAC, config: dict):
    """Load replay buffer if a configured path exists."""
    train_cfg                  = config['training']
    resume_replay_buffer_path  = train_cfg.get('resume_replay_buffer_path', None)

    if resume_replay_buffer_path:
        path = Path(resume_replay_buffer_path)
        if path.exists():
            print(f"Loading replay buffer from {path}")
            model.load_replay_buffer(str(path))
        else:
            print(f"Replay buffer path does not exist: {path}")


def train(config_path: str = "config.yaml"):
    """Train SAC for the configured curriculum stage."""
    config    = load_yaml(config_path)
    train_cfg = config['training']
    stage     = train_cfg.get('stage', 'full')
    n_envs    = train_cfg.get('n_envs', 8)

    log_dir   = Path("logs") / "sac" / stage
    ckpt_dir  = Path("checkpoints") / "sac" / stage
    log_dir.mkdir(parents=True,  exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_env = SubprocVecEnv(
        [_make_env_fn(config, str(log_dir)) for _ in range(n_envs)],
        start_method='spawn',
    )
    eval_env  = _make_eval_env(config, str(log_dir / "eval"))

    check_env(eval_env, warn=True)

    model = _load_or_create_model(config, train_env, log_dir)
    _try_load_replay_buffer(model, config)

    checkpoint_cb = CheckpointCallback(
        save_freq          = 10_000,
        save_path          = str(ckpt_dir),
        name_prefix        = f"sac_manipulation_{stage}",
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
        keys      = [
            'reach', 'grasp_bonus', 'lift', 'place', 'success_bonus',
            'drop_penalty', 'time', 'action_penalty',
            'reach_dist', 'place_dist', 'obj_height',
            'grasped', 'new_grasp', 'lost_grasp',
            'reach_success', 'lift_success', 'place_success', 'success'
        ],
        log_every = 1000,
    )

    try:
        model.learn(
            total_timesteps = train_cfg['total_timesteps'],
            callback        = [checkpoint_cb, eval_cb, info_cb],
            progress_bar    = True,
            reset_num_timesteps = train_cfg.get('reset_num_timesteps', True),
        )
        final_path = ckpt_dir / f"sac_manipulation_{stage}_final"
        model.save(str(final_path))
        replay_path = ckpt_dir / f"sac_manipulation_{stage}_final_replay_buffer"
        model.save_replay_buffer(str(replay_path))
        print(f"Training complete for stage '{stage}'. Model saved to {final_path}.")
    except KeyboardInterrupt:
        interrupted_path = ckpt_dir / f"sac_manipulation_{stage}_interrupted"
        interrupted_rb   = ckpt_dir / f"sac_manipulation_{stage}_interrupted_replay_buffer"
        print("Training interrupted — saving current model.")
        model.save(str(interrupted_path))
        model.save_replay_buffer(str(interrupted_rb))
    finally:
        train_env.close()
        eval_env.close()


if __name__ == '__main__':
    train()