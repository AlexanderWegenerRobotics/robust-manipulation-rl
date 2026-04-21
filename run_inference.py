import sys, os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

import numpy as np
from pathlib import Path
from stable_baselines3 import SAC

from src.common.utils            import load_yaml
from src.policy.gym_wrapper import ManipulationEnv


def run_inference(
    checkpoint_path: str,
    config_path:     str = "config.yaml",
    n_episodes:      int = 5,
):
    config = load_yaml(config_path)
    env    = ManipulationEnv(config, log_dir=None, render_mode='human')
    model  = SAC.load(checkpoint_path, env=env)

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Running {n_episodes} episodes...\n")

    for ep in range(n_episodes):
        obs, _     = env.reset()
        done       = False
        total_rew  = 0.0
        step_count = 0
        success    = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()

            total_rew  += reward
            step_count += 1
            success     = bool(info.get('success', False))
            done        = terminated or truncated

        status = "SUCCESS" if success else "FAILED"
        print(f"Episode {ep + 1:02d} | {status} | steps: {step_count:4d} | reward: {total_rew:8.2f}")

    env.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/sac/best/best_model.zip')
    parser.add_argument('--config',     type=str, default='config.yaml')
    parser.add_argument('--episodes',   type=int, default=5)
    args = parser.parse_args()

    run_inference(
        checkpoint_path = args.checkpoint,
        config_path     = args.config,
        n_episodes      = args.episodes,
    )