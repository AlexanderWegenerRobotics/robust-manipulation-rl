import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.common.utils      import load_yaml
from src.simulation.sim    import Simulation
from src.simulation.rendering import make_renderer
from src.policy.reward     import RewardFunction
from src.policy.scripted_agent import ScriptedAgent
from src.common.logger     import EpisodeLogger


def run(config_path: str = "config.yaml"):
    config  = load_yaml(config_path)
    sim     = Simulation(config)
    reward  = RewardFunction(config)
    agent   = ScriptedAgent(config)
    logger  = EpisodeLogger(log_dir="logs/scripted", prefix="scripted")
    renderer = make_renderer(sim, config.get('rendering', {}))

    agent.reset()
    logger.reset()
    sim.reset()

    dt         = sim.dt
    max_steps  = 2000
    target_dt  = (1.0 / renderer.fps) if renderer is not None and renderer.enabled else None

    import time
    for step in range(max_steps):
        step_start = time.perf_counter()

        obs                  = sim.get_obs()
        target_pose, grasp   = agent.act(obs, dt)
        action               = np.array([*(target_pose.position - obs['ee_pos']), float(grasp)])
        action = np.clip(action, [-0.05, -0.05, -0.05, 0.0], [0.05, 0.05, 0.05, 1.0])
        reward_breakdown     = reward.compute(obs, action)

        logger.log_step(reward_breakdown, obs, action)
        sim.step(target_pose, grasp)

        if renderer is not None and renderer.enabled:
            renderer.render()
            if renderer.stop_request:
                break
            elapsed = time.perf_counter() - step_start
            if target_dt - elapsed > 0:
                time.sleep(target_dt - elapsed)

        from src.policy.scripted_agent import Phase
        if agent._phase == Phase.DONE:
            print(f"Episode complete at step {step}.")
            break

    log_path = logger.save()
    print(f"Log saved to {log_path}")
    _plot(logger.load(log_path), log_path)


def _plot(data: dict, log_path):
    def series(key):
        """Fetch a logged series by key, tolerating prefix variations."""
        for candidate in (f'reward/{key}', key, f'info/{key}', f'obs/{key}'):
            if candidate in data:
                return np.asarray(data[candidate]).squeeze()
        return None

    components = [
        ('phi',           'steelblue'),
        ('shape',         'seagreen'),
        ('reg',           'salmon'),
        ('success_bonus', 'gold'),
        ('total',         'black'),
    ]
    extras = [
        ('place_dist', 'mediumpurple'),
        ('obj_height', 'darkorange'),
        ('grasped',    'teal'),
    ]

    available = [(k, c) for k, c in components + extras if series(k) is not None]
    if not available:
        print("No plottable series found in log.")
        return

    n_steps = series(available[0][0]).shape[0]
    steps   = np.arange(n_steps)

    fig, axes = plt.subplots(len(available), 1, figsize=(12, 1.6 * len(available)), sharex=True)
    if len(available) == 1:
        axes = [axes]

    for ax, (key, color) in zip(axes, available):
        s = series(key)
        ax.plot(steps, s, color=color, linewidth=1.2)
        ax.set_ylabel(key, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(0.0, color='gray', linewidth=0.5, alpha=0.5)

    axes[-1].set_xlabel('Step')
    fig.suptitle('Scripted Agent — Reward and Task Progress', fontsize=11)
    plt.tight_layout()

    plot_path = Path(str(log_path).replace('.h5', '.png'))
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to {plot_path}")
    plt.show()


if __name__ == '__main__':
    run()
