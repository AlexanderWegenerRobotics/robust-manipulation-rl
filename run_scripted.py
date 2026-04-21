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
    steps      = np.arange(data['reward/total'].shape[0])
    components = ['reach', 'grasp', 'lift', 'place', 'reg', 'total']
    colors     = ['steelblue', 'seagreen', 'darkorange', 'mediumpurple', 'salmon', 'black']

    fig, axes = plt.subplots(len(components), 1, figsize=(12, 10), sharex=True)

    for ax, key, color in zip(axes, components, colors):
        series = data[f'reward/{key}'].squeeze()
        ax.plot(steps, series, color=color, linewidth=1.2)
        ax.set_ylabel(key, fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Step')
    fig.suptitle('Scripted Agent — Reward Components', fontsize=11)
    plt.tight_layout()

    plot_path = Path(str(log_path).replace('.h5', '.png'))
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to {plot_path}")
    plt.show()


if __name__ == '__main__':
    run()
