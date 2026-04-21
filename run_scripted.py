import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.common.utils           import load_yaml
from src.simulation.sim         import Simulation
from src.simulation.rendering   import make_renderer
from src.policy.reward          import RewardFunction
from src.policy.scripted_agent  import ScriptedAgent
from src.common.logger          import EpisodeLogger


def run(config_path: str = "config.yaml"):
    """Run the scripted agent through the environment and log post-step reward traces."""
    config   = load_yaml(config_path)
    sim      = Simulation(config)
    reward   = RewardFunction(config)
    agent    = ScriptedAgent(config)
    logger   = EpisodeLogger(log_dir="logs/scripted", prefix="scripted")
    renderer = make_renderer(sim, config.get('rendering', {}))

    agent.reset()
    logger.reset()
    sim.reset()

    obs = sim.get_obs()
    reward.reset(obs)

    dt        = sim.dt
    max_steps = 2000
    target_dt = (1.0 / renderer.fps) if renderer is not None and renderer.enabled else None

    import time
    for step in range(max_steps):
        step_start = time.perf_counter()

        obs                = sim.get_obs()
        target_pose, grasp = agent.act(obs, dt)

        grasp_norm = 1.0 if grasp == config['action']['grasp_values']['close'] else 0.0
        action     = np.array([
            *(target_pose.position - obs['ee_pos']),
            grasp_norm,
        ], dtype=np.float32)
        action     = np.clip(action, [-0.05, -0.05, -0.05, 0.0], [0.05, 0.05, 0.05, 1.0])

        sim.step(target_pose, grasp)
        next_obs          = sim.get_obs()
        reward_breakdown  = reward.compute(next_obs, action)

        logger.log_step(reward_breakdown, next_obs, action)

        if renderer is not None and renderer.enabled:
            renderer.render()
            if renderer.stop_request:
                break
            elapsed = time.perf_counter() - step_start
            if target_dt - elapsed > 0:
                time.sleep(target_dt - elapsed)

        if bool(reward_breakdown['success']):
            print(f"Stage success at step {step}.")
            break

        from src.policy.scripted_agent import Phase
        if agent._phase == Phase.DONE:
            print(f"Scripted policy reached DONE at step {step}.")
            break

    log_path = logger.save()
    print(f"Log saved to {log_path}")
    _print_summary(logger.load(log_path))
    _plot(logger.load(log_path), log_path)


def _print_summary(data: dict):
    """Print a compact summary of the logged scripted rollout."""
    def series(key):
        """Fetch a logged series by key, tolerating prefix variations."""
        for candidate in (f'reward/{key}', key, f'info/{key}', f'obs/{key}'):
            if candidate in data:
                return np.asarray(data[candidate]).squeeze()
        return None

    total         = series('total')
    reach         = series('reach')
    grasp_bonus   = series('grasp_bonus')
    lift          = series('lift')
    place         = series('place')
    success_bonus = series('success_bonus')
    reach_dist    = series('reach_dist')
    place_dist    = series('place_dist')
    obj_height    = series('obj_height')
    grasped       = series('grasped')
    success       = series('success')

    if total is None:
        print("No reward traces found in log.")
        return

    print("\nScripted rollout summary:")
    print(f"  steps:            {len(total)}")
    print(f"  total reward:     {float(np.sum(total)):.3f}")
    print(f"  mean reward:      {float(np.mean(total)):.3f}")

    if reach is not None:
        print(f"  reach sum:        {float(np.sum(reach)):.3f}")
    if grasp_bonus is not None:
        print(f"  grasp bonus sum:  {float(np.sum(grasp_bonus)):.3f}")
    if lift is not None:
        print(f"  lift sum:         {float(np.sum(lift)):.3f}")
    if place is not None:
        print(f"  place sum:        {float(np.sum(place)):.3f}")
    if success_bonus is not None:
        print(f"  success sum:      {float(np.sum(success_bonus)):.3f}")
    if reach_dist is not None:
        print(f"  final reach dist: {float(reach_dist[-1]):.4f}")
        print(f"  min reach dist:   {float(np.min(reach_dist)):.4f}")
    if place_dist is not None:
        print(f"  final place dist: {float(place_dist[-1]):.4f}")
        print(f"  min place dist:   {float(np.min(place_dist)):.4f}")
    if obj_height is not None:
        print(f"  max obj height:   {float(np.max(obj_height)):.4f}")
    if grasped is not None:
        print(f"  grasped steps:    {int(np.sum(grasped > 0.5))}")
    if success is not None:
        print(f"  success reached:  {bool(np.any(success > 0.5))}")


def _plot(data: dict, log_path):
    """Plot scripted rollout reward components and task progress traces."""
    def series(key):
        """Fetch a logged series by key, tolerating prefix variations."""
        for candidate in (f'reward/{key}', key, f'info/{key}', f'obs/{key}'):
            if candidate in data:
                return np.asarray(data[candidate]).squeeze()
        return None

    components = [
        ('reach',          'steelblue'),
        ('grasp_bonus',    'seagreen'),
        ('lift',           'darkorange'),
        ('place',          'mediumpurple'),
        ('success_bonus',  'gold'),
        ('drop_penalty',   'firebrick'),
        ('time',           'gray'),
        ('action_penalty', 'salmon'),
        ('total',          'black'),
    ]
    extras = [
        ('reach_dist', 'dodgerblue'),
        ('place_dist', 'purple'),
        ('obj_height', 'chocolate'),
        ('grasped',    'teal'),
        ('success',    'green'),
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