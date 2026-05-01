import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.common.utils           import load_yaml
from src.simulation.sim         import Simulation
from src.simulation.rendering   import make_renderer
from src.policy.reward          import RewardFunction
from src.policy.scripted_agent  import ScriptedAgent, Phase
from src.common.logger          import EpisodeLogger
from src.robot.pose             import Pose


def run(config_path: str = "config.yaml"):
    """Run the original scripted agent and plot reward traces afterward."""
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

    release_started      = False
    release_counter      = 0
    release_dwell        = 12
    retreat_started      = False
    retreat_target       = None
    retreat_height       = 0.10
    reached_place_phase  = False
    aborted_before_place = False

    fixed_quat = np.array(config['action']['fixed_quaternion'], dtype=np.float32)
    grasp_open = config['action']['grasp_values']['open']

    for step in range(max_steps):
        step_start = time.perf_counter()
        obs = sim.get_obs()

        if agent._phase in (Phase.PLACE_MOVE, Phase.PLACE_DESCEND):
            reached_place_phase = True

        if agent._phase != Phase.DONE:
            target_pose, grasp = agent.act(obs, dt)

        else:
            if not reached_place_phase:
                aborted_before_place = True
                print(f"Scripted rollout reached DONE before place at step {step}.")
                break

            if not release_started:
                release_started = True
                release_counter = 0
                target_pose     = Pose(position=obs['ee_pos'].copy(), quaternion=fixed_quat)
                grasp           = grasp_open

            elif release_counter < release_dwell:
                release_counter += 1
                target_pose = Pose(position=obs['ee_pos'].copy(), quaternion=fixed_quat)
                grasp       = grasp_open

            else:
                if not retreat_started:
                    retreat_started = True
                    retreat_target  = obs['ee_pos'].copy()
                    retreat_target[2] += retreat_height

                target_pose = Pose(position=retreat_target.copy(), quaternion=fixed_quat)
                grasp       = grasp_open

        grasp_norm = 1.0 if grasp == config['action']['grasp_values']['close'] else 0.0
        action     = np.array([
            *(target_pose.position - obs['ee_pos']),
            grasp_norm,
        ], dtype=np.float32)
        action     = np.clip(action, [-0.05, -0.05, -0.05, 0.0], [0.05, 0.05, 0.05, 1.0])

        sim.step(target_pose, grasp)
        next_obs         = sim.get_obs()
        reward_breakdown = reward.compute(next_obs, action)

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

        if retreat_started and np.linalg.norm(next_obs['ee_pos'] - retreat_target) < 0.02:
            print(f"Scripted rollout completed at step {step}.")
            break

    log_path = logger.save()
    print(f"Log saved to {log_path}")
    data = logger.load(log_path)
    _print_summary(data, aborted_before_place=aborted_before_place, reached_place_phase=reached_place_phase)
    _plot(data, log_path)


def _print_summary(data: dict, aborted_before_place: bool, reached_place_phase: bool):
    """Print a compact summary of the logged scripted rollout."""
    def series(key):
        for candidate in (f'reward/{key}', key, f'info/{key}', f'obs/{key}'):
            if candidate in data:
                return np.asarray(data[candidate]).squeeze()
        return None

    keys = [
        'total', 'reach', 'grasp_bonus', 'hold', 'carry', 'lift', 'place',
        'release_bonus', 'success_bonus', 'drop_penalty', 'reach_dist',
        'place_dist', 'obj_height', 'lift_progress', 'carry_progress',
        'grasped', 'success', 'released_on_target', 'place_success'
    ]
    s = {k: series(k) for k in keys}

    if s['total'] is None:
        print("No reward traces found in log.")
        return

    print("\nScripted rollout summary:")
    print(f"  steps:                {len(s['total'])}")
    print(f"  total reward:         {float(np.sum(s['total'])):.3f}")
    print(f"  mean reward:          {float(np.mean(s['total'])):.3f}")
    print(f"  reached place phase:  {reached_place_phase}")
    print(f"  aborted before place: {aborted_before_place}")

    for k in ['reach', 'grasp_bonus', 'hold', 'carry', 'lift', 'place', 'release_bonus', 'success_bonus', 'drop_penalty']:
        if s[k] is not None:
            print(f"  {k:20s} {float(np.sum(s[k])):.3f}")

    if s['reach_dist'] is not None:
        print(f"  final reach dist:     {float(s['reach_dist'][-1]):.4f}")
        print(f"  min reach dist:       {float(np.min(s['reach_dist'])):.4f}")
    if s['place_dist'] is not None:
        print(f"  final place dist:     {float(s['place_dist'][-1]):.4f}")
        print(f"  min place dist:       {float(np.min(s['place_dist'])):.4f}")
    if s['obj_height'] is not None:
        print(f"  max obj height:       {float(np.max(s['obj_height'])):.4f}")
    if s['lift_progress'] is not None:
        print(f"  max lift progress:    {float(np.max(s['lift_progress'])):.4f}")
    if s['carry_progress'] is not None:
        print(f"  max carry progress:   {float(np.max(s['carry_progress'])):.4f}")
    if s['grasped'] is not None:
        print(f"  grasped steps:        {int(np.sum(s['grasped'] > 0.5))}")
    if s['released_on_target'] is not None:
        print(f"  released on target:   {bool(np.any(s['released_on_target'] > 0.5))}")
    if s['place_success'] is not None:
        print(f"  place success seen:   {bool(np.any(s['place_success'] > 0.5))}")
    if s['success'] is not None:
        print(f"  success reached:      {bool(np.any(s['success'] > 0.5))}")


def _plot(data: dict, log_path):
    """Plot scripted rollout reward components and task progress traces."""
    def series(key):
        for candidate in (f'reward/{key}', key, f'info/{key}', f'obs/{key}'):
            if candidate in data:
                return np.asarray(data[candidate]).squeeze()
        return None

    components = [
        ('reach',          'steelblue'),
        ('grasp_bonus',    'seagreen'),
        ('hold',           'teal'),
        ('carry',          'darkcyan'),
        ('lift',           'darkorange'),
        ('place',          'mediumpurple'),
        ('release_bonus',  'goldenrod'),
        ('success_bonus',  'gold'),
        ('drop_penalty',   'firebrick'),
        ('time',           'gray'),
        ('action_penalty', 'salmon'),
        ('total',          'black'),
    ]
    extras = [
        ('reach_dist',         'dodgerblue'),
        ('place_dist',         'purple'),
        ('obj_height',         'chocolate'),
        ('lift_progress',      'orange'),
        ('carry_progress',     'darkgreen'),
        ('grasped',            'teal'),
        ('place_success',      'olive'),
        ('released_on_target', 'brown'),
        ('success',            'green'),
    ]

    available = [(k, c) for k, c in components + extras if series(k) is not None]
    if not available:
        print("No plottable series found in log.")
        return

    n_steps = series(available[0][0]).shape[0]
    steps   = np.arange(n_steps)

    fig, axes = plt.subplots(len(available), 1, figsize=(12, 1.55 * len(available)), sharex=True)
    if len(available) == 1:
        axes = [axes]

    for ax, (key, color) in zip(axes, available):
        vals = series(key)
        ax.plot(steps, vals, color=color, linewidth=1.2)
        ax.set_ylabel(key, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(0.0, color='gray', linewidth=0.5, alpha=0.5)

    axes[-1].set_xlabel('Step')
    fig.suptitle('Scripted Full Task — Reward and Task Progress', fontsize=11)
    plt.tight_layout()

    plot_path = Path(str(log_path).replace('.h5', '.png'))
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to {plot_path}")
    plt.show()


if __name__ == '__main__':
    run()