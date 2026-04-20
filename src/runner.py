from typing import Optional
import time

from src.simulation.sim import Simulation
from src.simulation.rendering import Renderer
from src.robot.pose import Pose


class Runner:
    def __init__(self, sim: Simulation, policy=None):
        self.sim    = sim
        self.policy = policy

    def run_episode(self, target_pose: Pose, max_steps: int, renderer: Optional[Renderer] = None) -> list:
        """Run a full episode, collecting observations each step."""
        self.sim.reset()
        trajectory = []
        target_dt  = (1.0 / renderer.fps) if renderer is not None and renderer.enabled else None

        for _ in range(max_steps):
            step_start = time.perf_counter()

            obs    = self.sim.get_obs()
            target = self._get_target(obs, target_pose)
            trajectory.append(obs)
            self.sim.step(target)

            if renderer is not None and renderer.enabled:
                renderer.render()
                if renderer.stop_request:
                    break
                elapsed = time.perf_counter() - step_start
                if target_dt - elapsed > 0:
                    time.sleep(target_dt - elapsed)

        return trajectory

    def _get_target(self, obs: dict, fallback: Pose) -> Pose:
        """Resolve action from policy if available, else use fallback target."""
        if self.policy is not None:
            return Pose.from_7d(self.policy.act(obs))
        return fallback

    def reset(self):
        """Reset the underlying simulation."""
        self.sim.reset()