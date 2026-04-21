import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.simulation.sim    import Simulation
from src.robot.pose        import Pose
from src.policy.reward     import RewardFunction
from src.common.logger     import EpisodeLogger


class ManipulationEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, config: dict, log_dir: str = None, render_mode: str = None):
        super().__init__()

        self._config      = config
        self._render_mode = render_mode
        self._sim         = Simulation(config)
        self._reward_fn   = RewardFunction(config)
        self._logger      = EpisodeLogger(log_dir, prefix='train') if log_dir else None
        self._renderer    = None

        action_cfg        = config['action']
        self._delta_low   = action_cfg['delta_bounds'][0]
        self._delta_high  = action_cfg['delta_bounds'][1]
        self._fixed_quat  = np.array(action_cfg['fixed_quaternion'])
        self._grasp_open  = action_cfg['grasp_values']['open']
        self._grasp_close = action_cfg['grasp_values']['close']

        self._max_steps  = config['training']['max_episode_steps']
        self._step_count = 0

        self.action_space = spaces.Box(
            low  = np.array([self._delta_low,  self._delta_low,  self._delta_low,  0.0], dtype=np.float32),
            high = np.array([self._delta_high, self._delta_high, self._delta_high, 1.0], dtype=np.float32),
        )

        obs_dim = self._obs_dim()
        self.observation_space = spaces.Box(
            low   = -np.inf,
            high  =  np.inf,
            shape = (obs_dim,),
            dtype = np.float32,
        )

    def reset(self, seed=None, options=None):
        """Reset simulation, initialise reward potential and dwell budget."""
        super().reset(seed=seed)
        self._sim.reset()
        self._step_count = 0

        if self._logger:
            self._logger.reset()

        obs = self._sim.get_obs()
        self._reward_fn.reset(obs)

        return self._flatten(obs), {'TimeLimit.truncated': False}

    def step(self, action: np.ndarray):
        """Apply action, step simulation, compute shaped reward, return gym tuple."""
        action      = np.clip(action, self.action_space.low, self.action_space.high)
        delta       = action[:3]
        grasp_norm  = float(action[3])
        grasp_cmd   = self._grasp_close if grasp_norm > 0.5 else self._grasp_open

        obs         = self._sim.get_obs()
        target_pos  = obs['ee_pos'] + delta
        target_pose = Pose(position=target_pos, quaternion=self._fixed_quat)

        self._sim.step(target_pose, grasp_cmd)
        obs         = self._sim.get_obs()

        logged_action = np.append(delta, grasp_norm)
        breakdown     = self._reward_fn.compute(obs, logged_action)
        reward        = float(breakdown['total'])

        self._step_count += 1
        terminated        = breakdown['success']
        truncated         = self._step_count >= self._max_steps

        if self._logger:
            self._logger.log_step(breakdown, obs, logged_action)
            if terminated or truncated:
                self._logger.save()

        if self._renderer is not None:
            self._renderer.render()

        info = {
            'phi':             float(breakdown['phi']),
            'shape':           float(breakdown['shape']),
            'dwell':           float(breakdown['dwell']),
            'dwell_remaining': float(breakdown['dwell_remaining']),
            'reg':             float(breakdown['reg']),
            'success_bonus':   float(breakdown['success_bonus']),
            'place_dist':      float(breakdown['place_dist']),
            'obj_height':      float(breakdown['obj_height']),
            'grasped':         float(breakdown['grasped']),
            'success':         float(breakdown['success']),
            'is_success':      bool(breakdown['success']),
        }

        return self._flatten(obs), reward, terminated, truncated, info

    def render(self):
        """Initialise renderer on first call and render current frame."""
        if self._renderer is None and self._render_mode == 'human':
            from src.simulation.rendering import make_renderer
            self._renderer = make_renderer(self._sim, self._config.get('rendering', {}))
        if self._renderer is not None:
            self._renderer.render()

    def close(self):
        """Clean up renderer if active."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _flatten(self, obs: dict) -> np.ndarray:
        """Flatten obs dict into a fixed-length float32 vector for the policy."""
        return np.concatenate([
            obs['ee_pos'],
            obs['ee_quat'],
            obs['obj_pos'],
            obs['obj_quat'],
            obs['q'],
            obs['qd'],
            [obs['gripper_width']],
            [float(obs['grasped'])],
            obs['target_pos'],
        ]).astype(np.float32)

    def _obs_dim(self) -> int:
        """Compute flat observation dimension from a live sim reset."""
        self._sim.reset()
        obs = self._sim.get_obs()
        return self._flatten(obs).shape[0]