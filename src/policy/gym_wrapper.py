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
        self._stage       = config['training'].get('stage', 'full')

        action_cfg                  = config['action']
        self._delta_low             = action_cfg['delta_bounds'][0]
        self._delta_high            = action_cfg['delta_bounds'][1]
        self._fixed_quat            = np.array(action_cfg['fixed_quaternion'])
        self._grasp_open            = action_cfg['grasp_values']['open']
        self._grasp_close           = action_cfg['grasp_values']['close']
        self._gripper_close_thresh  = float(action_cfg.get('close_threshold', 0.65))
        self._gripper_open_thresh   = float(action_cfg.get('open_threshold', 0.35))
        self._gripper_is_closed     = False

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
        """Reset simulation and initialise reward state for a new episode."""
        super().reset(seed=seed)
        self._sim.reset()
        self._step_count        = 0
        self._gripper_is_closed = False

        if self._logger:
            self._logger.reset()

        obs = self._sim.get_obs()
        self._reward_fn.reset(obs)

        return self._flatten(obs), {'TimeLimit.truncated': False}

    def step(self, action: np.ndarray):
        """Apply one policy action, step simulation, and return the gym transition."""
        action     = np.clip(action, self.action_space.low, self.action_space.high)
        delta      = action[:3]
        grasp_norm = float(action[3])

        if grasp_norm >= self._gripper_close_thresh:
            self._gripper_is_closed = True
        elif grasp_norm <= self._gripper_open_thresh:
            self._gripper_is_closed = False

        grasp_cmd   = self._grasp_close if self._gripper_is_closed else self._grasp_open

        obs         = self._sim.get_obs()
        target_pos  = obs['ee_pos'] + delta
        target_pose = Pose(position=target_pos, quaternion=self._fixed_quat)

        self._sim.step(target_pose, grasp_cmd)
        obs         = self._sim.get_obs()

        logged_action = np.append(delta, grasp_norm)
        breakdown     = self._reward_fn.compute(obs, logged_action)
        reward        = float(breakdown['total'])

        self._step_count += 1
        terminated        = bool(breakdown['success'])
        truncated         = self._step_count >= self._max_steps

        if self._logger:
            self._logger.log_step(breakdown, obs, logged_action)
            if terminated or truncated:
                self._logger.save()

        if self._renderer is not None:
            self._renderer.render()

        info = {
            'stage':                self._stage,
            'reach':                float(breakdown['reach']),
            'align_xy':             float(breakdown['align_xy']),
            'align_z':              float(breakdown['align_z']),
            'grasp_bonus':          float(breakdown['grasp_bonus']),
            'hold':                 float(breakdown['hold']),
            'lift':                 float(breakdown['lift']),
            'place':                float(breakdown['place']),
            'release_bonus':        float(breakdown['release_bonus']),
            'success_bonus':        float(breakdown['success_bonus']),
            'drop_penalty':         float(breakdown['drop_penalty']),
            'time':                 float(breakdown['time']),
            'action_penalty':       float(breakdown['action_penalty']),
            'premature_close':      float(breakdown['premature_close']),
            'premature_close_flag': float(breakdown['premature_close_flag']),   
            'gripper_switch':       float(breakdown['gripper_switch']),
            'gripper_switch_flag':  float(breakdown['gripper_switch_flag']),
            'grasp_stable_count':   float(breakdown['grasp_stable_count']),
            'place_stable_count':   float(breakdown['place_stable_count']),
            'reach_dist':           float(breakdown['reach_dist']),
            'xy_dist':              float(breakdown['xy_dist']),
            'z_err':                float(breakdown['z_err']),
            'place_dist':           float(breakdown['place_dist']),
            'obj_height':           float(breakdown['obj_height']),
            'grasped':              float(breakdown['grasped']),
            'new_grasp':            float(breakdown['new_grasp']),
            'lost_grasp':           float(breakdown['lost_grasp']),
            'reach_success':        float(breakdown['reach_success']),
            'lift_success':         float(breakdown['lift_success']),
            'micro_lift_success':   float(breakdown['micro_lift_success']),
            'place_success':        float(breakdown['place_success']),
            'released_on_target':   float(breakdown['released_on_target']),
            'has_lifted':           float(breakdown['has_lifted']),
            'success':              float(breakdown['success']),
            'is_success':           bool(breakdown['success']),
            'gripper_is_closed':    float(self._gripper_is_closed),
        }

        return self._flatten(obs), reward, terminated, truncated, info

    def render(self):
        """Initialise the renderer on demand and draw the current frame."""
        if self._renderer is None and self._render_mode == 'human':
            from src.simulation.rendering import make_renderer
            self._renderer = make_renderer(self._sim, self._config.get('rendering', {}))
        if self._renderer is not None:
            self._renderer.render()

    def close(self):
        """Clean up active rendering resources."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _flatten(self, obs: dict) -> np.ndarray:
        """Flatten dict observations into a fixed-length float32 policy vector."""
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
        """Compute the flattened observation dimension from a live reset."""
        self._sim.reset()
        obs = self._sim.get_obs()
        return self._flatten(obs).shape[0]