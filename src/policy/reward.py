import numpy as np


class RewardFunction:
    """Hybrid shaped reward for pick-and-place.

    r_t = gamma * Phi(s') - Phi(s)           # PBRS shaping
        + grasp_dwell_payout                  # bounded flat dwell while grasped
        + r_success * 1[success]              # sparse terminal bonus
        - w_reg * ||a||^2                     # action regularisation

    PBRS (Ng/Harada/Russell 1999) gives optimal-policy-preserving shaping for the
    reach/lift/place geometry, with Phi >= 0 so camping is weakly penalised.

    The grasp dwell is a flat +w_grasp_dwell per step while grasped, capped at
    grasp_cap total per episode. This is NOT PBRS and does distort the optimal
    policy's value function, but the cap bounds total dwell reward below the
    success bonus so success remains the optimal terminal outcome. Its purpose
    is to give SAC's critic a strong local signal on the grasp subgoal, which
    pure PBRS failed to provide (dense signal was too weak for the critic to
    commit to grasping in 1M steps).
    """

    def __init__(self, config: dict):
        task_cfg   = config['task']
        reward_cfg = config['reward']
        train_cfg  = config['training']

        self.target_pos         = np.array(task_cfg['target_pos'], dtype=np.float32)
        self.table_height       = task_cfg['table_height']
        self.place_success_dist = task_cfg['place_success_dist']
        self.place_success_z    = task_cfg['place_success_z_tol']
        self.lift_target_h      = task_cfg['lift_target_h']
        self.obj_size_z         = config['object']['size'][2]

        self.w_reach       = reward_cfg['w_reach']
        self.w_lift        = reward_cfg['w_lift']
        self.w_place       = reward_cfg['w_place']
        self.w_grasp_jump  = reward_cfg['w_grasp_jump']
        self.r_success     = reward_cfg['r_success']
        self.w_reg         = reward_cfg['w_reg']
        self.phi_offset    = reward_cfg['phi_offset']
        self.w_grasp_dwell = reward_cfg['w_grasp_dwell']
        self.grasp_cap     = reward_cfg['grasp_cap']

        self.gamma = float(train_cfg['gamma'])

        self._prev_phi:           float | None = None
        self._grasp_pay_remaining: float       = 0.0

    def reset(self, obs: dict):
        """Initialise potential and reset dwell budget at episode start."""
        self._prev_phi            = self._potential(obs)
        self._grasp_pay_remaining = self.grasp_cap

    def compute(self, obs: dict, action: np.ndarray) -> dict:
        """Compute r = shape + dwell + success + reg, update episode state."""
        phi_next = self._potential(obs)
        phi_prev = self._prev_phi if self._prev_phi is not None else phi_next

        r_shape = self.gamma * phi_next - phi_prev
        r_reg   = -self.w_reg * float(np.dot(action, action))

        grasped = bool(obs['grasped'])
        if grasped and self._grasp_pay_remaining > 0.0:
            r_dwell                    = min(self.w_grasp_dwell, self._grasp_pay_remaining)
            self._grasp_pay_remaining -= r_dwell
        else:
            r_dwell = 0.0

        place_dist = float(np.linalg.norm(obs['obj_pos'][:2] - self.target_pos[:2]))
        z_err      = abs(float(obs['obj_pos'][2]) - float(self.target_pos[2]))
        success    = (grasped
                     and place_dist < self.place_success_dist
                     and z_err < self.place_success_z)

        r_succ = self.r_success if success else 0.0
        total  = r_shape + r_dwell + r_reg + r_succ

        self._prev_phi = phi_next

        return {
            'phi':           float(phi_next),
            'shape':         float(r_shape),
            'dwell':         float(r_dwell),
            'dwell_remaining': float(self._grasp_pay_remaining),
            'reg':           float(r_reg),
            'success_bonus': float(r_succ),
            'total':         float(total),
            'success':       bool(success),
            'place_dist':    place_dist,
            'obj_height':    float(obs['obj_pos'][2] - self.table_height),
            'grasped':       grasped,
        }

    def _potential(self, obs: dict) -> float:
        """Phi(s): offset + reach + (grasped ? grasp_jump + lift - place : 0), >= 0."""
        ee_pos  = obs['ee_pos']
        obj_pos = obs['obj_pos']
        grasped = bool(obs['grasped'])

        grasp_point = obj_pos + np.array([0.0, 0.0, -self.obj_size_z * 0.3])
        reach_dist  = float(np.linalg.norm(ee_pos - grasp_point))

        phi = self.phi_offset - self.w_reach * reach_dist

        if grasped:
            obj_h      = float(obj_pos[2] - self.table_height)
            lift_prog  = min(max(obj_h, 0.0), self.lift_target_h)
            place_dist = float(np.linalg.norm(obj_pos[:2] - self.target_pos[:2]))

            phi += self.w_grasp_jump + self.w_lift * lift_prog - self.w_place * place_dist

        return phi