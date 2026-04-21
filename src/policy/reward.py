import numpy as np


class RewardFunction:
    """Potential-based shaped reward for pick-and-place.

    r_t = gamma * Phi(s_{t+1}) - Phi(s_t) + r_success * 1[success] - w_reg * ||a||^2

    Phi is constructed so Phi(s) >= 0 everywhere reachable, by adding a constant
    offset. This is required because with gamma < 1 the sitting-still shaping
    reward equals (gamma - 1) * Phi(s); if Phi can go negative, the agent is
    paid to camp in negative-Phi states. With Phi >= 0, (gamma - 1) * Phi <= 0
    always, so sitting still is weakly penalised and progress is weakly rewarded.

    Adding a constant to Phi is PBRS-invariant (Ng/Harada/Russell 1999), so the
    optimal policy is preserved.
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

        self.w_reach      = reward_cfg['w_reach']
        self.w_lift       = reward_cfg['w_lift']
        self.w_place      = reward_cfg['w_place']
        self.w_grasp_jump = reward_cfg['w_grasp_jump']
        self.r_success    = reward_cfg['r_success']
        self.w_reg        = reward_cfg['w_reg']
        self.phi_offset   = reward_cfg['phi_offset']

        self.gamma = float(train_cfg['gamma'])

        self._prev_phi: float | None = None

    def reset(self, obs: dict):
        """Initialise potential from first observation of an episode."""
        self._prev_phi = self._potential(obs)

    def compute(self, obs: dict, action: np.ndarray) -> dict:
        """Compute shaped reward r = gamma * Phi(s') - Phi(s) + success + reg."""
        phi_next = self._potential(obs)
        phi_prev = self._prev_phi if self._prev_phi is not None else phi_next

        r_shape = self.gamma * phi_next - phi_prev
        r_reg   = -self.w_reg * float(np.dot(action, action))

        grasped    = bool(obs['grasped'])
        place_dist = float(np.linalg.norm(obs['obj_pos'][:2] - self.target_pos[:2]))
        z_err      = abs(float(obs['obj_pos'][2]) - float(self.target_pos[2]))
        success    = (grasped
                     and place_dist < self.place_success_dist
                     and z_err < self.place_success_z)

        r_succ = self.r_success if success else 0.0
        total  = r_shape + r_reg + r_succ

        self._prev_phi = phi_next

        return {
            'phi':     float(phi_next),
            'shape':   float(r_shape),
            'reg':     float(r_reg),
            'success_bonus': float(r_succ),
            'total':   float(total),
            'success': bool(success),
            'place_dist': place_dist,
            'obj_height': float(obs['obj_pos'][2] - self.table_height),
            'grasped':    grasped,
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