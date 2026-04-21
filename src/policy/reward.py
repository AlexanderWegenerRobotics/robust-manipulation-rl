import numpy as np


class RewardFunction:
    def __init__(self, config: dict):
        task_cfg   = config['task']
        reward_cfg = config['reward']
        train_cfg  = config['training']
        object_cfg = config['object']

        self.stage              = train_cfg.get('stage', 'full')
        self.target_pos         = np.array(task_cfg['target_pos'], dtype=np.float32)
        self.table_height       = float(task_cfg['table_height'])
        self.place_success_dist = float(task_cfg['place_success_dist'])
        self.place_success_z    = float(task_cfg['place_success_z_tol'])
        self.lift_target_h      = float(task_cfg['lift_target_h'])
        self.obj_size_z         = float(object_cfg['size'][2])

        self.reach_thresh       = float(reward_cfg.get('reach_success_dist', 0.03))
        self.lift_thresh        = float(reward_cfg.get('lift_success_height', self.lift_target_h))

        self.w_reach            = float(reward_cfg.get('w_reach', 1.0))
        self.w_grasp_bonus      = float(reward_cfg.get('w_grasp_bonus', 10.0))
        self.w_lift             = float(reward_cfg.get('w_lift', 4.0))
        self.w_place            = float(reward_cfg.get('w_place', 3.0))
        self.r_success          = float(reward_cfg.get('r_success', 50.0))
        self.w_action           = float(reward_cfg.get('w_action', reward_cfg.get('w_reg', 1e-3)))
        self.w_time             = float(reward_cfg.get('w_time', 0.01))
        self.r_drop             = float(reward_cfg.get('r_drop', 0.0))

        self._prev_grasped      = False

    def reset(self, obs: dict):
        """Reset stage-local event memory at episode start."""
        self._prev_grasped = bool(obs['grasped'])

    def compute(self, obs: dict, action: np.ndarray) -> dict:
        """Compute dense stage-based reward and task success signals."""
        ee_pos       = obs['ee_pos']
        obj_pos      = obs['obj_pos']
        grasped      = bool(obs['grasped'])
        new_grasp    = grasped and not self._prev_grasped
        lost_grasp   = self._prev_grasped and not grasped

        grasp_point  = obj_pos + np.array([0.0, 0.0, -self.obj_size_z * 0.3], dtype=np.float32)
        reach_dist   = float(np.linalg.norm(ee_pos - grasp_point))
        obj_height   = float(max(obj_pos[2] - self.table_height, 0.0))
        place_dist   = float(np.linalg.norm(obj_pos[:2] - self.target_pos[:2]))
        z_err        = abs(float(obj_pos[2]) - float(self.target_pos[2]))

        r_reach      = -self.w_reach * reach_dist
        r_grasp      = self.w_grasp_bonus if new_grasp else 0.0
        r_lift       = 0.0
        r_place      = 0.0
        r_succ       = 0.0
        r_drop       = -self.r_drop if lost_grasp and self.r_drop > 0.0 else 0.0
        r_time       = -self.w_time
        r_action     = -self.w_action * float(np.dot(action, action))

        reach_success = reach_dist < self.reach_thresh
        lift_success  = obj_height >= self.lift_thresh
        place_success = grasped and place_dist < self.place_success_dist and z_err < self.place_success_z

        if self.stage == 'reach_grasp':
            success = grasped
        elif self.stage == 'full':
            if grasped:
                r_lift = self.w_lift * min(obj_height, self.lift_target_h)
                if obj_height >= self.lift_thresh:
                    r_place = -self.w_place * place_dist
            success = place_success
            if success:
                r_succ = self.r_success
        else:
            raise ValueError(f"Unsupported training stage '{self.stage}'")

        total = r_reach + r_grasp + r_lift + r_place + r_succ + r_drop + r_time + r_action

        self._prev_grasped = grasped

        return {
            'total':         float(total),
            'reach':         float(r_reach),
            'grasp_bonus':   float(r_grasp),
            'lift':          float(r_lift),
            'place':         float(r_place),
            'success_bonus': float(r_succ),
            'drop_penalty':  float(r_drop),
            'time':          float(r_time),
            'action_penalty': float(r_action),
            'reach_dist':    float(reach_dist),
            'place_dist':    float(place_dist),
            'obj_height':    float(obj_height),
            'grasped':       grasped,
            'new_grasp':     bool(new_grasp),
            'lost_grasp':    bool(lost_grasp),
            'reach_success': bool(reach_success),
            'lift_success':  bool(lift_success),
            'place_success': bool(place_success),
            'success':       bool(success),
        }