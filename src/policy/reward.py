import numpy as np


class RewardFunction:
    def __init__(self, config: dict):
        task_cfg   = config['task']
        reward_cfg = config['reward']
        train_cfg  = config['training']
        object_cfg = config['object']

        self.stage               = train_cfg.get('stage', 'full')
        self.target_pos          = np.array(task_cfg['target_pos'], dtype=np.float32)
        self.table_height        = float(task_cfg['table_height'])
        self.place_success_dist  = float(task_cfg['place_success_dist'])
        self.place_success_z     = float(task_cfg['place_success_z_tol'])
        self.lift_target_h       = float(task_cfg['lift_target_h'])
        self.obj_size_z          = float(object_cfg['size'][2])

        self.reach_thresh        = float(reward_cfg.get('reach_success_dist', 0.03))
        self.lift_thresh         = float(reward_cfg.get('lift_success_height', self.lift_target_h))
        self.micro_lift_height   = float(reward_cfg.get('micro_lift_height', 0.008))

        self.w_align_xy          = float(reward_cfg.get('w_align_xy', 3.0))
        self.w_align_z           = float(reward_cfg.get('w_align_z', 2.0))
        self.w_grasp_bonus       = float(reward_cfg.get('w_grasp_bonus', 10.0))
        self.w_hold              = float(reward_cfg.get('w_hold', 0.0))
        self.w_lift              = float(reward_cfg.get('w_lift', 4.0))
        self.w_place             = float(reward_cfg.get('w_place', 3.0))
        self.r_success           = float(reward_cfg.get('r_success', 50.0))
        self.w_action            = float(reward_cfg.get('w_action', reward_cfg.get('w_reg', 1e-3)))
        self.w_time              = float(reward_cfg.get('w_time', 0.01))
        self.r_drop              = float(reward_cfg.get('r_drop', 0.0))

        self.w_premature_close   = float(reward_cfg.get('w_premature_close', 0.0))
        self.close_ok_xy_dist    = float(reward_cfg.get('close_ok_xy_dist', 0.03))
        self.close_ok_z_err      = float(reward_cfg.get('close_ok_z_err', 0.02))
        self.w_gripper_switch    = float(reward_cfg.get('w_gripper_switch', 0.0))

        self.grasp_stable_steps  = int(reward_cfg.get('grasp_stable_steps', 5))

        self._prev_grasped       = False
        self._prev_close_cmd     = False
        self._grasp_stable_count = 0
        self._grasp_bonus_paid   = False

    def reset(self, obs: dict):
        """Reset stage-local event memory at episode start."""
        self._prev_grasped       = bool(obs['grasped'])
        self._prev_close_cmd     = False
        self._grasp_stable_count = 1 if bool(obs['grasped']) else 0
        self._grasp_bonus_paid   = False

    def compute(self, obs: dict, action: np.ndarray) -> dict:
        """Compute dense stage-based reward and task success signals."""
        ee_pos        = obs['ee_pos']
        obj_pos       = obs['obj_pos']
        grasped       = bool(obs['grasped'])
        raw_new_grasp = grasped and not self._prev_grasped
        lost_grasp    = self._prev_grasped and not grasped

        if grasped:
            self._grasp_stable_count += 1
        else:
            self._grasp_stable_count = 0

        grasp_z       = float(obj_pos[2] - self.obj_size_z * 0.3)
        xy_dist       = float(np.linalg.norm(ee_pos[:2] - obj_pos[:2]))
        z_err         = abs(float(ee_pos[2]) - grasp_z)
        reach_dist    = float(np.linalg.norm(ee_pos - (obj_pos + np.array([0.0, 0.0, -self.obj_size_z * 0.3], dtype=np.float32))))
        obj_height    = float(max(obj_pos[2] - self.table_height, 0.0))
        place_dist    = float(np.linalg.norm(obj_pos[:2] - self.target_pos[:2]))
        place_z_err   = abs(float(obj_pos[2]) - float(self.target_pos[2]))

        grasp_cmd_close = float(action[3]) > 0.5
        premature_close = grasp_cmd_close and (xy_dist > self.close_ok_xy_dist or z_err > self.close_ok_z_err)
        gripper_switch  = grasp_cmd_close != self._prev_close_cmd

        new_grasp = raw_new_grasp and not self._grasp_bonus_paid

        r_align_xy   = -self.w_align_xy * xy_dist
        r_align_z    = -self.w_align_z * z_err
        r_reach      = r_align_xy + r_align_z
        r_grasp      = self.w_grasp_bonus if new_grasp else 0.0
        r_hold       = self.w_hold if grasped else 0.0
        r_lift       = 0.0
        r_place      = 0.0
        r_succ       = 0.0
        r_drop       = -self.r_drop if lost_grasp and self.r_drop > 0.0 else 0.0
        r_time       = -self.w_time
        r_action     = -self.w_action * float(np.dot(action, action))
        r_prem_close = -self.w_premature_close if premature_close and self.w_premature_close > 0.0 else 0.0
        r_switch     = -self.w_gripper_switch if gripper_switch and self.w_gripper_switch > 0.0 else 0.0

        reach_success      = reach_dist < self.reach_thresh
        lift_success       = obj_height >= self.lift_thresh
        micro_lift_success = obj_height >= self.micro_lift_height
        place_success      = grasped and place_dist < self.place_success_dist and place_z_err < self.place_success_z

        if self.stage == 'reach_grasp':
            success = (
                self._grasp_stable_count >= self.grasp_stable_steps
                and micro_lift_success
            )

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

        total = r_reach + r_grasp + r_hold + r_lift + r_place + r_succ + r_drop + r_time + r_action + r_prem_close + r_switch

        if new_grasp:
            self._grasp_bonus_paid = True

        self._prev_grasped   = grasped
        self._prev_close_cmd = grasp_cmd_close

        return {
            'total':                float(total),
            'reach':                float(r_reach),
            'align_xy':             float(r_align_xy),
            'align_z':              float(r_align_z),
            'grasp_bonus':          float(r_grasp),
            'hold':                 float(r_hold),
            'lift':                 float(r_lift),
            'place':                float(r_place),
            'success_bonus':        float(r_succ),
            'drop_penalty':         float(r_drop),
            'time':                 float(r_time),
            'action_penalty':       float(r_action),
            'premature_close':      float(r_prem_close),
            'gripper_switch':       float(r_switch),
            'reach_dist':           float(reach_dist),
            'xy_dist':              float(xy_dist),
            'z_err':                float(z_err),
            'place_dist':           float(place_dist),
            'obj_height':           float(obj_height),
            'grasped':              grasped,
            'new_grasp':            bool(new_grasp),
            'lost_grasp':           bool(lost_grasp),
            'premature_close_flag': bool(premature_close),
            'gripper_switch_flag':  bool(gripper_switch),
            'grasp_stable_count':   float(self._grasp_stable_count),
            'reach_success':        bool(reach_success),
            'lift_success':         bool(lift_success),
            'micro_lift_success':   bool(micro_lift_success),
            'place_success':        bool(place_success),
            'success':              bool(success),
        }