import numpy as np


class RewardFunction:
    def __init__(self, config: dict):
        task_cfg   = config['task']
        reward_cfg = config['reward']

        self.target_pos         = np.array(task_cfg['target_pos'])
        self.table_height       = task_cfg['table_height']
        self.lift_margin        = task_cfg['lift_margin']
        self.reach_gate_dist    = task_cfg['reach_gate_dist']
        self.place_success_dist = task_cfg['place_success_dist']

        self.k_reach           = reward_cfg['k_reach']
        self.w_grasp           = reward_cfg['w_grasp']
        self.w_lift            = reward_cfg['w_lift']
        self.w_place           = reward_cfg['w_place']
        self.w_place_bonus     = reward_cfg['w_place_bonus']
        self.w_reg             = reward_cfg['w_reg']
        self.w_push_penalty    = reward_cfg['w_push_penalty']
        self.w_premature_close = reward_cfg['w_premature_close']

    def compute(self, obs: dict, action: np.ndarray, obj_start_pos: np.ndarray) -> dict:
        """Compute all reward components and return breakdown plus total and success flag."""
        ee_pos     = obs['ee_pos']
        obj_pos    = obs['obj_pos']
        grasped    = obs['grasped']
        grasp_norm = float(action[3])

        reach_dist = np.linalg.norm(ee_pos - obj_pos)
        place_dist = np.linalg.norm(obj_pos - self.target_pos)
        obj_height = obj_pos[2] - self.table_height

        r_reach     = self._reach(reach_dist)
        r_grasp     = self._grasp(reach_dist, grasped, ee_pos, obj_pos)
        r_lift      = self._lift(grasped, obj_height)
        r_place     = self._place(grasped, obj_height, place_dist)
        r_reg       = self._reg(action)
        r_push      = self._push_penalty(grasped, obj_pos, obj_start_pos)
        r_premature = self._premature_close(grasp_norm, reach_dist)

        total   = r_reach + r_grasp + r_lift + r_place + r_reg + r_push + r_premature
        success = bool(grasped and place_dist < self.place_success_dist)

        return {
            'reach':     r_reach,
            'grasp':     r_grasp,
            'lift':      r_lift,
            'place':     r_place,
            'reg':       r_reg,
            'push':      r_push,
            'premature': r_premature,
            'total':     total,
            'success':   success,
        }

    def _reach(self, reach_dist: float) -> float:
        """Exponential reward pulling EE toward object, always active."""
        return float(np.exp(-self.k_reach * reach_dist))

    def _grasp(self, reach_dist: float, grasped: bool, ee_pos: np.ndarray, obj_pos: np.ndarray) -> float:
        """Flat bonus when EE is close, above object, and object is grasped."""
        above = ee_pos[2] > obj_pos[2] - 0.02
        if reach_dist < self.reach_gate_dist and grasped and above:
            return self.w_grasp
        return 0.0

    def _lift(self, grasped: bool, obj_height: float) -> float:
        """Reward proportional to object height above table, gated on grasp."""
        if not grasped:
            return 0.0
        return self.w_lift * max(0.0, obj_height)

    def _place(self, grasped: bool, obj_height: float, place_dist: float) -> float:
        """Distance penalty plus sparse success bonus, gated on grasp and lift."""
        if not grasped or obj_height < self.lift_margin:
            return 0.0
        r = -self.w_place * place_dist
        if place_dist < self.place_success_dist:
            r += self.w_place_bonus
        return r

    def _reg(self, action: np.ndarray) -> float:
        """Action regularisation penalty, always active."""
        return -self.w_reg * float(np.dot(action, action))

    def _push_penalty(self, grasped: bool, obj_pos: np.ndarray, obj_start_pos: np.ndarray) -> float:
        """Penalise XY displacement of object from start position when not grasped."""
        if grasped:
            return 0.0
        displacement = np.linalg.norm(obj_pos[:2] - obj_start_pos[:2])
        return -self.w_push_penalty * displacement

    def _premature_close(self, grasp_norm: float, reach_dist: float) -> float:
        """Penalise closing gripper while EE is far from object."""
        if grasp_norm > 0.5 and reach_dist > self.reach_gate_dist:
            return -self.w_premature_close * grasp_norm
        return 0.0