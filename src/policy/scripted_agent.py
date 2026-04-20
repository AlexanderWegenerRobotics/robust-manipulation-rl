import numpy as np
from enum import Enum, auto

from src.robot.pose import Pose
from src.robot.trajectory import TrajectoryPlanner


class Phase(Enum):
    REACH   = auto()
    DESCEND = auto()
    GRASP   = auto()
    LIFT    = auto()
    PLACE   = auto()
    DONE    = auto()


class ScriptedAgent:
    def __init__(self, config: dict):
        action_cfg  = config['action']
        object_cfg  = config['object']
        task_cfg    = config['task']

        self._fixed_quat     = np.array(action_cfg['fixed_quaternion'])
        self._grasp_open     = action_cfg['grasp_values']['open']
        self._grasp_close    = action_cfg['grasp_values']['close']

        self._obj_size_z     = object_cfg['size'][2]
        self._table_height   = task_cfg['table_height']
        self._target_pos     = np.array(task_cfg['target_pos'])

        self._hover_height   = 0.25
        self._carry_height   = 0.30
        self._grasp_dwell    = 20

        self._planner        = TrajectoryPlanner()
        self._phase          = Phase.REACH
        self._grasp_cmd      = self._grasp_open
        self._dwell_steps    = 0
        self._initialized    = False

    def reset(self):
        """Reset state machine and planner to initial phase."""
        self._phase        = Phase.REACH
        self._grasp_cmd    = self._grasp_open
        self._dwell_steps  = 0
        self._initialized  = False

    def act(self, obs: dict, dt: float) -> tuple[Pose, int]:
        """Advance state machine and return (target_pose, grasp_command)."""
        ee_pos   = obs['ee_pos']
        ee_quat  = obs['ee_quat']
        obj_pos  = obs['obj_pos']
        grasped  = obs['grasped']

        if not self._initialized:
            self._plan_phase(self._phase, ee_pos, obj_pos)
            self._initialized = True

        if not self._planner.is_done():
            out = self._planner.step(dt)
            return Pose(position=out['pos'], quaternion=self._fixed_quat), self._grasp_cmd

        next_phase = self._transition(self._phase, grasped, obj_pos, ee_pos)
        if next_phase != self._phase:
            self._phase = next_phase
            self._plan_phase(self._phase, ee_pos, obj_pos)

        out = self._planner.step(dt)
        return Pose(position=out['pos'], quaternion=self._fixed_quat), self._grasp_cmd

    def _transition(self, phase: Phase, grasped: bool, obj_pos: np.ndarray, ee_pos: np.ndarray) -> Phase:
        """Determine next phase when current trajectory is complete."""
        if phase == Phase.REACH:
            return Phase.DESCEND

        if phase == Phase.DESCEND:
            return Phase.GRASP

        if phase == Phase.GRASP:
            self._grasp_cmd = self._grasp_close
            self._dwell_steps += 1
            if self._dwell_steps >= self._grasp_dwell:
                return Phase.LIFT
            return Phase.GRASP

        if phase == Phase.LIFT:
            return Phase.PLACE

        if phase == Phase.PLACE:
            return Phase.DONE

        return Phase.DONE

    def _plan_phase(self, phase: Phase, ee_pos: np.ndarray, obj_pos: np.ndarray):
        """Plan a min-jerk trajectory for the given phase."""
        q = self._fixed_quat

        if phase == Phase.REACH:
            target = np.array([obj_pos[0], obj_pos[1], self._hover_height])
            self._planner.plan_with_speed(ee_pos, q, target, q, max_speed=0.10)

        elif phase == Phase.DESCEND:
            grasp_z = self._table_height + self._obj_size_z * 2 + 0.01
            target  = np.array([obj_pos[0], obj_pos[1], grasp_z])
            self._planner.plan_with_speed(ee_pos, q, target, q, max_speed=0.05)

        elif phase == Phase.GRASP:
            self._planner.plan(ee_pos, q, ee_pos, q, duration=0.1)

        elif phase == Phase.LIFT:
            target = np.array([ee_pos[0], ee_pos[1], self._carry_height])
            self._planner.plan_with_speed(ee_pos, q, target, q, max_speed=0.08)

        elif phase == Phase.PLACE:
            mid    = np.array([self._target_pos[0], self._target_pos[1], self._carry_height])
            target = np.array([self._target_pos[0], self._target_pos[1], self._table_height + self._obj_size_z * 2])
            self._planner.plan_with_speed(ee_pos, q, mid,    q, max_speed=0.10)
            self._planner.plan_with_speed(mid,    q, target, q, max_speed=0.05)

        elif phase == Phase.DONE:
            self._planner.plan(ee_pos, q, ee_pos, q, duration=0.1)
