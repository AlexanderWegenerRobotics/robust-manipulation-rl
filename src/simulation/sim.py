import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import mujoco
import yaml
import threading
import time
from dataclasses import dataclass
from typing import Optional

from src.common.utils import load_yaml
from src.robot.robot_kinematics import RobotKinematics
from src.robot.pose import Pose
from src.robot.control import ImpedanceController


ARM_DOF = 7


@dataclass
class RobotState:
    q:   np.ndarray
    qd:  np.ndarray
    tau: np.ndarray
    ee_pose: Pose


class Simulation:
    def __init__(self, config: str):
        self.config = config
        self.mj_model = mujoco.MjModel.from_xml_path(self.config['simulation']['scene_path'])
        self.mj_data  = mujoco.MjData(self.mj_model)
        self._lock    = threading.Lock()

        self.steps_per_action = self.config['simulation']['steps_per_action']
        self.q0 = np.array(self.config['simulation']['q0'])

        robot_cfg = self.config['robot']
        self.kinematics = RobotKinematics(urdf_path = robot_cfg['urdf_path'], ee_frame_name= robot_cfg['ee_frame_name'])
        self.controller = ImpedanceController(self.config['control'], self.kinematics)

        self.obj_name = config["object"]["name"]
        self.obj_start_pos, self.obj_start_quat = config["object"]["pos"], config["object"]["quat"]

        self._target_pose: Optional[Pose] = None
        self.reset()

        grasp_cfg = self.config['grasp_detection']
        self._left_finger_id  = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, grasp_cfg['contact_bodies'][0])
        self._right_finger_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, grasp_cfg['contact_bodies'][1])
        self._box_body_id     = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, grasp_cfg['object_body'])
        self._grasp_width_min = grasp_cfg['width_min']
        self._grasp_width_max = grasp_cfg['width_max']
        
        robot_cfg = self.config['robot']
        self._finger_idx_left  = robot_cfg['finger_joints']['index_left']
        self._finger_idx_right = robot_cfg['finger_joints']['index_right']

    def reset(self):
        """Reset simulation to initial joint configuration q0, zeroing all derivatives."""
        with self._lock:
            self.mj_data.qpos[:ARM_DOF] = self.q0
            self.mj_data.qvel[:]        = 0.0
            self.mj_data.qacc[:]        = 0.0
            self.mj_data.ctrl[:]        = 0.0
            self.mj_data.qfrc_applied[:]= 0.0
            mujoco.mj_forward(self.mj_model, self.mj_data)
            self._target_pose = self.kinematics.forward_kinematics(self.q0)
        self.set_object_pose(self.obj_name, self.obj_start_pos, self.obj_start_quat)

    def step(self, target_pose: Pose, grasp: int = 255):
        """Run steps_per_action physics steps, applying impedance torques and gripper command."""
        self._target_pose = target_pose

        for _ in range(self.steps_per_action):
            with self._lock:
                q   = self.mj_data.qpos[:ARM_DOF].copy()
                qd  = self.mj_data.qvel[:ARM_DOF].copy()

                tau = self.controller.compute_control(q, qd, self._target_pose)
                self.mj_data.ctrl[:ARM_DOF] = tau
                self.mj_data.ctrl[self._gripper_ctrl_idx] = grasp

                mujoco.mj_step(self.mj_model, self.mj_data)

    def _detect_contact(self) -> bool:
        box_id = self._box_body_id
        for i in range(self.mj_data.ncon):
            c = self.mj_data.contact[i]
            b1 = self.mj_model.geom_bodyid[c.geom1]
            b2 = self.mj_model.geom_bodyid[c.geom2]
            left_touch  = (b1 == self._left_finger_id  or b2 == self._left_finger_id)
            right_touch = (b1 == self._right_finger_id or b2 == self._right_finger_id)
            box_involved = (b1 == box_id or b2 == box_id)
            if box_involved and (left_touch or right_touch):
                return True
        return False

    def get_obs(self) -> dict:
        with self._lock:
            q   = self.mj_data.qpos[:ARM_DOF].copy()
            qd  = self.mj_data.qvel[:ARM_DOF].copy()
            tau = self.mj_data.ctrl[:ARM_DOF].copy()
            finger_l = self.mj_data.qpos[self._finger_idx_left]
            finger_r = self.mj_data.qpos[self._finger_idx_right]
            contact  = self._detect_contact()

        ee_pose        = self.kinematics.forward_kinematics(q)
        obj_pos, obj_quat = self.get_object_pose(self.obj_name)
        gripper_width  = finger_l + finger_r
        grasped        = contact and (self._grasp_width_min <= gripper_width <= self._grasp_width_max)

        return {
            'q':             q,
            'qd':            qd,
            'tau':           tau,
            'ee_pos':        ee_pose.position,
            'ee_quat':       ee_pose.quaternion,
            'obj_pos':       obj_pos,
            'obj_quat':      obj_quat,
            'gripper_width': gripper_width,
            'contact':       contact,
            'grasped':       grasped,
            'target_pos':    np.array(self.config['task']['target_pos']),
        }

    def get_state(self) -> RobotState:
        """Return full RobotState dataclass."""
        with self._lock:
            q   = self.mj_data.qpos[:ARM_DOF].copy()
            qd  = self.mj_data.qvel[:ARM_DOF].copy()
            tau = self.mj_data.ctrl[:ARM_DOF].copy()

        ee_pose = self.kinematics.forward_kinematics(q)
        return RobotState(q=q, qd=qd, tau=tau, ee_pose=ee_pose)

    @property
    def dt(self) -> float:
        return self.mj_model.opt.timestep * self.steps_per_action
    
    def set_object_pose(self, name: str, pos: np.ndarray, quat: np.ndarray = np.array([1, 0, 0, 0])):
        """Set position and orientation of a named object in world space."""
        body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id == -1:
            raise ValueError(f"Body '{name}' not found in model")

        joint_ids = [j for j in range(self.mj_model.njnt) if self.mj_model.jnt_bodyid[j] == body_id]

        with self._lock:
            if joint_ids and self.mj_model.jnt_type[joint_ids[0]] == mujoco.mjtJoint.mjJNT_FREE:
                qadr = self.mj_model.jnt_qposadr[joint_ids[0]]
                dadr = self.mj_model.jnt_dofadr[joint_ids[0]]
                self.mj_data.qpos[qadr:qadr + 3] = pos
                self.mj_data.qpos[qadr + 3:qadr + 7] = quat
                self.mj_data.qvel[dadr:dadr + 6] = 0.0
            else:
                self.mj_model.body_pos[body_id]  = pos
                self.mj_model.body_quat[body_id] = quat
            mujoco.mj_forward(self.mj_model, self.mj_data)

    def get_object_pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Return current world-space position and quaternion (wxyz) of a named object."""
        body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id == -1:
            raise ValueError(f"Body '{name}' not found in model")

        with self._lock:
            pos  = self.mj_data.xpos[body_id].copy()
            quat = self.mj_data.xquat[body_id].copy()

        return pos, quat


if __name__ == '__main__':
    from src.common.utils import load_yaml
    from src.simulation.rendering import make_renderer
    from src.runner import Runner

    print("Loading simulation...")
    config     = load_yaml("config.yaml")
    sim        = Simulation(config)
    print("Simulation loaded.")

    obs = sim.get_obs()
    print(f"Initial EE position: {obs['ee_pos']}")

    target = Pose(
        position   = obs['ee_pos'] + np.array([0.0, 0.0, 0.1]),
        quaternion = obs['ee_quat'],
    )
    print(f"Target EE position: {target.position}")

    renderer = make_renderer(sim, config.get('rendering', {}))
    runner   = Runner(sim)

    print("Running episode...")
    trajectory = runner.run_episode(target, max_steps=200, renderer=renderer)

    final_obs = trajectory[-1]
    final_err = np.linalg.norm(target.position - final_obs['ee_pos'])
    print(f"Final position error: {final_err:.4f} m")
    print(f"Collected {len(trajectory)} observations.")
    print("Test complete.")