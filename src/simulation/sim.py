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
        self.mj_model = mujoco.MjModel.from_xml_path(self.config['scene_path'])
        self.mj_data  = mujoco.MjData(self.mj_model)
        self._lock    = threading.Lock()

        self.steps_per_action = self.config['steps_per_action']
        self.q0 = np.array(self.config['q0'])

        robot_cfg = self.config['robot']
        self.kinematics = RobotKinematics(urdf_path = robot_cfg['urdf_path'], ee_frame_name= robot_cfg['ee_frame_name'])
        self.controller = ImpedanceController(self.config['control'], self.kinematics)

        self._target_pose: Optional[Pose] = None
        self.reset()

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

    def step(self, target_pose: Pose):
        """Run steps_per_action physics steps, applying impedance torques toward target_pose."""
        self._target_pose = target_pose

        for _ in range(self.steps_per_action):
            with self._lock:
                q   = self.mj_data.qpos[:ARM_DOF].copy()
                qd  = self.mj_data.qvel[:ARM_DOF].copy()

                tau = self.controller.compute_control(q, qd, self._target_pose)
                self.mj_data.ctrl[:ARM_DOF] = tau

                mujoco.mj_step(self.mj_model, self.mj_data)

    def get_obs(self) -> dict:
        """Return current observation: joint state + EE pose as flat arrays."""
        with self._lock:
            q   = self.mj_data.qpos[:ARM_DOF].copy()
            qd  = self.mj_data.qvel[:ARM_DOF].copy()
            tau = self.mj_data.ctrl[:ARM_DOF].copy()

        ee_pose = self.kinematics.forward_kinematics(q)

        return {
            'q':       q,
            'qd':      qd,
            'tau':     tau,
            'ee_pos':  ee_pose.position,
            'ee_quat': ee_pose.quaternion,
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


if __name__ == '__main__':
    from src.common.utils import load_yaml
    from src.simulation.rendering import make_renderer
    from src.runner import Runner

    print("Loading simulation...")
    config     = load_yaml("config/global_config.yaml")
    env_config = load_yaml(config["env_config"])
    sim        = Simulation(env_config)
    print("Simulation loaded.")

    obs = sim.get_obs()
    print(f"Initial EE position: {obs['ee_pos']}")

    target = Pose(
        position   = obs['ee_pos'] + np.array([0.0, 0.0, 0.1]),
        quaternion = obs['ee_quat'],
    )
    print(f"Target EE position: {target.position}")

    renderer = make_renderer(sim, env_config.get('rendering', {}))
    runner   = Runner(sim)

    print("Running episode...")
    trajectory = runner.run_episode(target, max_steps=200, renderer=renderer)

    final_obs = trajectory[-1]
    final_err = np.linalg.norm(target.position - final_obs['ee_pos'])
    print(f"Final position error: {final_err:.4f} m")
    print(f"Collected {len(trajectory)} observations.")
    print("Test complete.")