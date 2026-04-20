import numpy as np
import h5py
from pathlib import Path
from datetime import datetime


class EpisodeLogger:
    def __init__(self, log_dir: str, prefix: str = "episode"):
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._prefix  = prefix
        self._buffer: dict[str, list] = {}
        self._episode = 0

    def reset(self):
        """Clear buffer to start a new episode."""
        self._buffer = {}

    def log_step(self, reward_breakdown: dict, obs: dict, action: np.ndarray):
        """Append one timestep of reward components, key obs fields, and action."""
        entries = {
            'reward/reach':        reward_breakdown['reach'],
            'reward/grasp':        reward_breakdown['grasp'],
            'reward/lift':         reward_breakdown['lift'],
            'reward/place':        reward_breakdown['place'],
            'reward/reg':          reward_breakdown['reg'],
            'reward/total':        reward_breakdown['total'],
            'obs/ee_pos':          obs['ee_pos'],
            'obs/obj_pos':         obs['obj_pos'],
            'obs/gripper_width':   obs['gripper_width'],
            'obs/grasped':         float(obs['grasped']),
            'obs/contact':         float(obs['contact']),
            'action':              action,
        }
        for key, val in entries.items():
            self._buffer.setdefault(key, []).append(np.atleast_1d(np.array(val, dtype=np.float32)))

    def save(self) -> Path:
        """Flush buffer to HDF5 and return the file path."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path      = self._log_dir / f"{self._prefix}_{self._episode:04d}_{timestamp}.h5"

        with h5py.File(path, 'w') as f:
            for key, steps in self._buffer.items():
                f.create_dataset(key, data=np.stack(steps, axis=0), compression='gzip')

        self._episode += 1
        self.reset()
        return path

    def load(self, path: str) -> dict[str, np.ndarray]:
        """Load a saved episode from HDF5 into a dict of arrays."""
        data = {}
        with h5py.File(path, 'r') as f:
            self._load_group(f, '', data)
        return data

    def _load_group(self, group: h5py.Group, prefix: str, out: dict):
        """Recursively load all datasets from an HDF5 group."""
        for key, item in group.items():
            full_key = f"{prefix}/{key}" if prefix else key
            if isinstance(item, h5py.Dataset):
                out[full_key] = item[:]
            else:
                self._load_group(item, full_key, out)
