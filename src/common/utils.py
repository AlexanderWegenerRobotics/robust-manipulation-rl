import yaml
from pathlib import Path
import numpy as np

def load_yaml(path: str):
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data