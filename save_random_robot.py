"""
Save a random (unevolved, untrained) robot for the "before evolution" visualizer demo.
The robot will have NO trained control params, so it will flop around in the visualizer.
"""

import numpy as np
from robot import sample_robot
from utils import load_config

if __name__ == "__main__":
    config = load_config("config.yaml")
    np.random.seed(config["seed"])

    robot = sample_robot()
    robot["max_n_masses"] = robot["n_masses"]
    robot["max_n_springs"] = robot["n_springs"]
    # NO control_params = visualizer uses random weights = poor movement

    np.save("random_robot.npy", robot)
    print("Saved random_robot.npy (untrained - will move poorly in visualizer)")
    print("Run: python visualizer.py --input random_robot.npy --port 5001")
