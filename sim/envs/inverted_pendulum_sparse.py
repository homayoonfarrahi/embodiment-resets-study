from pathlib import Path
import numpy as np
from gymnasium.envs.mujoco.inverted_pendulum_v5 import InvertedPendulumEnv

class InvertedPendulumSparse(InvertedPendulumEnv):
    def __init__(self, render_mode=None, terminate_when_unhealthy=False, xml_file='inverted_pendulum_no_reset.xml'):
        xml_file = str((Path(__file__).parent / xml_file).resolve())
        super().__init__(xml_file=xml_file, render_mode=render_mode)
        self.terminate_when_unhealthy = terminate_when_unhealthy

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        reward = float(not terminated)
        terminated = terminated if self.terminate_when_unhealthy else False
        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        theta = ((self.data.qpos[1] + np.pi) % (2 * np.pi)) - np.pi
        return np.concatenate([[self.data.qpos[0]], [theta], self.data.qvel]).ravel()
