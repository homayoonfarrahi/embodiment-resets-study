from pathlib import Path
import numpy as np
from gymnasium.envs.mujoco.half_cheetah_v5 import HalfCheetahEnv

class HalfCheetahSparse(HalfCheetahEnv):
    def __init__(self, xml_file='half_cheetah_cntg.xml', render_mode=None, term_cost=50.):
        xml_file = str((Path(__file__).parent / xml_file).resolve())
        super().__init__(xml_file=xml_file, render_mode=render_mode)
        self.term_cost = term_cost
        self.pos_min, self.pos_max = np.inf, -np.inf
        self.dist_threshold = 3

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        # ctrl_cost = self.control_cost(action)
        # forward_reward = info['x_velocity']
        # tc = float(terminated) * self.term_cost
        # reward = forward_reward - ctrl_cost

        reward = 0
        if info['x_position'] < self.pos_min:
            self.pos_min = info['x_position']
        if info['x_position'] > self.pos_max:
            self.pos_max = info['x_position']

        if info['x_position'] > self.pos_min + self.dist_threshold:
            reward = 100
            self.pos_min = info['x_position']
        elif info['x_position'] < self.pos_max - self.dist_threshold:
            reward = -100
            self.pos_max = info['x_position']

        return observation, reward, terminated, truncated, info
