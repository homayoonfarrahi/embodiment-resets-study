from pathlib import Path
from gymnasium.envs.mujoco.half_cheetah_v5 import HalfCheetahEnv

class HalfCheetahCntg(HalfCheetahEnv):
    def __init__(self, xml_file='half_cheetah_cntg.xml', render_mode=None, term_cost=50.):
        xml_file = str((Path(__file__).parent / xml_file).resolve())
        super().__init__(xml_file=xml_file, render_mode=render_mode)
        self.term_cost = term_cost

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        ctrl_cost = self.control_cost(action)
        forward_reward = info['x_velocity']
        tc = float(terminated) * self.term_cost
        reward = forward_reward - ctrl_cost - tc
        
        return observation, reward, terminated, truncated, info
