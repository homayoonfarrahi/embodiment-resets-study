from pathlib import Path
from gymnasium.envs.mujoco.ant_v4 import AntEnv

class AntCntg(AntEnv):
    def __init__(self, render_mode=None, term_cost=50., terminate_when_unhealthy=True, corridor=True):
        xml_file = 'ant_cntg_corridor.xml' if corridor else 'ant_cntg.xml'
        xml_file = str((Path(__file__).parent / xml_file).resolve())
        super().__init__(xml_file=xml_file, render_mode=render_mode, terminate_when_unhealthy=terminate_when_unhealthy, healthy_z_range=(0.3, 2.0))
        self.term_cost = term_cost

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        info['og_reward'] = reward

        ctrl_cost = self.control_cost(action)
        forward_reward = info['x_velocity']
        tc = float(terminated) * self.term_cost
        reward = forward_reward - ctrl_cost - tc
        if self._use_contact_forces:
            reward -= self.contact_cost
        return observation, reward, terminated, truncated, info
