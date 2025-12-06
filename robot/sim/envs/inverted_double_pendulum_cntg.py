from gymnasium.envs.mujoco.inverted_double_pendulum_v4 import InvertedDoublePendulumEnv

class InvertedDoublePendulumCntg(InvertedDoublePendulumEnv):
    def __init__(self, render_mode=None, term_cost=500.):
        super().__init__(render_mode=render_mode)
        self.term_cost = term_cost

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        info['og_reward'] = reward

        x, _, y = self.data.site_xpos[0]
        dist_penalty = 0.01 * x**2 + (y - 2) ** 2
        v1, v2 = self.data.qvel[1:3]
        vel_penalty = 1e-3 * v1**2 + 5e-3 * v2**2
        tc = float(terminated) * self.term_cost
        reward = - dist_penalty - vel_penalty - tc
        return observation, reward, terminated, truncated, info
