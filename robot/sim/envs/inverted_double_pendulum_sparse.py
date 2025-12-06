from gymnasium.envs.mujoco.inverted_double_pendulum_v4 import InvertedDoublePendulumEnv

class InvertedDoublePendulumSparse(InvertedDoublePendulumEnv):
    def __init__(self, render_mode=None, terminate_when_unhealthy=False):
        super().__init__(render_mode=render_mode)
        self.terminate_when_unhealthy = terminate_when_unhealthy

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        _, _, y = self.data.site_xpos[0]
        terminated = bool(y <= 1) if self.terminate_when_unhealthy else False
        # reward = reward - 50 if self.terminate_when_unhealthy and terminated else reward
        reward = float(y >= 1) * 1
        return observation, reward, terminated, truncated, info
