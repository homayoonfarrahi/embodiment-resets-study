from gymnasium.envs.mujoco.walker2d_v4 import Walker2dEnv

class Walker2dCntg(Walker2dEnv):
    def __init__(self, render_mode=None, term_cost=100., terminate_when_unhealthy=True):
        super().__init__(render_mode=render_mode, terminate_when_unhealthy=terminate_when_unhealthy)
        self.term_cost = term_cost

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        info['og_reward'] = reward

        ctrl_cost = self.control_cost(action)
        forward_reward = self._forward_reward_weight * info['x_velocity']
        tc = float(terminated) * self.term_cost
        reward = forward_reward - ctrl_cost - tc
        return observation, reward, terminated, truncated, info
