from gymnasium.envs.mujoco.hopper_v4 import HopperEnv

class HopperCntg(HopperEnv):
    def __init__(self, render_mode=None, term_cost=500., term_cost_coef=1.0, terminate_when_unhealthy=True):
        super().__init__(render_mode=render_mode, terminate_when_unhealthy=terminate_when_unhealthy)
        self.term_cost = term_cost
        self.term_cost_coef = term_cost_coef

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        info['og_reward'] = reward

        ctrl_cost = self.control_cost(action)
        forward_reward = self._forward_reward_weight * info['x_velocity']
        tc = float(terminated) * self.term_cost
        # tc = float(terminated) * (info['x_position'] / self.dt) \
        #                 * self._forward_reward_weight * self.term_cost_coef
        reward = forward_reward - ctrl_cost - tc

        if not self._terminate_when_unhealthy:
            reward += int(self.is_healthy) * 1

        return observation, reward, terminated, truncated, info
