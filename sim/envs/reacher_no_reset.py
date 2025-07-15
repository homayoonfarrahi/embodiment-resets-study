from pathlib import Path
import numpy as np
from gymnasium.envs.mujoco.reacher_v5 import ReacherEnv

class ReacherEnvNoReset(ReacherEnv):
    def __init__(self, render_mode=None, sparse_reward=True, joint_limit=True):
        self.sparse_reward = sparse_reward
        xml_file = 'reacher_no_reset_joint_limit.xml' if joint_limit else 'reacher_no_reset.xml'
        xml_file = str((Path(__file__).parent / xml_file).resolve())
        super().__init__(xml_file=xml_file, render_mode=render_mode)
        self.goal = None

    def step(self, a):
        vec = self.get_body_com("fingertip") - self.get_body_com("target")
        reward_dist = -np.linalg.norm(vec)
        reward_ctrl = -np.square(a).sum()
        reward = reward_dist + reward_ctrl
        og_reward = reward.item()
        if self.sparse_reward:
            reward = 0

        self.do_simulation(a, self.frame_skip)
        if self.render_mode == "human":
            self.render()

        newvec = self.get_body_com("fingertip") - self.get_body_com("target")
        newvel = self.data.body('fingertip').cvel[3:]
        if np.linalg.norm(newvec) < .05 and np.linalg.norm(newvel) < .05:
            self.reset_goal()
            self.data.qpos[-2:] = self.goal
            self.set_state(self.data.qpos, self.data.qvel)
            reward += 100

        ob = self._get_obs()
        results = (
            ob,
            reward,
            False,
            False,
            dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl, og_reward=og_reward),
        )

        return results

    def _reset_simulation(self):
        # mujoco.mj_resetData(self.model, self.data)
        pass

    def reset_model(self):
        qpos, qvel = self.data.qpos, self.data.qvel

        if self.goal is None:
            self.reset_goal()
        qpos[-2:], qvel[-2:] = self.goal, 0
        qpos[:-2], qvel[:-2] = self.reset_arm()

        self.set_state(qpos, qvel)
        observation = self._get_obs()
        return observation

    def reset_arm(self):
        qpos = (
            self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
            + self.init_qpos
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )

        return qpos[:-2], qvel[:-2]

    def reset_goal(self):
        while True:
            self.goal = self.np_random.uniform(low=-0.2, high=0.2, size=2)
            if np.linalg.norm(self.goal) < 0.2:
                break

