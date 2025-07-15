import os
from pathlib import Path
from typing import Optional
from gymnasium import spaces
from gymnasium_robotics.envs.robot_env import BaseRobotEnv
import numpy as np
from gymnasium.utils.ezpickle import EzPickle
from gymnasium_robotics.envs.fetch import MujocoFetchEnv

DEFAULT_CAMERA_CONFIG = {
    "distance": 2.5,
    "azimuth": 132.0,
    "elevation": -14.0,
    "lookat": np.array([1.3, 0.75, 0.55]),
}

class MujocoFetchEnvNoReset(MujocoFetchEnv):
    def __init__(self, default_camera_config: dict = DEFAULT_CAMERA_CONFIG, **kwargs):
        super().__init__(default_camera_config=default_camera_config, **kwargs)
        self.observation_space = spaces.Dict(
            dict(
                desired_goal=spaces.Box(
                    -np.inf, np.inf, shape=(3,), dtype="float64"
                ),
                achieved_goal=spaces.Box(
                    -np.inf, np.inf, shape=(3,), dtype="float64"
                ),
                observation=spaces.Box(
                    -np.inf, np.inf, shape=(7,), dtype="float64"
                ),
            )
        )

    def _reset_sim(self):
        # Reset buffers for joint states, actuators, warm-start, control buffers etc.
        self._mujoco.mj_resetData(self.model, self.data)

        self.data.time = self.initial_time
        self.data.qpos[:] = np.copy(self.initial_qpos)
        self.data.qvel[:] = np.copy(self.initial_qvel)
        if self.model.na != 0:
            self.data.act[:] = None

        # Randomize start position of object.
        if self.has_object:
            object_xpos = self.initial_gripper_xpos[:2]
            while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
                object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(
                    -self.obj_range, self.obj_range, size=2
                )
            object_qpos = self._utils.get_joint_qpos(
                self.model, self.data, "object0:joint"
            )
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self._utils.set_joint_qpos(
                self.model, self.data, "object0:joint", object_qpos
            )

        self._mujoco.mj_forward(self.model, self.data)
        return True

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        reward = 0.0
        if info['is_success']:
            reward = 100
            self.goal = self._sample_goal()
            obs = self._get_obs()
        obs['observation'] = obs['observation'][3:]
        return obs, reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super(BaseRobotEnv, self).reset(seed=seed)
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        if self.goal.shape[0] == 0:
            self.goal = self._sample_goal().copy()
        obs = self._get_obs()
        if self.render_mode == "human":
            self.render()

        obs['observation'] = obs['observation'][3:]

        return obs, {}

class MujocoFetchReachEnvNoReset(MujocoFetchEnvNoReset, EzPickle):
    def __init__(self, reward_type: str = "sparse", **kwargs):
        initial_qpos = {
            "robot0:slide0": 0.4049,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
        }
        xml_file = 'fetch_reach_no_reset.xml'
        xml_file = str((Path(__file__).parent / xml_file).resolve())
        # xml_file = os.path.join("fetch", "reach.xml")
        print(f"Using XML file: {xml_file}")
        MujocoFetchEnvNoReset.__init__(
            self,
            model_path=xml_file,
            has_object=False,
            block_gripper=True,
            n_substeps=20,
            gripper_extra_height=0.2,
            target_in_the_air=True,
            target_offset=0.0,
            obj_range=0.15,
            target_range=0.15,
            distance_threshold=0.1,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
            **kwargs,
        )
        EzPickle.__init__(self, reward_type=reward_type, **kwargs)

