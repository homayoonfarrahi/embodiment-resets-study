from typing import Dict, List, Optional, Union
import numpy as np
from gymnasium import spaces
from gymnasium_robotics.envs.maze.maze_v4 import MazeEnv
from gymnasium_robotics.envs.maze.point_maze import PointMazeEnv
from gymnasium_robotics.envs.maze.maps import U_MAZE

class PointMazeEnvNoReset(PointMazeEnv):
    def __init__(
        self,
        maze_map: List[List[Union[str, int]]] = U_MAZE,
        render_mode: Optional[str] = None,
        reward_type: str = "sparse",
        continuing_task: bool = True,
        reset_target: bool = False,
        **kwargs,
    ):
        super().__init__(
            maze_map=maze_map,
            render_mode=render_mode,
            reward_type=reward_type,
            continuing_task=continuing_task,
            reset_target=reset_target,
            **kwargs,
        )
        # obs_shape: tuple = self.point_env.observation_space.shape
        obs_shape = (2,)
        self.observation_space = spaces.Dict(
            dict(
                observation=spaces.Box(
                    -np.inf, np.inf, shape=obs_shape, dtype="float64"
                ),
                achieved_goal=spaces.Box(-np.inf, np.inf, shape=(2,), dtype="float64"),
                desired_goal=spaces.Box(-np.inf, np.inf, shape=(2,), dtype="float64"),
            )
        )

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        # obs.pop('achieved_goal')
        obs['observation'] = obs['observation'][2:]
        reward = 0.0
        if info['success']:
            reward = 100
        return obs, reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Optional[np.ndarray]]] = None,
    ):
        super(MazeEnv, self).reset(seed=seed)

        if options is None:
            if not hasattr(self, 'goal') or self.goal is None:
                goal = self.generate_target_goal()
                # Add noise to goal position
                self.goal = self.add_xy_position_noise(goal)
            reset_pos = self.generate_reset_pos()
        else:
            if not hasattr(self, 'goal') or self.goal is None:
                if "goal_cell" in options and options["goal_cell"] is not None:
                    # assert that goal cell is valid
                    assert self.maze.map_length > options["goal_cell"][0]
                    assert self.maze.map_width > options["goal_cell"][1]
                    assert (
                        self.maze.maze_map[options["goal_cell"][0]][options["goal_cell"][1]]
                        != 1
                    ), f"Goal can't be placed in a wall cell, {options['goal_cell']}"

                    goal = self.maze.cell_rowcol_to_xy(options["goal_cell"])

                else:
                    goal = self.generate_target_goal()

                # Add noise to goal position
                self.goal = self.add_xy_position_noise(goal)

            if "reset_cell" in options and options["reset_cell"] is not None:
                # assert that goal cell is valid
                assert self.maze.map_length > options["reset_cell"][0]
                assert self.maze.map_width > options["reset_cell"][1]
                assert (
                    self.maze.maze_map[options["reset_cell"][0]][
                        options["reset_cell"][1]
                    ]
                    != 1
                ), f"Reset can't be placed in a wall cell, {options['reset_cell']}"

                reset_pos = self.maze.cell_rowcol_to_xy(options["reset_cell"])

            else:
                reset_pos = self.generate_reset_pos()

        # Update the position of the target site for visualization
        self.update_target_site_pos()
        # Add noise to reset position
        self.reset_pos = self.add_xy_position_noise(reset_pos)

        # Update the position of the target site for visualization
        self.update_target_site_pos()

        self.point_env.init_qpos[:2] = self.reset_pos

        obs, info = self.point_env.reset(seed=seed)
        obs_dict = self._get_obs(obs)
        info["success"] = bool(
            np.linalg.norm(obs_dict["achieved_goal"] - self.goal) <= 0.45
        )

        obs_dict['observation'] = obs_dict['observation'][2:]

        return obs_dict, info

