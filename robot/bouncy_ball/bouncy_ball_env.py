import multiprocessing
from relod.envs.visual_ur5_reacher.reacher_env_min_time import ReacherEnv
import numpy as np
from senseact.utils import NormalizedEnv
from senseact.devices.ur import ur_utils
import cv2, math
from statistics import mean
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from tqdm import tqdm
import torch

from robot.bouncy_ball.ball_pull_up_game import BallPullUpGame
from robot.bouncy_ball.ball_in_goal_game import BallInGoalGame

def get_mask(image):
    image = np.transpose(image, [1,2,0])
    image = image[:,:,-3:]

    lower = [0, 0, 120]
    upper = [50, 50, 255]
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")

    mask = cv2.inRange(image, lower, upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.fillPoly(mask, pts=contours, color=(255, 255, 255))
    
    return mask

def get_center(image):
    mask = get_mask(image)

    m = cv2.moments(mask)
    if math.isclose(m["m00"], 0.0, rel_tol=1e-6, abs_tol=0.0):
        x = 0
        y = 0
    else:
        x = int(m["m10"] / m["m00"])
        y = int(m["m01"] / m["m00"])

    cv2.circle(mask, (x, y), 1, (0,0,0), -1)
    cv2.imshow('mask', mask)
    cv2.waitKey(1)

    width = len(mask[0])
    height = len(mask)
    x = -1.0 + x/width*2
    y = -1.0 + y/height*2
    return x, y

class BouncyBallEnv:
    def __init__(self,
                 setup='Visual-UR5-min-time',
                 ip='129.128.159.210',
                 seed=9,
                 camera_id=0,
                 image_width=160,
                 image_height=90,
                 target_type='size',
                 image_history=3,
                 joint_history=1,
                 episode_length=30,
                 dt=0.04,
                 size_tol=0.015,
                 center_tol=0.1,
                 reward_tol=1.0,
                 continual=False,
                ):
        self._image_width = image_width
        self._image_height = image_height
        self._dt = dt
        self._target_type = target_type
        self._size_tol = size_tol
        self._center_tol = center_tol
        self._reward_tol = reward_tol
        self.total_steps = 0

        # state
        np.random.seed(seed)
        rand_state = np.random.get_state()
        env = ReacherEnv(
            setup=setup,
            host=ip,
            dof=5,
            camera_id=camera_id,
            image_width=image_width,
            image_height=image_height,
            channel_first=True,
            control_type="velocity",
            target_type="reaching",
            image_history=image_history,
            joint_history=joint_history,
            reset_type="zero",
            reward_type="dense",
            derivative_type="none",
            deriv_action_max=5,
            first_deriv_max=2,
            accel_max=1.4,
            speed_max=2,
            speedj_a=1.4,
            episode_length_time=episode_length,
            episode_length_step=None,
            actuation_sync_period=1,
            dt=dt,
            run_mode="multiprocess",
            # run_mode="singlethread",
            rllab_box=False,
            movej_t=1.5,
            delay=0.0,
            random_state=rand_state
        )

        self.game = BallPullUpGame(continual=continual)
        # self.game = BallInGoalGame(continual=continual)
        self.game.start()
        self.prev_ef_pos = None

        self._env = NormalizedEnv(env)
        env.start()

        self._reset = False

    def _compute_target_size(self, image):
        mask = get_mask(image)

        target_size = np.sum(mask/255.) / mask.size

        return target_size

    def _compute_target_offset(self, image, target_location):
        (x, y) = get_center(image)

        return abs(x-target_location[0]), abs(y-target_location[1])

    def _compute_reward(self, image, joint):
        """Computes reward at a given time step.
        Returns:
            A float reward.
        """
        image = np.transpose(image, [1,2,0])
        image = image[:, :, -3:]
        lower = [0, 0, 120]
        upper = [50, 50, 255]
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")

        mask = cv2.inRange(image, lower, upper)
        cv2.imshow('', mask)
        cv2.waitKey(1)
        
        size_x, size_y = mask.shape
        # reward for reaching task, may not be suitable for tracking
        if 255 in mask:
            xs, ys = np.where(mask == 255.)
            reward_x = 1 / 2  - np.abs(xs - int(size_x / 2)) / size_x
            reward_y = 1 / 2 - np.abs(ys - int(size_y / 2)) / size_y
            reward = np.sum(reward_x * reward_y) / self._image_width / self._image_height
        else:
            reward = 0
        reward *= 800
        reward = np.clip(reward, 0, 4)

        '''
        When the joint 4 is perpendicular to the mounting ground:
            joint 0 + joint 4 == 0
            joint 1 + joint 2 + joint 3 == -pi
        '''
        # chagne
        # scale = (np.abs(joint[0] + joint[4]) + np.abs(np.pi + np.sum(joint[1:4])))
        # return reward - scale
        return reward 

    @property
    def image_space(self):
        return self._env.observation_space['image']

    @property
    def proprioception_space(self):
        return self._env.observation_space['joint']

    @property
    def action_space(self):
        return self._env.action_space

    def reset(self):
        self.prev_ef_pos = None
        self.game.reset()

        obs_dict = self._env.reset()
        image = obs_dict['image']
        prop = obs_dict['joint']

        self._reset = True

        return image, prop

    def step(self, action):
        assert self._reset
        obs_dict, reward, done, _ = self._env.step(action)
        image = obs_dict['image']
        prop = obs_dict['joint']
        # done = 0
        info = {}

        # if self._target_type == 'size':
        #     done = self._compute_target_size(image) >= self._size_tol
        # elif self._target_type == 'center':
        #     offset = self._compute_target_offset(image, (0, 0))
        #     done = offset[0] <= self._center_tol*2 and offset[1] <= self._center_tol*2
        # elif self._target_type == 'reward':
        #     r = self._compute_reward(image, prop)
        #     # print('r:',r)
        #     done = r >= self._reward_tol
        # elif self._target_type == 'size_center':
        #     offset = self._compute_target_offset(image, (0, 0))
        #     done = (self._compute_target_size(image) >= self._size_tol) and \
        #             (offset[0] <= self._center_tol*2 and offset[1] <= self._center_tol*2)
        # else:
        #     raise NotImplementedError()

        if done:
            self._reset = False
            self._env.stop_arm()

        # reward = -1
        # reward = 0
        # reward = self.game.ball_in_goal[0]
        reward = -10 if self.game.ball_in_goal[0] == 0 else 1

        # calculate end-effector movement and apply acceleration to the ball
        ts = self._compute_target_size(image)
        # q = np.append(self._env._q_[-1], [0], axis=0)
        q = np.append(prop[:len(self._env._joint_indices)], [0], axis=0)
        transf = ur_utils.forward(q, self._env._ik_params)
        np.set_printoptions(formatter={'float': lambda x: f'{x:.2f}'})
        ef_pos = transf[:3, 3]
        ef_dir = transf[:3, 0]
        # print(ef_pos)
        # print(ef_dir)
        self.game.set_total_steps(self.total_steps)
        cursor_pos = self.calculate_cursor_pos(ef_pos, ef_dir)
        self.game.set_cursor_pos(cursor_pos)
        if self.prev_ef_pos is None:
            self.prev_ef_pos = ef_pos
        ef_delta = ef_pos - self.prev_ef_pos
        self.game.push_ball(ef_delta, ts)
        self.prev_ef_pos = ef_pos

        return image, prop, reward, done, info

    def get_reset_reward(self):
        return self.game.ball_in_goal[0]

    def calculate_cursor_pos(self, ef_pos, ef_dir):
        # Calculates where the camera is pointing on the monitor
        # assuming monitor is parallel to the x-z plane.
        # Monitor bottom-left corner is at (-.37, .74, .38).
        # Monitor width and height (x-z) is (.525, .295).
        # Camera is .07 above the end-effector
        if ef_dir[1] < 1e-4:
            return np.array([-10, -10])

        # m_pos = np.array([-.37, .74, .38])
        m_pos = np.array([-.37, .85, .38])
        m_size = np.array([.525, .295])
        cam_ef_dist = .07

        # derive camera position from ef_pos
        x_dir = np.array([1, 0, 0])
        move_dir = np.cross(x_dir, ef_dir)
        move_dir = move_dir / np.linalg.norm(move_dir)
        cam_pos = ef_pos + cam_ef_dist * move_dir

        t = (.74 - cam_pos[1]) / ef_dir[1]
        c_pos = cam_pos + ef_dir * t
        c_pos_scaled = np.array([
            (c_pos[0] - m_pos[0]) / m_size[0],
            (c_pos[2] - m_pos[2]) / m_size[1],
        ])
        return c_pos_scaled

    def close(self):
        self._env.close()

