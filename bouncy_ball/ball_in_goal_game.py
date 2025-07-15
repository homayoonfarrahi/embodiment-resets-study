import sys
import multiprocessing
from threading import current_thread
import time
import numpy as np
import pygame
import signal
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.animation import FuncAnimation

from robot.bouncy_ball.base_game import BaseGame

class BallInGoalGame(BaseGame):
    def __init__(self):
        super().__init__()
        self.goal_id = 0
        self.goal_positions = np.array([[.15, .15], [1, .15]])
        self.goal_size = np.array([.65, .5])
        self.goal_line_wid = 90
        self.control_type = 'position'

    def game_reset(self):
        super().game_reset()
        self.goal_id = np.random.choice([0, 1])
        self.ball_pos[0] = self.lims[0] * np.random.random()

    def game_step_draw(self, canvas):
        super().game_step_draw(canvas)
        self.ball_in_goal[0] = int(self.is_ball_in_goal())

        # draw the goal
        goal_pos = self.to_pixel_2d(self.goal_positions[self.goal_id])
        goal_size = self.world_len_to_pixel(self.goal_size)
        rect = pygame.Rect(goal_pos, goal_size)
        color = (0, 0, 0) if self.ball_in_goal[0] == 0 else (0, 50, 0)
        pygame.draw.rect(canvas, color, rect, self.goal_line_wid)

    def is_ball_in_goal(self):
        gp = self.goal_positions[self.goal_id]
        gs = self.goal_size
        in_goal = False
        if (gp[0] <= self.ball_pos[0] <= gp[0] + gs[0]) \
                and (gp[1] <= self.ball_pos[1] <= gp[1] + gs[1]):
            in_goal = True
        # print(gp, gs, self.ball_pos[1], in_goal)
        return in_goal

if __name__ == "__main__":
    game = BallInGoalGame()
    game.start()
    t0 = time.time()
    while True:
        if (time.time() - t0) % 4 > 2:
            game.acceleration[1] = -10
        else:
            game.acceleration[1] = 0
        time.sleep(.5)
        game.reset()
