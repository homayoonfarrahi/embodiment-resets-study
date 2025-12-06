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

class BallPullUpGame(BaseGame):
    def __init__(self, continual=False):
        super().__init__(continual=continual)
        self.g = .1
        self.ball_color = 'red'
        self.bounce_factor = .3
        self.goal_boundary = .85
        self.mouse_control = False

    def game_reset(self):
        return super().game_reset()

    def game_step_draw(self, canvas):
        super().game_step_draw(canvas)
        self.ball_in_goal[0] = 1 if self.ball_pos[1] <= self.goal_boundary else 0

        # draw the boundary line
        boundary_y = self.to_pixel_2d([0, self.goal_boundary])[1]
        boundary_color = (0, 255, 0) if self.ball_in_goal[0] else (0, 0, 0)
        pygame.draw.line(
            canvas,
            boundary_color,
            (0, boundary_y),
            (self.window_size[0], boundary_y),
            width=30
        )

    def update_task_params(self):
        t = self.total_steps[0]
        c1, c2 = 'olivedrab', 'tomato'
        g, c = .1, c1
        # if t <= 2e6:
        #     g, c = .1, c1
        # elif (t - 2e6) % 2e6 <= 1e6:
        #     g, c = .3, c2
        # else:
        #     g, c = .1, c1

        if t % 1e6 <= 500000:
            g, c = .1, c1
        else:
            g, c = .3, c2

        self.g = g
        self.ball_color = c


if __name__ == "__main__":
    game = BallPullUpGame()
    game.start()
    t0 = time.time()
    while True:
        if (time.time() - t0) % 4 > 2:
            game.acceleration[1] = -10
        else:
            game.acceleration[1] = 0
        time.sleep(1)
