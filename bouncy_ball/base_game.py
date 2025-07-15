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

class BaseGame:
    def __init__(self, continual=False):
        self.continual = continual
        self.clock = None
        self.window = None
        self.window_size = np.array([1920, 1080])
        # self.window_size = np.array([1280, 720])
        # self.window_size = np.array([640, 360])
        self.lims = np.array([16./9., 1.])
        self.max_fps = 50.
        self.render_dt = 1. / self.max_fps
        self.vel = np.array([0., 0.])
        self.radius=.1
        self.margin = 0
        self.bounce_factor = .5
        self.cursor_rad = .3
        self.prev_cursor_pos = None
        self.mouse_control = False
        self.control_type = 'acceleration'
        self.g = 9.81
        self.ball_color = 'olivedrab'
        self.total_steps = multiprocessing.Array('f', [0.])
        self.ball_pos = multiprocessing.Array('f', np.random.random(2)) # in [0, 1)
        self.acceleration = multiprocessing.Array('f', [0., 0.])
        self.delta_pos = multiprocessing.Array('f', [0., 0.])
        self.ball_in_goal = multiprocessing.Array('f', [0.])
        self.ball_grasped = multiprocessing.Array('f', [0.])
        self.reset_signal = multiprocessing.Array('f', [0.])
        self.cursor_pos = multiprocessing.Array('f', [0., 0.])

    def pixel_radius(self):
        return self.radius * self.window_size[1] / self.lims[1]

    def world_len_to_pixel(self, world_len):
        pr = self.pixel_radius()
        return world_len / self.lims * (self.window_size - 2*pr - 2*self.margin)

    def to_pixel_2d(self, world_p):
        p_rad = self.pixel_radius()
        px = p_rad + self.margin + \
                world_p[0] / self.lims[0] * (self.window_size[0] - 2*p_rad - 2*self.margin)
        py = p_rad + self.margin + \
                world_p[1] / self.lims[1] * (self.window_size[1] - 2*p_rad - 2*self.margin)
        return [px, py]

    def to_world_2d(self, pix_p):
        p_rad = self.pixel_radius()
        wx = (pix_p[0] - p_rad - self.margin) \
            * self.lims[0] / (self.window_size[0] - 2*p_rad - 2*self.margin)
        wy = (pix_p[1] - p_rad - self.margin) \
            * self.lims[1] / (self.window_size[1] - 2*p_rad - 2*self.margin)
        return [wx, wy]

    def animate(self):
        if self.mouse_control:
            world_mouse = self.to_world_2d(pygame.mouse.get_pos())
            self.cursor_pos[:] = world_mouse

        if self.prev_cursor_pos is None:
            self.prev_cursor_pos = np.array(self.cursor_pos)

        self.delta_pos[:] = np.array(self.cursor_pos) - np.array(self.prev_cursor_pos)
        self.prev_cursor_pos = np.array(self.cursor_pos)

        cp, bp = np.array(self.cursor_pos), np.array(self.ball_pos)
        cb_dist = np.linalg.norm(cp - bp)
        self.ball_in_cursor = False
        self.cursor_color = (0, 0, 128)
        if cb_dist < self.cursor_rad:
            self.ball_in_cursor = True
            self.cursor_color = (0, 128, 128)

        # apply external acceleration or position control
        if self.control_type == 'acceleration' and self.ball_in_cursor:
            accel = np.array(self.delta_pos)
            accel[0] *= 10
            accel[1] *= 100
            self.vel += accel * self.render_dt
            # if np.linalg.norm(accel) > 1e-4:
            #     accel[0] *= 10
            #     self.vel = .002*accel / self.render_dt
        elif self.control_type == 'position' and self.ball_grasped[0] == 1:
            self.ball_pos[:] = np.array(self.ball_pos) + np.array(self.delta_pos)

        if self.ball_grasped[0] == 0 or self.control_type == 'acceleration':
            self.vel[1] += self.g * self.render_dt
        else:
            self.vel[:] = [0, 0]

        new_center = self.ball_pos + self.vel * self.render_dt
        if new_center[0] <= 0 or new_center[0] >= self.lims[0]:
            self.vel[0] *= -self.bounce_factor
        if new_center[1] <= 0 or new_center[1] >= self.lims[1]:
            self.vel[1] *= -self.bounce_factor
        self.ball_pos[0] = np.clip(new_center[0], 0, self.lims[0])
        self.ball_pos[1] = np.clip(new_center[1], 0, self.lims[1])

    def draw_cursor(self, canvas):
        x, y = self.cursor_pos[0], self.cursor_pos[1]
        px, py = self.to_pixel_2d((x, y))
        p_rad = self.world_len_to_pixel([self.cursor_rad, 0])[0]
        p_wid = 100
        if -p_rad <= px <= self.window_size[0] + p_rad \
            and -p_rad <= py <= self.window_size[1] + p_rad:
            # pygame.draw.line(canvas, self.cursor_color, (px - p_rad, py), (px + p_rad, py), width=p_wid)
            # pygame.draw.line(canvas, self.cursor_color, (px, py - p_rad), (px, py + p_rad), width=p_wid)
            pygame.draw.circle(canvas, self.cursor_color, (px, py), p_rad, width=p_wid)

    def render(self):
        # register cleanup method
        signal.signal(signal.SIGTERM, self.cleanup)

        if self.window is None:
            pygame.init()
            pygame.display.init()
            flags = pygame.DOUBLEBUF | pygame.HWSURFACE | pygame.NOFRAME #| pygame.FULLSCREEN #| pygame.OPENGL 
            self.window = pygame.display.set_mode(self.window_size, flags, display=1)

        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface(self.window_size, pygame.SRCALPHA)
        self.reset()
        while True:
            if self.reset_signal[0] == 1:
                self.game_reset()
                self.reset_signal[0] = 0

            if self.continual:
                self.update_task_params()

            canvas.fill((255, 255, 255))
            self.animate()
            self.game_step_draw(canvas)

            self.draw_cursor(canvas)

            # draw the ball
            pos = self.to_pixel_2d(self.ball_pos)
            rad = self.pixel_radius()
            pygame.draw.circle(
                canvas,
                self.ball_color,
                pos,
                rad,
            )

            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.flip()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.max_fps)

    def update_task_params(self):
        pass

    def set_cursor_pos(self, cursor_pos):
        if not self.mouse_control:
            px = cursor_pos[0] * self.window_size[0]
            py = (1 - cursor_pos[1]) * self.window_size[1]
            world_p = self.to_world_2d((px, py))
            self.cursor_pos[:] = world_p

    def push_ball(self, ef_delta, target_size):
        # if target_size >= .001:
        if self.control_type == 'acceleration':
            self.acceleration[0] = ef_delta[0] * 1000
            self.acceleration[1] = ef_delta[2] * -2000
        # else:
        #     self.acceleration[:] = [0, 0]

        # ball grasping logic
        cp, bp = np.array(self.cursor_pos), np.array(self.ball_pos)
        cb_dist = np.linalg.norm(cp - bp)
        if cb_dist < .8 * self.cursor_rad:
            self.ball_grasped[0] = 1
        elif cb_dist > 1 * self.cursor_rad:
            self.ball_grasped[0] = 0

    def set_total_steps(self, ts):
        self.total_steps[0] = ts

    def cleanup(self, signum, frame):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
        sys.exit(0)

    def start(self):
        self.proc = multiprocessing.Process(target=self.render)
        self.proc.daemon = True
        self.proc.start()

    def reset(self):
        self.acceleration[:] = [0, 0]
        self.delta_pos[:] = [0, 0]
        self.ball_grasped[0] = 0
        self.reset_signal[0] = 1

    def game_reset(self):
        pass

    def game_step_draw(self, canvas):
        pass

    def step(self):
        return

    def reset_plot(self):
        return

if __name__ == "__main__":
    game = BaseGame()
    game.start()
    t0 = time.time()
    while True:
        if (time.time() - t0) % 4 > 2:
            game.acceleration[1] = -10
        else:
            game.acceleration[1] = 0
        time.sleep(1)
