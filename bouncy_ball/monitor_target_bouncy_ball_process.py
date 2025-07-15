import multiprocessing
from threading import current_thread
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.animation import FuncAnimation

class MonitorTargetBouncyBallProcess:
    def __init__(self):
        self.use_animation = True
        self.max_fps = 50.
        self.render_dt = 1. / self.max_fps
        self.radius=7
        self.width=160
        self.height=90
        self.margin = 0
        self.g = -9.81

    def get_scaled_center(self):
        sx = self.radius + self.margin + \
                self.center[0] * (self.width - 2*self.radius - 2*self.margin)
        sy = self.radius + self.margin + \
                self.center[1] * (self.height - 2*self.radius - 2*self.margin)
        return (sx, sy)

    def init_fig(self):
        mpl.style.use('fast')
        mpl.rcParams['toolbar'] = 'None'
        # if not self.use_animation:
        #     plt.ion()
        self.fig = plt.figure()
        plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
        self.fig.canvas.toolbar_visible = False
        self.ax = plt.axes(xlim=(0, self.width), ylim=(0, self.height))
        plt.axis('off')

        self.target = plt.Circle((0, 0), self.radius, color='red')
        self.target.set_animated(True)
        self.ax.add_patch(self.target)

        figManager = plt.get_current_fig_manager()
        figManager.full_screen_toggle()
        plt.show(block=False)
        plt.pause(.1)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.background = self.fig.canvas.copy_from_bbox(self.fig.bbox)

        self.center = np.random.random(2) # in [0, 1)
        self.vel = np.array([0., 0.])

    def animate(self, frame):
        self.vel[1] += self.g * self.render_dt
        self.center = np.clip(self.center + self.vel * self.render_dt, 0, 1)
        if frame == 50:
            self.center = np.random.random(2) # in [0, 1)
            self.vel = np.array([0., 0.])
        scaled_center = self.get_scaled_center()
        self.target.set_center(scaled_center)
        return self.target,

    def render(self):
        self.init_fig()
        prev_time = time.time()
        if self.use_animation:
            animation = FuncAnimation(self.fig, self.animate, frames=100, 
                            interval=int(self.render_dt * 1000), blit=True)
            plt.show()
        else:
            while True:
                self.fig.canvas.restore_region(self.background)
                self.animate(0)
                self.ax.draw_artist(self.target)

                t0 = time.time()
                # self.fig.canvas.draw()
                # self.fig.canvas.flush_events()
                self.fig.canvas.blit(self.fig.bbox)
                # self.fig.canvas.flush_events()
                t1 = (time.time() - t0) * 1000.
                # print(f'{t1:.2f}')

                elapsed_time = time.time() - prev_time
                # print(max(0, self.render_dt - elapsed_time))
                time.sleep(max(0, self.render_dt - elapsed_time))
                # print(f'{elapsed_time * 1000:.0f} ms')
                prev_time = time.time()

    def start(self):
        self.proc = multiprocessing.Process(target=self.render)
        self.proc.daemon = True
        self.proc.start()

    def reset_plot(self):
        return

    def step(self):
        return

if __name__ == "__main__":
    mt = MonitorTargetBouncyBallProcess()
    mt.start()
    while True:
        time.sleep(1)
