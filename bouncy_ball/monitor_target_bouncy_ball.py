import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.animation import FuncAnimation

class MonitorTargetBouncyBall:
    def __init__(self):
        self.radius=7
        self.width=160
        self.height=90
        self.margin = 0
        self.g = -.981
        self.use_animation = False
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

    def get_scaled_center(self):
        sx = self.radius + self.margin + \
                self.center[0] * (self.width - 2*self.radius - 2*self.margin)
        sy = self.radius + self.margin + \
                self.center[1] * (self.height - 2*self.radius - 2*self.margin)
        return (sx, sy)

    def reset_plot(self):
        self.fig.canvas.restore_region(self.background)
        self.vel = np.array([0., 0.])
        self.center = np.random.random(2) # in [0, 1)

        scaled_center = self.get_scaled_center()
        self.target.set_center(scaled_center)
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        time.sleep(0.032)

        if self.use_animation:
            self.anim = FuncAnimation(self.fig, self.animate, init_func=None,
                         frames=200, interval=20, blit=True)
            plt.show()
            # plt.show(block=False)

    def animate(self, frame):
        print(frame, self.vel)
        dt = .04
        self.vel[1] += self.g * dt
        self.center = np.clip(self.center + self.vel * dt, 0, 1)
        scaled_center = self.get_scaled_center()
        self.target.set_center(scaled_center)

        # t0 = time.time()
        # # self.fig.canvas.draw()
        # self.fig.canvas.flush_events()
        # t1 = (time.time() - t0) * 1000.
        # print(f'{t1:.2f}')
        return self.target,

    def step(self):
        self.fig.canvas.restore_region(self.background)
        dt = .04
        self.vel[1] += self.g * dt
        self.center = np.clip(self.center + self.vel * dt, 0, 1)
        scaled_center = self.get_scaled_center()
        self.target.set_center(scaled_center)
        self.ax.draw_artist(self.target)

        t0 = time.time()
        # self.fig.canvas.draw()
        # self.fig.canvas.flush_events()
        self.fig.canvas.blit(self.fig.bbox)
        # self.fig.canvas.flush_events()
        t1 = (time.time() - t0) * 1000.
        print(f'{t1:.2f}')

if __name__ == "__main__":
    mt = MonitorTargetBouncyBall()
    mt.reset_plot()
    for j in range(1000):
        mt.step()
        # time.sleep(.04)
