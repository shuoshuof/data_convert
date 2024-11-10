# -*- coding: utf-8 -*-
"""
@Time ： 2024/11/9 18:01
@Auth ： shuoshuof
@File ：base_visualizer.py
@Project ：data_convert
"""
from abc import ABC,abstractmethod
import math

from vedo import *

settings.default_backend = "vtk"
settings.immediate_rendering = False

class BaseVedoVisualizer(ABC):
    def __init__(self,num_subplots,**kwargs):
        self.num_subplots = num_subplots
        self._init_plotter()

        self.pause_button = self.plotter.at(num_subplots
                                            ).add_button(
            self._stop_button_func,
            states=["\u23F5 Play", "\u23F8 Pause"],
            font="Kanopus",
            size=32,
        )

        self.counter = 0
        self.timer_id = None

        self.plotter.add_callback('timer', self.loop, enable_picking=False)

    def _init_plotter(self):
        cols = int(math.sqrt(self.num_subplots))
        rows = int(math.ceil(self.num_subplots / cols))
        self.plotter = Plotter(shape=(cols, rows), sharecam=False)

    def _stop_button_func(self, obj, ename):
        if self.timer_id is not None:
            self.plotter.timer_callback("destroy",self.timer_id)
        if "Play" in self.pause_button.status():
            self.timer_id = self.plotter.timer_callback("create",dt=5)
        self.pause_button.switch()

    @abstractmethod
    def loop_func(self,event,**kwargs):
        raise NotImplementedError

    def loop(self,event):
        self.counter += 1
        self.loop_func(event)

    def show(self):
        self.plotter.timer_callback("start")
        self.plotter.interactive()

class RobotVisualizer(BaseVedoVisualizer):
    def __init__(self,num_subplots:int=1):
        super().__init__(num_subplots)
        self._add_robot()
    def _add_robot(self):
        self.robot = Box(pos=(0, 0, 0), c='b', alpha=0.1)
        for i in range(self.num_subplots):
            self.plotter.at(i).show(self.robot)
    def loop_func(self, event,**kwargs):
        for i in range(self.num_subplots):
            self.plotter.at(i).add(self.robot)
        self.plotter.render()



if __name__ == '__main__':
    vis = RobotVisualizer()
    vis.show()






