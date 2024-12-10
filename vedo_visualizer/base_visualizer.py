# -*- coding: utf-8 -*-
"""
@Time ： 2024/11/9 18:01
@Auth ： shuoshuof
@File ：base_visualizer.py
@Project ：data_convert
"""
from typing import List
from abc import ABC,abstractmethod
import math
import copy
import time

from vedo import *

from vedo_visualizer.vedo_robot import BaseVedoRobot

settings.default_backend = "vtk"
settings.immediate_rendering = False

class BaseVedoVisualizer(ABC):
    def __init__(self,num_subplots,**kwargs):
        self.num_subplots = num_subplots
        self._init_plotter()

        self.pause_button = self.plotter.at(num_subplots-1).add_button(
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
    def loop(self,event):
        self.counter += 1

    def show(self):
        self.plotter.timer_callback("start")
        self.plotter.interactive()

class RobotVisualizer(BaseVedoVisualizer):
    def __init__(self, num_subplots:int, robots:List[BaseVedoRobot], data:List):
        super().__init__(num_subplots)
        self.robots_list = [copy.deepcopy(robots) for _ in range(num_subplots)]
        self.num_vedo_robot = len(robots)

        self.data = data
        # assert len(self.data) == self.num_subplots
        self._add_robot()

    def _add_robot(self):
        for i in range(self.num_subplots):
            for j in range(self.num_vedo_robot):
                self.plotter.at(i).add(*self.robots_list[i][j].robot_geometries)
        self.plotter.show()
    def update_plt(self):
        for i in range(self.num_subplots):
            self.plotter.at(i).clear()
            for j in range(self.num_vedo_robot):
                self.plotter.at(i).add(*self.robots_list[i][j].robot_geometries)
        self.plotter.render()
    @abstractmethod
    def update_robots(self):
        raise NotImplementedError

    def loop(self, event):
        start = time.time()
        self.counter +=1
        self.update_robots()
        self.update_plt()
        # print(f'fps: {round((1/(time.time()-start)),2)}')

if __name__ == '__main__':
    pass






