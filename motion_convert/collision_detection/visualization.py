import matplotlib.pyplot as plt
import torch
from vedo import *
from typing import List,Union
from motion_convert.collision_detection.obb import OBB,OBBCollisionDetector
from poselib.poselib.core.rotation3d import *
import time


class ObbVisualizer:
    def __init__(self,obb_collision_detector:Union[OBBCollisionDetector]):

        self.obb_collision_detector = obb_collision_detector

        self.plotter = Plotter()

        self.world_frame = Box(pos=[0, 0, 0], length=2, width=2, height=2).wireframe()
        self.plotter.add(self.world_frame)

        self.button = self.plotter.add_button(
            self._button_func,
            states=["\u23F5 Play  ", "\u23F8 Pause"],
            font="Kanopus",
            size=32,
        )
        self.counter = 0
        self.timer_id =None

        self.plotter.add_callback('timer',self.loop,enable_picking=False)
        self.plotter.show()


    def _button_func(self, obj, ename):
        if self.timer_id is not None:
            self.plotter.timer_callback("destroy",self.timer_id)
        if "Play" in self.button.status():
            self.timer_id = self.plotter.timer_callback("create",dt=10)

        self.button.switch()

    def loop(self,event):
        self.counter += 1

        self.obb_collision_detector.update_obbs_transform(
            global_rotations=torch.tensor([[0,0,0,1],[0,0,0,1]]),
            global_translations=torch.tensor([[self.counter/1000,0,0],[-self.counter/1000,0,0]]),
        )
        self.plotter.clear()
        obb_boxes = [ObbBox(obb) for obb in self.obb_collision_detector.obbs()]

        self.plotter.add(*obb_boxes,self.world_frame)
        self.plotter.render()





class ObbBox(Box):
    def __init__(self, obb:OBB,trail=10):

        pos = torch_to_numpy(obb.global_translation)
        length, width, height = 2*torch_to_numpy(obb.extents)
        super().__init__(pos,length,width,height)
        self.update(obb)

        if trail > 0:
            self.add_trail(trail)

    def update(self,obb:OBB):
        obb_rotation_angle,obb_rotation_axis = quat_to_angle_axis(obb.global_rotation)
        obb_rotation_axis = torch_to_numpy(obb_rotation_axis)
        obb_rotation_angle = torch_to_numpy(obb_rotation_angle)
        self.rotate(angle=obb_rotation_angle, axis=obb_rotation_axis)
        self.pos(*obb.global_translation)




def torch_to_numpy(t):
    if isinstance(t, torch.Tensor):
        return t.cpu().detach().numpy()
    return t

if __name__ == "__main__":

    obb1 = OBB(
        initial_axes=torch.eye(3),
        extents=torch.tensor([1e-2,1e-2,1e-2]),
        global_rotation=torch.tensor([0,0,0,1]),
        global_translation=torch.tensor((0,0,0)),
    )

    obb2 = OBB(
        initial_axes=torch.eye(3),
        extents=torch.tensor([1e-2,1e-2,1e-2]),
        global_rotation=torch.tensor([0,0,0,1]),
        global_translation=torch.tensor((0,0,0)),
    )

    obb_detector = OBBCollisionDetector([obb1,obb2]*20)

    # obbv = ObbVisualizer(obb_detector.get_obbs())

    # for i in range(1000):
    #
    #     obb_detector.update_obbs_transform(
    #         global_rotations=torch.tensor([[0,0,0,1],[0,0,0,1]]),
    #         global_translations=torch.tensor([[i/1000,0,0],[-i/1000,0,0]]),
    #     )
    #     obbv.update_obbs_boxes(obb_detector.get_obbs())
    #     obbv.render()

    obb_visualizer = ObbVisualizer(obb_detector)
