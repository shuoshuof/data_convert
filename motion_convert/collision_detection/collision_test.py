# -*- coding: utf-8 -*-
"""
@Time ： 2024/11/7 16:50
@Auth ： shuoshuof
@File ：collision_test.py
@Project ：data_convert
"""
import numpy as np
import torch
from vedo import *

import matplotlib.pyplot as plt

from poselib.poselib.core.rotation3d import *

from motion_convert.collision_detection.obb import OBBCollisionDetector, OBB
from motion_convert.collision_detection.obb_visualizer import OBBVisualizer,OBBBox, OBBVertices
from motion_convert.utils.torch_ext import to_torch,to_numpy

class TestOBBVisualizer(OBBVisualizer):


    def loop_fun(self,event):
        self.obb_collision_detector.update_obbs_transform(
            global_rotations=torch.concatenate([torch.tensor([0,0,0,1]).unsqueeze(0), quat_from_angle_axis(torch.tensor([1.57/2]),torch.Tensor([0,1,0]))] ),
            global_translations=torch.tensor([[0, 0, 0], [3-self.counter*0.1, 0, 0]]),
        )

        collision_mat = self.obb_collision_detector.check_collision()
        print(collision_mat)

        self.plotter.clear()
        obb_boxes = [OBBBox(obb) for obb in self.obb_collision_detector.obbs]
        obb_vertices = [ OBBVertices(obb,idx) for idx, obb in enumerate(self.obb_collision_detector.obbs)]

        vedo_vertices = obb_boxes[1].vertices[:8]
        cal_vertices = to_numpy(self.obb_collision_detector.obbs_vertices()[1])
        # assert np.allclose(cal_vertices,vedo_vertices,rtol=1e-3,atol=1e-3)

        self.plotter.add(*obb_boxes,*obb_vertices,self.world_frame)
        self.plotter.render()


if __name__ == '__main__':
    obb1 = OBB(
        initial_axes=torch.eye(3),
        extents=torch.tensor([1,1,1]),
        global_rotation=torch.tensor([0,0,0,1]),
        global_translation=torch.tensor([0,0,0]),
    )

    obb2 = OBB(
        initial_axes=torch.eye(3),
        extents=torch.tensor([1,1,1]),
        global_rotation=torch.tensor([0,0,0,1]),
        global_translation=torch.tensor([3,0,0]),
    )


    obb_detector = OBBCollisionDetector([obb1, obb2])

    obb_visualizer = TestOBBVisualizer(obb_detector)



