# -*- coding: utf-8 -*-
"""
@Time ： 2024/11/7 16:50
@Auth ： shuoshuof
@File ：collision_test.py
@Project ：data_convert
"""
import time

import numpy as np
import torch
from vedo import *

import matplotlib.pyplot as plt

from poselib.poselib.core.rotation3d import *

from motion_convert.collision_detection.obb import OBBCollisionDetector, OBB
from motion_convert.collision_detection.obb_visualizer import *
from motion_convert.utils.torch_ext import to_torch,to_numpy

class TestOBBVisualizer(OBBVisualizer):

    def loop_fun(self,event):
        self.obb_collision_detector.update_obbs_transform(
            # global_rotations=torch.concatenate([ quat_from_angle_axis(torch.tensor([self.counter*0.02]),torch.Tensor([0.5,1.1,3])),
            #                                     quat_from_angle_axis(torch.tensor([1.57/2]),torch.Tensor([0,1,0])),
            #                                    torch.tensor([0,0,0,1]).unsqueeze(0)]),
            global_rotations=torch.concatenate([ torch.tensor([0,0,0,1]).unsqueeze(0),
                                                quat_from_angle_axis(torch.tensor([1.7/2]),torch.Tensor([0.3,1,1.2])),
                                               torch.tensor([0,0,0,1]).unsqueeze(0)]),
            global_translations=torch.tensor([[0, 0, 0],
                                              [3-self.counter*0.01, 0 , 0],
                                              [0, 3-self.counter*0.02, 0]]),
        )

        collision_mat = self.obb_collision_detector.check_collision()
        # print(collision_mat)

        self.plotter.clear()
        obb_boxes = [OBBBox(obb) for obb in self.obb_collision_detector.obbs]
        obb_vertices = [ OBBVertices(obb,idx) for idx, obb in enumerate(self.obb_collision_detector.obbs)]
        obb_axes = [ OBBAxes(obb) for idx, obb in enumerate(self.obb_collision_detector.obbs)]
        obb_text = [OBBText(obb,idx) for idx, obb in enumerate(self.obb_collision_detector.obbs)]


        vedo_vertices = obb_boxes[1].vertices[:8]
        cal_vertices = to_numpy(self.obb_collision_detector.obbs_vertices()[1])
        assert np.allclose(cal_vertices,vedo_vertices,rtol=1e-3,atol=1e-3)

        collision0with1, collision0with2 = collision_mat[0, 1], collision_mat[0, 2]
        my_collision0with1, my_collision0with2 = my_check_collision(self.obb_collision_detector)

        print('-----------------------------')
        print("detector:",collision0with1,collision0with2)
        print("cal:",my_collision0with1,my_collision0with2)
        # assert collision0with1==my_collision0with1 and collision0with2==my_collision0with2
        # if not(collision0with1==my_collision0with1 and collision0with2==my_collision0with2):
        #     time.sleep(1)
        self.plotter.add(self.world_frame,*obb_boxes,*obb_vertices,*obb_axes,*obb_text)
        self.plotter.render()

def my_check_collision(obb_detector:OBBCollisionDetector):
    obbs_vertices = obb_detector.obbs_vertices()
    obb_extents = obb_detector.obbs_extents()
    obbs_global_translation = obb_detector.obbs_global_translation()
    obb0_collision_min_bound = obbs_global_translation[0]-obb_extents[0]
    obb0_collision_max_bound = obbs_global_translation[0]+obb_extents[0]

    # TODO: check the correctness of the collision detection
    collision0with1 = torch.any(torch.all(obbs_vertices[1]>=obb0_collision_min_bound,dim=1) & torch.all(obbs_vertices[1]<=obb0_collision_max_bound,dim=1))
    collision0with2 = torch.any(torch.all(obbs_vertices[2]>=obb0_collision_min_bound,dim=1) & torch.all(obbs_vertices[2]<=obb0_collision_max_bound,dim=1))

    return collision0with1,collision0with2

if __name__ == '__main__':
    obb0 = OBB(
        initial_axes=torch.eye(3),
        extents=torch.tensor([1,1,1]),
        global_rotation=torch.tensor([0,0,0,1]),
        global_translation=torch.tensor([0,0,0]),
    )

    obb1 = OBB(
        initial_axes=torch.eye(3),
        extents=torch.tensor([0.5,0.5,0.5]),
        global_rotation=torch.tensor([0,0,0,1]),
        global_translation=torch.tensor([3,0,0]),
    )

    obb2 = OBB(
        initial_axes=torch.eye(3),
        extents=torch.tensor([1,1,1]),
        global_rotation=torch.tensor([0,0,0,1]),
        global_translation=torch.tensor([0,3,0]),
    )

    obb_detector = OBBCollisionDetector([obb0,obb1,obb2])

    obb_visualizer = TestOBBVisualizer(obb_detector)



