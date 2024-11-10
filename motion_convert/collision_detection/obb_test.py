# -*- coding: utf-8 -*-
"""
@Time ： 2024/11/8 15:00
@Auth ： shuoshuof
@File ：obb_test.py
@Project ：data_convert
"""

import numpy as np
import torch
from collections import OrderedDict
import unittest

from vedo import *

from poselib.poselib.core.rotation3d import *

from motion_convert.collision_detection.obb import OBBCollisionDetector, OBB
from motion_convert.collision_detection.obb_visualizer import *
from motion_convert.utils.torch_ext import to_torch,to_numpy



# TODO : evaluate the correctness of the collision detection
# check the collision through calculate whether the vertices in the box0
# set the box0 static, box1, box2 move and rotate
# set random rotation to box1 and box2
def init_random_test_config():
    motion_length = 600
    motion_global_translations = torch.tensor(
        [[[0,0,0]]*motion_length,
         [[3-i*0.01,0,0] for i in range(motion_length)],
         [[0,3-i*0.2,0] for i in range(motion_length)]]
    ).transpose(0,1)
    assert motion_global_translations.shape == (motion_length,3,3)
    obbs_config = OrderedDict(
        obb0=OBB(
            initial_axes=torch.eye(3),
            extents=torch.tensor([1,1,1]),
            global_rotation=torch.tensor([0,0,0,1]),
            global_translation=torch.tensor([0,0,0]),
        ),
        obb1=OBB(
            initial_axes=torch.eye(3),
            extents=torch.clamp(torch.rand(3),min=0.2,max=1/1.74),
            global_rotation=quat_from_angle_axis(torch.rand(1)*3.14,torch.rand(3)).squeeze(0),
            global_translation=torch.tensor([3,0,0]),
        ),
        obb2=OBB(
            initial_axes=torch.eye(3),
            extents=torch.clamp(torch.rand(3),min=0.2,max=1/1.74),
            global_rotation=quat_from_angle_axis(torch.rand(1)*3.14,torch.rand(3)).squeeze(0),
            global_translation=torch.tensor([0,3,0]),
        ),
    )
    return obbs_config, motion_global_translations

def my_check_collision(obb_detector:OBBCollisionDetector):
    obbs_vertices = obb_detector.obbs_vertices()
    obb_extents = obb_detector.obbs_extents()
    obbs_global_translation = obb_detector.obbs_global_translation()
    obb0_collision_min_bound = obbs_global_translation[0]-obb_extents[0]
    obb0_collision_max_bound = obbs_global_translation[0]+obb_extents[0]

    collision0with1 = torch.any(torch.all(obbs_vertices[1]>=obb0_collision_min_bound,dim=1) & torch.all(obbs_vertices[1]<=obb0_collision_max_bound,dim=1))
    collision0with2 = torch.any(torch.all(obbs_vertices[2]>=obb0_collision_min_bound,dim=1) & torch.all(obbs_vertices[2]<=obb0_collision_max_bound,dim=1))

    return collision0with1,collision0with2



def generate_test(i, obbs_config, motion_global_translations):
    def test(self):
        obb_collision_detector = OBBCollisionDetector(list(obbs_config.values()))
        for global_translations in motion_global_translations:
            obb_collision_detector.update_obbs_transform(
                global_rotations=None,
                global_translations=global_translations,
            )

            obb_collision_detector.obbs_vertices()
            collision_mat = obb_collision_detector.check_collision()
            collision0with1, collision0with2 = collision_mat[0, 1], collision_mat[0, 2]
            my_collision0with1, my_collision0with2 = my_check_collision(obb_collision_detector)

            # 比较碰撞结果
            self.assertEqual((collision0with1, collision0with2), (my_collision0with1, my_collision0with2))

    # 为该测试方法命名
    test.__name__ = f'test_obbs_{i}'
    return test

class TestOBB(unittest.TestCase):
    pass

def init_tests():
    num_tests = 100
    for i in range(num_tests):
        obbs_config, motion_global_translations = init_random_test_config()
        test_method = generate_test(i, obbs_config, motion_global_translations)
        setattr(TestOBB, f'test_obbs_{i}', test_method)

init_tests()

if __name__ == '__main__':

    unittest.main()