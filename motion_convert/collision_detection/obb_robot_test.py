# -*- coding: utf-8 -*-
"""
@Time ： 2024/11/16 13:56
@Auth ： shuoshuof
@File ：obb_robot_test.py
@Project ：data_convert
"""
import pickle
import unittest
from collections import OrderedDict
import torch

from motion_convert.collision_detection.obb_robot import OBBRobotCollisionDetector,OBBRobot

from poselib.poselib.core.rotation3d import *



def init_random_test_config():
    motion_length = 600
    motion_global_translations = torch.tensor(
        [[[0,0,0]]*motion_length,
         [[3-i*0.01,0,0] for i in range(motion_length)],
         [[0,3-i*0.2,0] for i in range(motion_length)]]
    ).transpose(0,1)
    assert motion_global_translations.shape == (motion_length,3,3)
    obbs_config = OrderedDict(
        obb0=OrderedDict(
            initial_axes=torch.eye(3),
            extents=torch.tensor([1,1,1]),
            initial_offsets=torch.tensor([0,0,0]),
            global_rotation=torch.tensor([0,0,0,1]),
            global_translation=torch.tensor([0,0,0]),
        ),
        obb1=OrderedDict(
            initial_axes=torch.eye(3),
            extents=torch.clamp(torch.rand(3),min=0.2,max=1/1.74),
            initial_offsets=torch.tensor([0., 0., 0.]),
            global_rotation=quat_from_angle_axis(torch.rand(1)*3.14,torch.rand(3)).squeeze(0),
            global_translation=torch.tensor([5,0,0]),
        ),
        obb2=OrderedDict(
            initial_axes=torch.eye(3),
            extents=torch.clamp(torch.rand(3),min=0.2,max=1/1.74),
            initial_offsets=torch.tensor([0., 0., 0.]),
            global_rotation=quat_from_angle_axis(torch.rand(1)*3.14,torch.rand(3)).squeeze(0),
            global_translation=torch.tensor([0,5,0]),
        ),
    )
    return obbs_config, motion_global_translations

def my_check_collision(obb_detector:OBBRobotCollisionDetector):
    obbs_vertices = obb_detector.obb_robot_vertices()
    obb_extents = obb_detector.obb_robot_extents()
    obbs_global_translation = obb_detector.obb_robot_global_translation()
    obb0_collision_min_bound = obbs_global_translation[0]-obb_extents[0]
    obb0_collision_max_bound = obbs_global_translation[0]+obb_extents[0]

    collision0with1 = torch.any(torch.all(obbs_vertices[1]>=obb0_collision_min_bound,dim=1) & torch.all(obbs_vertices[1]<=obb0_collision_max_bound,dim=1))
    collision0with2 = torch.any(torch.all(obbs_vertices[2]>=obb0_collision_min_bound,dim=1) & torch.all(obbs_vertices[2]<=obb0_collision_max_bound,dim=1))

    return collision0with1,collision0with2



def generate_test(i, obbs_config, motion_global_translations):
    def test(self):
        obb_robot = OBBRobot.from_dict(obb_dict=obbs_config)
        obb_collision_detector = OBBRobotCollisionDetector(
            obb_robot=obb_robot,
            use_zero_pose_mask=False,
        )
        for global_translation in motion_global_translations:
            obb_collision_detector.update_obbs_transform(
                link_global_translations=global_translation,
                link_global_rotations =None,
                from_link_transform=False
            )

            collision_mat = obb_collision_detector.check_collision()
            collision0with1, collision0with2 = collision_mat[0, 1], collision_mat[0, 2]
            my_collision0with1, my_collision0with2 = my_check_collision(obb_collision_detector)

            # 比较碰撞结果
            try:
                self.assertEqual((collision0with1, collision0with2), (my_collision0with1, my_collision0with2))
            except AssertionError as e:
                with open(f'motion_convert/collision_detection/test_log/{i}.pkl', 'wb') as f:
                    pickle.dump({"obb_robot":obb_robot,"motion_global_translations":motion_global_translations},f)
                raise e

    # 为该测试方法命名
    test.__name__ = f'test_obbs_{i}'
    return test

class TestOBB(unittest.TestCase):
    pass

def init_tests():
    num_tests = 500
    for i in range(num_tests):
        obbs_config, motion_global_translations = init_random_test_config()
        test_method = generate_test(i, obbs_config, motion_global_translations)
        setattr(TestOBB, f'test_obbs_{i}', test_method)

init_tests()

if __name__ == '__main__':

    unittest.main()