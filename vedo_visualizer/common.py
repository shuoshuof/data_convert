# -*- coding: utf-8 -*-
"""
@Time ： 2024/11/11 14:54
@Auth ： shuoshuof
@File ：common.py
@Project ：data_convert
"""
import os.path
import pickle
from typing import List, Union
import torch

from poselib.poselib.skeleton.skeleton3d import SkeletonMotion,SkeletonState

from motion_convert.collision_detection.obb_robot import OBBRobot,OBBRobotCollisionDetector

from vedo_visualizer.base_visualizer import RobotVisualizer
from vedo_visualizer.vedo_robot import VedoRobot,VedoOBBRobot,BaseVedoRobot,VedoRobotWithCollision

from vedo import *

settings.default_backend = "vtk"
settings.immediate_rendering = False

class SkMotionVisualizer(RobotVisualizer):
    def __init__(self, num_subplots:int, robots:List[BaseVedoRobot],data:List[Union[SkeletonMotion,SkeletonState]]) -> None:
        super().__init__(num_subplots,robots, data)

    def update_robots(self):
        # print(self.counter)
        for i in range(self.num_subplots):
            if self.counter>=len(self.data[i]):
                continue
            for j in range(self.num_vedo_robot):
                link_global_rotations = self.data[i].global_rotation[self.counter]
                link_global_translations = self.data[i].global_translation[self.counter]
                self.robots_list[i][j].robot_transform(link_global_translations,link_global_rotations)

def hu_v2_to_hu_v5(motion:SkeletonMotion):
    v2tov5_indices = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                      29, 30, 31, 32]
    with open('asset/hu_pose/hu_v5_zero_pose.pkl', 'rb') as f:
        hu_v5_zero_pose:SkeletonState = pickle.load(f)
    new_local_rotations = motion.local_rotation.clone()[:,v2tov5_indices]
    new_root_translations = motion.root_translation.clone()
    new_root_translations -= new_root_translations[0,...].clone()
    new_sk_state = SkeletonState.from_rotation_and_root_translation(
        hu_v5_zero_pose.skeleton_tree,
        new_local_rotations,
        new_root_translations,
        is_local=True
    )
    return SkeletonMotion.from_skeleton_state(new_sk_state,fps=motion.fps)


def vis_sk_motion1(motions:List[Union[SkeletonMotion,SkeletonState]], divide=False, vis_links_indices=None):

    vedo_hu_robot = VedoRobot.from_urdf(urdf_path='asset/hu/hu_v5.urdf')

    obb_robot = OBBRobot.from_urdf(urdf_path='asset/hu/hu_v5.urdf',divide=divide)

    link_collision_mask = torch.ones((obb_robot.num_links,obb_robot.num_links),dtype=torch.bool)
    link_collision_mask[11,13] = 0
    link_collision_mask[13,11] = 0
    link_collision_mask[11,22] = 0
    link_collision_mask[22,11] = 0
    link_collision_mask[1,3] = 0
    link_collision_mask[3,1] = 0
    link_collision_mask[6,8] = 0
    link_collision_mask[8,6] = 0


    obb_detector = OBBRobotCollisionDetector(obb_robot=obb_robot, link_collision_mask=link_collision_mask)
    vedo_obb_robot = VedoOBBRobot.from_obb_detector(obb_detector,vis_links_indices=vis_links_indices)

    robots = [vedo_obb_robot,vedo_hu_robot]

    new_motions = [hu_v2_to_hu_v5(motion) for motion in motions]

    vis = SkMotionVisualizer(len(motions),robots,new_motions)
    vis.show()

def vis_robot_collision(motions:List[Union[SkeletonMotion,SkeletonState]],divide=True):

    obb_robot = OBBRobot.from_urdf(urdf_path='asset/hu/hu_v5.urdf', divide=divide)
    link_collision_mask = torch.ones((obb_robot.num_links,obb_robot.num_links),dtype=torch.bool)
    link_collision_mask[11,13] = 0
    link_collision_mask[13,11] = 0
    link_collision_mask[11,22] = 0
    link_collision_mask[22,11] = 0
    link_collision_mask[1,3] = 0
    link_collision_mask[3,1] = 0
    link_collision_mask[6,8] = 0
    link_collision_mask[8,6] = 0

    obb_detector = OBBRobotCollisionDetector(obb_robot=obb_robot, link_collision_mask=link_collision_mask)

    vedo_collision_robot = VedoRobotWithCollision.from_urdf_and_detector(urdf_path='asset/hu/hu_v5.urdf',obb_detector=obb_detector)

    robots = [vedo_collision_robot]
    from motion_convert.utils.motion_process import  hu_zero_motion
    # new_motions = [hu_zero_motion()]
    new_motions = [hu_v2_to_hu_v5(motion) for motion in motions]
    vis = SkMotionVisualizer(len(motions),robots,new_motions)
    vis.show()

def vis_sk_motion(motions:List[Union[SkeletonMotion,SkeletonState]]):
    vedo_hu_robot = VedoRobot.from_urdf(urdf_path='asset/hu/hu_v5.urdf')

    robots = [vedo_hu_robot]

    new_motions = [hu_v2_to_hu_v5(motion) for motion in motions]

    vis = SkMotionVisualizer(len(motions),robots,new_motions)
    vis.show()

def vis_motion_data_dict(path):
    import joblib
    from motion_convert.utils.torch_ext import to_torch
    key = os.path.basename(path).split('.')[0]
    with open(path,'rb') as f:
        motion_dict = joblib.load(f)
    motion_data = motion_dict[key]
    # v2tov5_indices = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
    #                   29, 30, 31, 32]
    global_rotations = to_torch(motion_data['pose_quat_global'])
    root_translations = to_torch(motion_data['root_trans_offset'])

    with open('asset/zero_pose/hu_zero_pose.pkl', 'rb') as f:
        zero_pose:SkeletonState = pickle.load(f)

    state = SkeletonState.from_rotation_and_root_translation(
        zero_pose.skeleton_tree,
        global_rotations,
        root_translations,
        False
    )

    motion = SkeletonMotion.from_skeleton_state(state,fps=motion_data['fps'])

    vedo_hu_robot = VedoRobot.from_urdf(urdf_path='asset/hu/hu_v5.urdf')

    robots = [vedo_hu_robot]

    new_motions = [hu_v2_to_hu_v5(motion)]

    vis = SkMotionVisualizer(len(new_motions),robots,new_motions)
    vis.show()


if __name__ == '__main__':
    import copy
    # with open('motion_data/11_7_walk/hu_motion/walk_small_step1_11_07_22_mirror_motion.pkl','rb') as f:
    #     motion = pickle.load(f)
    #
    # with open('motion_data/11_17/hu_motion/stand_up_2_11_17_18_mirror_motion.pkl','rb') as f:
    #     motion = pickle.load(f)
    #
    # vis_sk_motion([motion])
    # vis_robot_collision([copy.deepcopy(motion)],divide=False)
    # vis_sk_motion([copy.deepcopy(motion)],divide=True,vis_links_indices=[16,18])
    # vis_sk_motion([copy.deepcopy(motion)],divide=True)
    # vis_robot_collision([copy.deepcopy(motion)],divide=True)

    vis_motion_data_dict('motion_data/12_4_small_walk/hu/walk4_12_04_19.pkl')

