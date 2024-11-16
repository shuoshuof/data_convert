# -*- coding: utf-8 -*-
"""
@Time ： 2024/11/16 18:07
@Auth ： shuoshuof
@File ：vedo_joint_test.py
@Project ：data_convert
"""

import pickle
from typing import List, Union
import torch

from motion_convert.collision_detection.obb_robot import OBBRobot,OBBRobotCollisionDetector

from vedo_visualizer.base_visualizer import RobotVisualizer
from vedo_visualizer.base_visualizer import RobotVisualizer
from vedo_visualizer.vedo_robot import VedoRobot,VedoOBBRobot,BaseVedoRobot,VedoRobotWithCollision

from poselib.poselib.skeleton.skeleton3d import SkeletonState
from poselib.poselib.core.rotation3d import *
import copy
class JointVisualizer(RobotVisualizer):
    def __init__(self, num_subplots:int, robots:List[BaseVedoRobot],data,zero_pose:SkeletonState) -> None:
        self.num_subplots = num_subplots
        self._init_plotter()
        self.zero_pose = zero_pose
        self.link_local_rotations = zero_pose.local_rotation.clone()
        # self.link_global_translations = zero_pose.global_translation.clone()
        self._generate_slider_funcs()
        self._add_sliders()
        # self.plotter.add_slider(self.slider1,xmin=-3.14, xmax=3.14,value=0,pos='bottom-left',title='1')

        self.pause_button = self.plotter.at(num_subplots-1).add_button(
            self._stop_button_func,
            states=["\u23F5 Play", "\u23F8 Pause"],
            font="Kanopus",
            size=32,
        )

        self.counter = 0
        self.timer_id = None

        self.plotter.add_callback('timer', self.loop, enable_picking=False)
        self.robots_list = [copy.deepcopy(robots) for _ in range(num_subplots)]
        self.num_vedo_robot = len(robots)

        self.data = data
        # assert len(self.data) == self.num_subplots
        self._add_robot()

    def _add_sliders(self):
        for i in range(self.zero_pose.skeleton_tree.num_joints-1):
            x1, x2 = 0.1,0.2
            y1, y2 = (i+1)*0.05,(i+1)*0.05
            self.plotter.add_slider(
                getattr(self,f'slider{i+1}'),
                xmin=-3.14,
                xmax=3.14,
                value=0,
                title=str(i+1),
                pos=[(x1,y1),(x2,y2)]
            )

    def _generate_slider_funcs(self):
        Hu_DOF_AXIS = [
            2, 0, 1, 1, 1,
            2, 0, 1, 1, 1,
            2,
            1, 0, 2, 1, 0, 1, 2, 1, 1,
            1, 0, 2, 1, 0, 1, 2, 1, 1,
            2, ]
        hu_dof_axis = torch.eye(3)[Hu_DOF_AXIS]
        for i in range(len(hu_dof_axis)):
            def slider(widget, event,joint_idx=i):
                angle = widget.value
                self.link_local_rotations[joint_idx+1] = quat_from_angle_axis(torch.tensor([angle]), hu_dof_axis[joint_idx])
            setattr(self,f'slider{i+1}',slider)

    def update_robots(self):
        # print(self.counter)
        for i in range(self.num_subplots):
            new_state = SkeletonState.from_rotation_and_root_translation(
                self.zero_pose.skeleton_tree,
                self.link_local_rotations.clone(),
                torch.tensor([0., 0., 0.]),
                is_local=True
            )
            link_global_rotations = new_state.global_rotation
            link_global_translations = new_state.global_translation
            for j in range(self.num_vedo_robot):
                self.robots_list[i][j].robot_transform(link_global_translations,link_global_rotations)


if __name__ == '__main__':
    vedo_hu_robot = VedoRobot.from_urdf(urdf_path='asset/hu/hu_v5.urdf')

    obb_robot = OBBRobot.from_urdf(urdf_path='asset/hu/hu_v5.urdf',divide=False)

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
    vedo_obb_robot = VedoOBBRobot.from_obb_detector(obb_detector,vis_links_indices=None)

    robots = [vedo_obb_robot,vedo_hu_robot]

    with open('asset/hu_pose/hu_v5_zero_pose.pkl', 'rb') as f:
        hu_v5_zero_pose:SkeletonState = pickle.load(f)

    vis = JointVisualizer(num_subplots=1,robots=robots,data=[],zero_pose=hu_v5_zero_pose)
    vis.show()


