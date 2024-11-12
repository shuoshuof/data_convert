# -*- coding: utf-8 -*-
"""
@Time ： 2024/11/10 17:30
@Auth ： shuoshuof
@File ：vedo_robot.py
@Project ：data_convert
"""
import os
import copy
from collections import OrderedDict
import networkx as nx
import numpy as np

import torch

from vedo import *

from poselib.poselib.skeleton.skeleton3d import SkeletonMotion,SkeletonTree,SkeletonState
from poselib.poselib.core.rotation3d import *

from motion_convert.utils.torch_ext import to_numpy
from motion_convert.utils.parse_urdf import parse_urdf


#TODO: construct a base class for robot

class VedoRobot:
    def __init__(self,urdf_path):
        self.robot_zero_pose,link_mesh_file_names = parse_urdf(urdf_path)
        self._original_robot_mesh = self._generate_robot(urdf_path, self.robot_zero_pose, link_mesh_file_names)
        self._robot_mesh = self.original_robot_mesh
    @property
    def original_robot_mesh(self):
        return copy.deepcopy(self._original_robot_mesh)
    @property
    def robot_mesh(self):
        return copy.deepcopy(self._robot_mesh)

    def _generate_robot(self,urdf_path,robot_zero_pose:SkeletonState,link_mesh_file_names):
        robot_sk_tree = robot_zero_pose.skeleton_tree
        robot_meshes = []
        markers = []
        for link_idx,(joint_global_translation,link_file) in enumerate(zip(robot_zero_pose.global_translation,link_mesh_file_names)):
            file_path = os.path.join(os.path.dirname(urdf_path),link_file)
            mesh = Mesh(file_path,alpha=0.5)
            mesh.pos(joint_global_translation)
            robot_meshes.append(mesh)
            markers.append(Sphere(pos=joint_global_translation, r=0.02, c='red'))
        plotter = Plotter(axes=1,bg='white')
        plotter.show(*robot_meshes,*markers)
        return robot_meshes

    def robot_transform(self,link_global_translations:torch.Tensor, link_global_rotations:torch.Tensor):
        new_robot_mesh = []
        markers = []
        for link_mesh,link_global_translation,link_global_rotation in zip(self.original_robot_mesh, link_global_translations, link_global_rotations):
            link_rotation_angle, link_rotation_axis = quat_to_angle_axis(quat_normalize(link_global_rotation))
            link_rotation_angle = to_numpy(link_rotation_angle)
            link_rotation_axis = to_numpy(link_rotation_axis)

            link_mesh.rotate(angle=link_rotation_angle, axis=link_rotation_axis, rad=True)
            link_mesh.pos(link_global_translation.numpy())
            new_robot_mesh.append(link_mesh)
            markers.append(Sphere(pos=link_global_translation, r=0.02, c='red'))
        self._robot_mesh = new_robot_mesh
        self.markers = markers

class HuVedoRobot(VedoRobot):
    def __init__(self,urdf_path='asset/hu/hu_v5.urdf'):
        super().__init__(urdf_path)

if __name__ == '__main__':
    vedo_robot = VedoRobot(
        'asset/hu/hu_v5.urdf',
    )





