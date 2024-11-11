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

from urdfpy import URDF

class VedoRobot:
    def __init__(self,urdf_path):
        self.robot_zero_pose,link_mesh_file_names = self._parse_urdf(urdf_path)
        self._robot_mesh = self._generate_robot(urdf_path, self.robot_zero_pose, link_mesh_file_names)
    @property
    def robot_mesh(self):
        return copy.deepcopy(self._robot_mesh)

    def _parse_urdf(self,urdf_path):
        urdf_robot:URDF = URDF.load(urdf_path)
        # urdf_robot.show()
        fk_link = urdf_robot.link_fk()

        urdf_graph = copy.deepcopy(urdf_robot._G)

        link_parents = np.argmax(nx.adjacency_matrix(urdf_graph).todense(), axis=1).A1
        link_parents[0] = -1

        link_names = []
        link_mesh_file_names = []
        link_translations = []

        for link,transform in fk_link.items():
            link_name = link.name
            link_translation = transform[:3,3]

            link_mesh_file_names.append(link.visuals[0].geometry.mesh.filename)
            link_names.append(link_name)
            link_translations.append(link_translation)
        link_translations = np.array(link_translations)
        link_local_translations = np.zeros_like(link_translations)
        link_local_translations[1:] = link_translations[1:]-link_translations[link_parents[1:]]

        print(link_names)
        print(link_parents)

        robot_sk_tree = SkeletonTree.from_dict(
            OrderedDict({'node_names': link_names,
                         'parent_indices': {'arr': np.array(link_parents), 'context': {'dtype': 'int64'}},
                         'local_translation': {'arr': np.array(link_local_translations), 'context': {'dtype': 'float32'}}})
        )

        # from poselib.poselib.visualization.common import plot_skeleton_H
        robot_zero_pose = SkeletonState.zero_pose(robot_sk_tree)
        # plot_skeleton_H([robot_zero_pose])
        return robot_zero_pose, link_mesh_file_names
    def _generate_robot(self,urdf_path,robot_zero_pose:SkeletonState,link_mesh_file_names):
        robot_sk_tree = robot_zero_pose.skeleton_tree
        robot_meshes = []
        for link_idx,(joint_global_translation,link_file) in enumerate(zip(robot_zero_pose.global_translation,link_mesh_file_names)):
            file_path = os.path.join(os.path.dirname(urdf_path),link_file)
            mesh = Mesh(file_path)
            mesh.pos(joint_global_translation)
            robot_meshes.append(mesh)
        plotter = Plotter(axes=1,bg='white')
        plotter.show(*robot_meshes)
        return robot_meshes

    def robot_transform(self,link_global_translations:torch.Tensor, link_global_rotations:torch.Tensor):
        new_robot_mesh = []
        for link_mesh,link_global_translation,link_global_rotation in zip(self.robot_mesh,link_global_translations,link_global_rotations):
            link_mesh.pos(link_global_translation.numpy())
            link_rotation_angle, link_rotation_axis = quat_to_angle_axis(link_global_rotation)
            link_rotation_angle = to_numpy(link_rotation_angle)
            link_rotation_axis = to_numpy(link_rotation_axis)
            link_mesh.rotate(angle=link_rotation_angle, axis=link_rotation_axis, rad=True)
            new_robot_mesh.append(link_mesh)

        return new_robot_mesh

class HuVedoRobot(VedoRobot):
    def __init__(self,urdf_path='asset/hu/hu_v5.urdf'):
        super().__init__(urdf_path)

if __name__ == '__main__':
    vedo_robot = VedoRobot(
        'asset/hu/hu_v5.urdf',
    )





