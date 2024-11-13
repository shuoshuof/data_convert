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
from abc import ABC,abstractmethod

import networkx as nx
import numpy as np
from scipy.spatial.transform import Rotation as sRot

import torch

from vedo import *

from poselib.poselib.skeleton.skeleton3d import SkeletonMotion,SkeletonTree,SkeletonState
from poselib.poselib.core.rotation3d import *

from motion_convert.collision_detection.obb_robot import OBBRobotCollisionDetector,OBBRobot

from motion_convert.utils.torch_ext import to_numpy
from motion_convert.utils.parse_urdf import parse_urdf


#TODO: construct a base class for robot
class BaseVedoRobot:
    def __init__(self):
        self._original_robot_geometry = []
        self._robot_geometry = []
    def robot_geometry(self):
        return copy.deepcopy(self._robot_geometry)
    @abstractmethod
    def _generate_robot_geometry(self,**kwargs):
        pass
    @abstractmethod
    def robot_transform(self,**kwargs):
        raise NotImplementedError


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


class VedoOBBRobot(BaseVedoRobot):
    def __init__(self,obb_detector:OBBRobotCollisionDetector=None):
        self.obb_detector = obb_detector
        super().__init__()
    @classmethod
    def from_obb_detector(cls,obb_detector:OBBRobotCollisionDetector):
        vedo_robot = cls(obb_detector)
        robot_geometries = vedo_robot._generate_robot_geometries()
        vedo_robot._original_robot_geometry = robot_geometries
        vedo_robot._robot_geometry = copy.deepcopy(vedo_robot._original_robot_geometry)
        return vedo_robot
    def _generate_robot_geometries(self):

        robot_geometries = []
        for i in range(self.obb_detector.num_obbs):
            obb_box = OBBBox(
                self.obb_detector.obb_robot_center_pos()[i],
                self.obb_detector.obb_robot_extents()[i],
                self.obb_detector.obb_robot_global_rotation()[i],
            )
            robot_geometries.append(obb_box)

        return robot_geometries

    def robot_transform(self,global_translations,global_rotations):
        self.obb_detector.update_obbs_transform(global_translations,global_rotations)
        self._robot_geometry = self._generate_robot_geometries()

    def _update_robot_geometries(self):
        pass

    def show(self):
        plotter = Plotter(axes=1, bg='white')
        plotter.show(*self._robot_geometry)

class OBBBox(Box):
    def __init__(
            self,
            center_pos,
            extents,
            global_rotation,
            use_wireframe=True
    ):
        length, width, height = 2 * to_numpy(extents)
        super().__init__(to_numpy(center_pos),length,width,height)
        self.obb_transform(center_pos,global_rotation)

        if use_wireframe:
            self.wireframe()
    def obb_transform(self,center_pos, global_rotation):
        obb_rotation_angle,obb_rotation_axis = quat_to_angle_axis(global_rotation)
        obb_rotation_axis = to_numpy(obb_rotation_axis)
        obb_rotation_angle = to_numpy(obb_rotation_angle)
        self.rotate(angle=obb_rotation_angle, axis=obb_rotation_axis,rad=True)
        obb_center = to_numpy(center_pos)
        self.pos(*obb_center)

if __name__ == '__main__':
    # vedo_robot = VedoRobot(
    #     'asset/hu/hu_v5.urdf',
    # )

    obb_robot = OBBRobot(urdf_path='asset/hu/hu_v5.urdf')
    obb_detector = OBBRobotCollisionDetector(obb_robot=obb_robot)

    VedoOBBRobot.from_obb_detector(obb_detector).show()





