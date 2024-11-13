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


class BaseVedoRobot:
    def __init__(self):
        self._original_robot_geometries = []
        self._robot_geometries = []
    @property
    def robot_geometries(self):
        return copy.deepcopy(self._robot_geometries)
    @property
    def robot_original_geometries(self):
        return copy.deepcopy(self._original_robot_geometries)
    @abstractmethod
    def _generate_robot_geometries(self, **kwargs):
        pass
    @abstractmethod
    def robot_transform(self,**kwargs):
        raise NotImplementedError
    def show(self):
        plotter = Plotter(axes=1, bg='white')
        plotter.show(*self.robot_geometries)

class VedoRobot(BaseVedoRobot):
    def __init__(self,robot_zero_pose):
        self.robot_zero_pose = robot_zero_pose
        super().__init__()
    @classmethod
    def from_urdf(cls,urdf_path):
        robot_zero_pose,link_mesh_file_names = parse_urdf(urdf_path)
        vedo_robot = cls(robot_zero_pose)
        link_mesh_paths = [os.path.join(os.path.dirname(urdf_path),link_file) for link_file in link_mesh_file_names]
        robot_geometries = vedo_robot._generate_robot_geometries(link_mesh_paths)
        vedo_robot._original_robot_geometries = robot_geometries
        vedo_robot._robot_geometries = copy.deepcopy(vedo_robot._original_robot_geometries)
        return vedo_robot

    def _generate_robot_geometries(self,link_mesh_paths):
        robot_geometries = []
        for link_idx,(link_global_translation,mesh_path) in enumerate(zip(self.robot_zero_pose.global_translation,link_mesh_paths)):
            mesh = Mesh(mesh_path,alpha=0.5)
            mesh.pos(link_global_translation)
            robot_geometries.append(mesh)
        return robot_geometries

    def robot_transform(self,global_translations,global_rotations):
        new_robot_geometries = []
        for link_mesh,link_global_translation,link_global_rotation in zip(self.robot_original_geometries, global_translations, global_rotations):
            link_rotation_angle, link_rotation_axis = quat_to_angle_axis(quat_normalize(link_global_rotation))
            link_rotation_angle = to_numpy(link_rotation_angle)
            link_rotation_axis = to_numpy(link_rotation_axis)

            link_mesh.rotate(angle=link_rotation_angle, axis=link_rotation_axis, rad=True)
            link_mesh.pos(link_global_translation.numpy())
            new_robot_geometries.append(link_mesh)
        self._robot_geometries = new_robot_geometries

class VedoOBBRobot(BaseVedoRobot):
    def __init__(self,obb_detector:OBBRobotCollisionDetector=None):
        self.obb_detector = obb_detector
        super().__init__()
    @classmethod
    def from_obb_detector(cls,obb_detector:OBBRobotCollisionDetector):
        vedo_robot = cls(obb_detector)
        robot_geometries = vedo_robot._generate_robot_geometries()
        vedo_robot._original_robot_geometries = robot_geometries
        vedo_robot._robot_geometries = copy.deepcopy(vedo_robot._original_robot_geometries)
        return vedo_robot
    def _generate_robot_geometries(self,use_wireframe=True):

        robot_geometries = []
        for i in range(self.obb_detector.num_obbs):
            obb_box = OBBBox(
                self.obb_detector.obb_robot_center_pos()[i],
                self.obb_detector.obb_robot_extents()[i],
                self.obb_detector.obb_robot_global_rotation()[i],
                use_wireframe=use_wireframe
            )
            robot_geometries.append(obb_box)
            robot_geometries.append(Sphere(pos=to_numpy(self.obb_detector.obb_robot_global_translation())[i],r=0.02,c='red'))
            robot_geometries.append(Sphere(pos=to_numpy(self.obb_detector.obb_robot_center_pos())[i],r=0.02, c='blue'))
        return robot_geometries

    def robot_transform(self,global_translations,global_rotations):
        self.obb_detector.update_obbs_transform(global_translations,global_rotations,from_link_transform=True)
        self._robot_geometries = self._generate_robot_geometries()

    def _update_robot_geometries(self):
        pass

    def show(self):
        plotter = Plotter(axes=1, bg='white')
        plotter.show(*self.robot_geometries)

class OBBBox(Box):
    def __init__(
            self,
            center_pos,
            extents,
            global_rotation,
            use_wireframe=True
    ):
        length, width, height = 2 * to_numpy(extents)
        super().__init__([0,0,0],length,width,height)
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

    obb_robot = OBBRobot(urdf_path='asset/hu/hu_v5.urdf')
    obb_detector = OBBRobotCollisionDetector(obb_robot=obb_robot)



    # hu_vedo_robot = VedoRobot.from_urdf('asset/hu/hu_v5.urdf')
    # hu_vedo_robot.show()



