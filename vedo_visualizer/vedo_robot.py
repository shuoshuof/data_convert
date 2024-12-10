# -*- coding: utf-8 -*-
"""
@Time ： 2024/11/10 17:30
@Auth ： shuoshuof
@File ：vedo_robot.py
@Project ：data_convert
"""
import os
import copy
import time
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
    def from_urdf(cls,urdf_path,**kwargs):
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
            mesh = Mesh(mesh_path,alpha=0.3)
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
    def __init__(self,obb_detector:OBBRobotCollisionDetector,vis_links_indices=None):
        self.obb_detector = obb_detector
        self.vis_links_indices = vis_links_indices
        super().__init__()
    @classmethod
    def from_obb_detector(cls,obb_detector:OBBRobotCollisionDetector,vis_links_indices=None):
        vedo_robot = cls(obb_detector,vis_links_indices)
        robot_geometries = vedo_robot._generate_robot_geometries()
        vedo_robot._original_robot_geometries = robot_geometries
        vedo_robot._robot_geometries = copy.deepcopy(vedo_robot._original_robot_geometries)
        return vedo_robot
    def _generate_robot_geometries(self,collision_mat=None,use_wireframe=True):
        robot_geometries = []
        for i in range(self.obb_detector.num_obbs):
            if self.vis_links_indices is None or self.obb_detector.obb_robot_obb_link_indices[i] in self.vis_links_indices:
            # if self.vis_links_indices is None or i in self.vis_links_indices:
                obb_box = OBBBox(
                    self.obb_detector.obb_robot_center_pos()[i],
                    self.obb_detector.obb_robot_extents()[i],
                    self.obb_detector.obb_robot_global_rotation()[i],
                    use_wireframe=use_wireframe
                )

                # robot_geometries.append(Sphere(pos=to_numpy(self.obb_detector.obb_robot_global_translation())[i],r=0.02,c='red'))
                # robot_geometries.append(Sphere(pos=to_numpy(self.obb_detector.obb_robot_center_pos())[i],r=0.02, c='blue'))
                robot_geometries.append(Points(to_numpy(self.obb_detector.obb_robot_vertices()[i]),r=10,c='green'))
                obb_axes = OBBAxes(self.obb_detector.obb_robot_center_pos()[i], self.obb_detector.obb_robot_axes()[i])
                if collision_mat is not None and len(torch.argwhere(collision_mat[i])):
                    obb_axes.c("red")
                    robot_geometries.append(obb_axes)
                robot_geometries.append(obb_box)
        print("--------------------------------")
        if collision_mat is not None:
            for i in range(self.obb_detector.num_obbs):
                collision_links_indices = torch.argwhere(collision_mat[i])
                if len(collision_links_indices):
                    print(f"{i} collides with {collision_links_indices.tolist()}")
        return robot_geometries

    def robot_transform(self,global_translations,global_rotations):
        # start = time.time()
        self.obb_detector.update_obbs_transform(global_translations,global_rotations,from_link_transform=True)
        collision_mat = self.obb_detector.check_collision(return_obbs_collisions=True)
        # end = time.time()
        # print("time:",end-start)
        # self._robot_geometries.clear()
        self._robot_geometries = self._generate_robot_geometries(collision_mat)

    def _update_robot_geometries(self):
        pass

    def show(self):
        plotter = Plotter(axes=1, bg='white')
        plotter.show(*self.robot_geometries)



class VedoRobotWithCollision(VedoRobot):
    def __init__(self,robot_zero_pose,obb_detector:OBBRobotCollisionDetector=None,vis_links_indices=None):
        self.obb_detector = obb_detector
        self.vis_links_indices = vis_links_indices
        super().__init__(robot_zero_pose)

    @classmethod
    def from_urdf_and_detector(cls,urdf_path,obb_detector:OBBRobotCollisionDetector=None,vis_links_indices=None):
        robot_zero_pose,link_mesh_file_names = parse_urdf(urdf_path)
        vedo_robot = cls(robot_zero_pose,obb_detector,vis_links_indices)
        link_mesh_paths = [os.path.join(os.path.dirname(urdf_path),link_file) for link_file in link_mesh_file_names]
        robot_geometries = vedo_robot._generate_robot_geometries(link_mesh_paths)
        vedo_robot._original_robot_geometries = robot_geometries
        vedo_robot._robot_geometries = copy.deepcopy(vedo_robot._original_robot_geometries)
        return vedo_robot

    def robot_transform(self,global_translations,global_rotations):
        new_robot_geometries = []
        self.obb_detector.update_obbs_transform(global_translations,global_rotations,from_link_transform=True)
        collision_mat = self.obb_detector.check_collision()

        for link_idx,(link_mesh,link_global_translation,link_global_rotation) in enumerate(zip(self.robot_original_geometries, global_translations, global_rotations)):

            link_rotation_angle, link_rotation_axis = quat_to_angle_axis(quat_normalize(link_global_rotation))
            link_rotation_angle = to_numpy(link_rotation_angle)
            link_rotation_axis = to_numpy(link_rotation_axis)

            if torch.any(collision_mat[link_idx]):
                link_mesh.c("red")
                link_mesh.alpha(1)
            link_mesh.rotate(angle=link_rotation_angle, axis=link_rotation_axis, rad=True)
            link_mesh.pos(link_global_translation.numpy())
            new_robot_geometries.append(link_mesh)

        self._robot_geometries = new_robot_geometries



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

class OBBAxes(Lines):
    def __init__(self, center_pos, axes):
        axes = to_numpy(axes)/50
        start  = to_numpy(center_pos.unsqueeze(0).repeat(3,1))
        end = start + axes
        super().__init__(start, end, lw=3, c='blue')

if __name__ == '__main__':

    obb_robot = OBBRobot.from_urdf(urdf_path='asset/hu/hu_v5.urdf')
    obb_detector = OBBRobotCollisionDetector(obb_robot=obb_robot)



    # hu_vedo_robot = VedoRobot.from_urdf('asset/hu/hu_v5.urdf')
    # hu_vedo_robot.show()



