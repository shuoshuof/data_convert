# -*- coding: utf-8 -*-
"""
@Time ： 2024/11/9 17:50
@Auth ： shuoshuof
@File ：obb_robot.py
@Project ：data_convert
"""

from typing import Union,List
import time
import copy

import torch

import trimesh

from poselib.poselib.core.rotation3d import quat_rotate
from poselib.poselib.skeleton.skeleton3d import SkeletonMotion,SkeletonTree,SkeletonState

from motion_convert.utils.torch_ext import to_torch

class OBBRobot:
    def __init__(self,urdf_path,device='cuda'):
        self.device = device
        self.load_robot(urdf_path)
    def load_robot(self,urdf_path:str):
        from motion_convert.utils.parse_urdf import parse_urdf,cal_urdf_mesh_bounding_boxes

        robot_zero_pose,_ = parse_urdf(urdf_path)
        meshes_bounding_boxes = cal_urdf_mesh_bounding_boxes(urdf_path)
        self._init_obbs(robot_zero_pose,meshes_bounding_boxes)

    def _init_obbs(self,robot_zero_pose:SkeletonState,meshes_bounding_boxes:List[trimesh.primitives.Box]):
        self._num_obbs = len(meshes_bounding_boxes)

        self._initial_axes = torch.zeros((self.num_obbs,3,3)).to(self.device)
        self._extents = torch.zeros((self.num_obbs,3)).to(self.device)
        self._origins = torch.zeros((self.num_obbs,3)).to(self.device)
        self._global_rotations = torch.zeros((self.num_obbs,4)).to(self.device)
        self._global_translations = torch.zeros((self.num_obbs,3)).to(self.device)

        for joint_idx, box in enumerate(meshes_bounding_boxes):
            self._initial_axes[joint_idx] = torch.eye(3)
            self._extents[joint_idx] = to_torch(copy.deepcopy(box.extents))/2
            self._origins[joint_idx] = to_torch(copy.deepcopy(box.transform[:3,3]))
            self._global_rotations[joint_idx] = torch.tensor([0,0,0,1])
            self._global_translations[joint_idx] = robot_zero_pose.global_translation[joint_idx]

        self._axes = self._cal_axes()
        self._vertices = self._cal_vertices()

        assert self._axes.shape == (self.num_obbs,3,3)
        assert self._vertices.shape == (self.num_obbs,8,3)

    @property
    def num_obbs(self):
        return self._num_obbs
    @property
    def axes(self):
        return self._axes.clone()
    @property
    def extents(self):
        return self._extents.clone()
    @property
    def origins(self):
        return self._origins.clone()
    @property
    def global_rotations(self):
        return self._global_rotations.clone()
    @property
    def global_translations(self):
        return self._global_translations.clone()
    @property
    def vertices(self):
        return self._vertices.clone()
    @property
    def center_pos(self):
        return self.global_translations + self.origins

    def _cal_axes(self):
        return quat_rotate(self.global_rotations.unsqueeze(1),self._initial_axes.clone())

    def _cal_vertices(self):
        signs = torch.tensor([-1,1],dtype=torch.float32).to(self.device)
        vertices = torch.cartesian_prod(signs,signs,signs)@(self.extents.unsqueeze(1)*self._axes)
        # the axes have been rotated, so vertices don't need to
        return vertices.clone() + self.global_translations.unsqueeze(1) + self.origins.unsqueeze(1)

    def update_transform(self, global_translations=None, global_rotations=None):
        assert global_translations.shape == (self._num_obbs,3)
        assert global_rotations.shape == (self._num_obbs,4)

        if global_translations is not None:
            self._global_translation = global_translations.to(self.device)
        if global_rotations is not None:
            self._global_rotation = global_rotations.to(self.device)
        self._axes = self._cal_axes()
        self._vertices = self._cal_vertices()

class OBBRobotCollisionDetector:
    def __init__(self,obb_robot:OBBRobot,device='cuda'):
        self._obb_robot = obb_robot
        self._num_obbs = self._obb_robot.num_obbs
        self.device = device
        # TODO: the collision of two near links from a robot may not be considered as collision
        self.collision_mask = torch.eye(self._num_obbs, dtype=torch.bool)

    @property
    def num_obbs(self):
        return self._num_obbs
    @property
    def obb_robot(self):
        return self._obb_robot
    def obb_robot_global_translation(self)->torch.Tensor:
        return self.obb_robot.global_translations
    def obb_robot_global_rotation(self)->torch.Tensor:
        return self.obb_robot.global_rotations
    def obb_robot_center_pos(self)->torch.Tensor:
        return self.obb_robot.center_pos
    def obb_robot_vertices(self)->torch.Tensor:
        return self.obb_robot.vertices
    def obb_robot_axes(self)->torch.Tensor:
        return self.obb_robot.axes
    def obb_robot_extents(self)->torch.Tensor:
        return self.obb_robot.extents
    def update_obbs_transform(self,global_rotations,global_translations):
        if global_rotations is None:
            global_rotations = [None] * self.num_obbs
        if global_translations is None:
            global_translations = [None] * self.num_obbs
        self.obb_robot.update_transform(global_translations=global_translations,global_rotations=global_rotations)

    def _cal_obbs_separating_axes_tensor(self):
        r"""
        main_axes: (num_obbs,num_obbs,6,3)
        cross_axes: (num_obbs,num_obbs,9,3)
        :return: splitting axes tensor with shape (num_obbs,num_obbs,15,3)
        """
        main_axes = torch.concatenate(
            [self.obb_robot_axes().unsqueeze(1).repeat(1, self.num_obbs, 1, 1),
             self.obb_robot_axes().unsqueeze(0).repeat(self.num_obbs, 1, 1, 1)],dim=-2)
        assert main_axes.shape == (self.num_obbs,self.num_obbs,6,3)

        # obbs_axes: (num_obbs,3,3) -> (num_obbs,1,3,3) -> (num_obbs,num_obbs,3,3) -> (num_obbs,num_obbs,3,1,3)
        edge1 = self.obb_robot_axes().unsqueeze(1).repeat(1, self.num_obbs, 1, 1).unsqueeze(-2)
        # obbs_axes: (num_obbs,3,3) -> (1,num_obbs,3,3) -> (num_obbs,num_obbs,3,3) -> (num_obbs,num_obbs,1,3,3)
        edge2 = self.obb_robot_axes().unsqueeze(0).repeat(self.num_obbs, 1, 1, 1).unsqueeze(-3)

        cross_axes = torch.cross(edge1,edge2,dim=-1).view(self.num_obbs,self.num_obbs,9,3)
        # the cross product of two parallel edges is zero, set a default vector
        default_vector = torch.tensor([1, 0, 0],dtype=torch.float32,device=self.device)
        cross_axes = torch.where(cross_axes.norm(dim=-1,keepdim=True)<=1e-6,default_vector,cross_axes)
        assert cross_axes.shape == (self.num_obbs,self.num_obbs,9,3)

        separating_axes = torch.cat([main_axes,cross_axes],dim=-2)
        assert separating_axes.shape == (self.num_obbs,self.num_obbs,15,3)

        return separating_axes
    def _cal_obbs_centers_tensor(self):
        r"""
        return the diff between each pair of obbs's centers with shape (num_obbs,num_obbs,2,3)
        """
        obbs_centers =  torch.concatenate(
            [self.obb_robot_center_pos().unsqueeze(1).repeat(1, self.num_obbs, 1).unsqueeze(-2),
             self.obb_robot_center_pos().unsqueeze(0).repeat(self.num_obbs, 1, 1).unsqueeze(-2)],dim=-2)
        assert obbs_centers.shape == (self.num_obbs,self.num_obbs,2,3)
        return obbs_centers

    def _cal_obbs_centers_diff_tensor(self, obbs_centers_tensor):
        return (obbs_centers_tensor.clone()[..., [0], :]- obbs_centers_tensor.clone()[..., [1], :]).abs()

    def _cal_obbs_centers_diff_proj_tensor(self,obb_separating_axes_tensor,obbs_centers_diff_tensor):
        return torch.sum(obb_separating_axes_tensor*obbs_centers_diff_tensor,dim=-1).abs()

    def _cal_obbs_vertices_tensor(self):
        r"""
        :return: vertices tensor with shape (num_obbs,num_obbs,2,8,3)
        """
        obbs_vertices = torch.concatenate(
            [self.obb_robot_vertices().unsqueeze(1).repeat(1, self.num_obbs, 1, 1).unsqueeze(-3),
             self.obb_robot_vertices().unsqueeze(0).repeat(self.num_obbs, 1, 1, 1).unsqueeze(-3)],dim=-3)
        assert obbs_vertices.shape == (self.num_obbs,self.num_obbs,2,8,3)
        return obbs_vertices
    def _cal_obbs_vertices_vec_proj_dist_tensor(self, obbs_centers_tensor,obb_separating_axes_tensor):
        r"""
        :param obbs_centers_tensor: shape (num_obbs,num_obbs,2,3)
        :param obb_separating_axes_tensor: shape (num_obbs,num_obbs,15,3)
        :return: projection distances tensor with shape (num_obbs,num_obbs,2,15)
        """
        obbs_vertices_tensor = self._cal_obbs_vertices_tensor()
        obbs_vertices_vec_tensor = obbs_vertices_tensor - obbs_centers_tensor.unsqueeze(-2)
        # [ (num_obbs,num_obbs,2,8,3)->(num_obbs,num_obbs,2,8,1,3)] [(num_obbs,num_obbs,15,3)->(num_obbs,num_obbs,1,1,15,3)]
        # -> (num_obbs,num_obbs,2,8,15)
        obbs_vertices_vec_proj = torch.sum(obbs_vertices_vec_tensor.unsqueeze(-2)*obb_separating_axes_tensor.unsqueeze(2).unsqueeze(-3),dim=-1)
        assert obbs_vertices_vec_proj.shape == (self.num_obbs,self.num_obbs,2,8,15)

        obbs_vertices_vec_proj_dist_tensor = torch.abs(obbs_vertices_vec_proj)
        obbs_vertices_vec_proj_dist_tensor = torch.max(obbs_vertices_vec_proj_dist_tensor,dim=-2).values
        assert obbs_vertices_vec_proj_dist_tensor.shape == (self.num_obbs,self.num_obbs,2,15)

        return obbs_vertices_vec_proj_dist_tensor

    def check_collision(self):
        r"""
              separating axes              vertices                centers diff             projection
        (num_obbs,num_obbs,15,3), (num_obbs,num_obbs,2,8,3), (num_obbs,num_obbs,1,3)   (num_obbs,num_obbs,15)
        :return: a collision matrix with shape (num_obbs,num_obbs)
        """
        # TODO: filter the collision between two obbs whose distance are far

        obb_separating_axes_tensor = self._cal_obbs_separating_axes_tensor()

        obbs_centers_tensor = self._cal_obbs_centers_tensor()
        obbs_centers_diff_tensor = self._cal_obbs_centers_diff_tensor(obbs_centers_tensor)

        # (num_obbs,num_obbs,15,3) (num_obbs,num_obbs,1,3) -> (num_obbs,num_obbs,15)
        obbs_centers_diff_proj_tensor = self._cal_obbs_centers_diff_proj_tensor(obb_separating_axes_tensor,obbs_centers_diff_tensor)

        assert obbs_centers_diff_proj_tensor.shape == (self.num_obbs,self.num_obbs,15)

        obbs_vertices_vec_proj_dist_tensor = self._cal_obbs_vertices_vec_proj_dist_tensor(obbs_centers_tensor,obb_separating_axes_tensor)
        obbs_vertices_vec_proj_dist_sum_tensor = torch.sum(obbs_vertices_vec_proj_dist_tensor,dim=-2)
        assert obbs_vertices_vec_proj_dist_sum_tensor.shape == (self.num_obbs,self.num_obbs,15)

        collision_mat = torch.where(obbs_centers_diff_proj_tensor>obbs_vertices_vec_proj_dist_sum_tensor,0,1)
        collision_mat = torch.min(collision_mat, dim=-1).values

        assert torch.allclose(collision_mat, collision_mat.transpose(0,1),atol=1e-6), 'collision_mat is not symmetric'
        assert collision_mat.shape == (self.num_obbs,self.num_obbs)

        mask = torch.logical_not(torch.eye(self.num_obbs)).to(self.device)

        return (collision_mat*mask).to(torch.bool)


if __name__ == '__main__':

    obb_robot = OBBRobot(urdf_path='asset/hu/hu_v5.urdf')
    obb_detector = OBBRobotCollisionDetector(obb_robot=obb_robot)

    import time
    for i in range(100):
        start = time.time()
        obb_detector.update_obbs_transform(
            global_rotations=torch.randn(obb_detector.num_obbs,4),
            global_translations=torch.randn(obb_detector.num_obbs,3),
        )
        obb_detector.check_collision()
        end = time.time()

        print(f"cal_obb: {(end - start) }")