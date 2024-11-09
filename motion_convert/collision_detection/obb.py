# -*- coding: utf-8 -*-
"""
@Time ： 2024/11/6 11:02
@Auth ： shuoshuof
@File ：obb.py
@Project ：data_convert
"""
from typing import Union

import torch
import time

from poselib.poselib.core.rotation3d import *

class OBB:
    def __init__(
            self,
            initial_axes:torch.Tensor,
            extents:torch.Tensor,
            global_rotation:torch.Tensor,
            global_translation:torch.Tensor,
            device='cuda'
    ):
        self._initial_axes = initial_axes.to(device)
        self._extents = extents.to(device)
        self._global_rotation = global_rotation.to(device)
        self._global_translation = global_translation.to(device)
        self.device = device

        assert self._initial_axes.shape == (3, 3)
        assert self._extents.shape == (3,)
        assert self._global_rotation.shape == (4,)
        assert self._global_translation.shape == (3,)

        # need to be updated by global_translation and global_rotation
        self._axes = self._cal_axes()
        self._vertices = self._cal_vertices()

        assert self._axes.shape == (3, 3)
        assert self._vertices.shape == (8, 3)
    @property
    def axes(self):
        return self._axes.clone()
    @property
    def extents(self):
        return self._extents.clone()
    @property
    def global_rotation(self):
        return self._global_rotation.clone()
    @property
    def global_translation(self):
        return self._global_translation.clone()
    @property
    def vertices(self):
        return self._vertices.clone()

    def _cal_vertices(self):
        signs = torch.tensor([-1,1],dtype=torch.float32).to(self.device)
        vertices = torch.cartesian_prod(signs,signs,signs)@(self._extents.unsqueeze(1)*self._axes)
        # the axes have been rotated, so vertices don't need to
        return vertices.clone() + self.global_translation

    def _cal_axes(self):
        return quat_rotate(self.global_rotation.unsqueeze(0),self._initial_axes).clone()

    def update_transform(self,global_translation=None,global_rotation=None):
        if global_translation is not None:
            self._global_translation = global_translation.to(self.device)
        if global_rotation is not None:
            self._global_rotation = global_rotation.to(self.device)
        self._axes = self._cal_axes()
        self._vertices = self._cal_vertices()

class OBBCollisionDetector:
    def __init__(self,obbs:List[OBB],device='cuda'):
        self._obbs = obbs
        self._num_obbs = len(obbs)
        self.device = device
        # TODO: the collision of two near links from a robot may not be considered as collision
        self.collision_mask = torch.eye(self._num_obbs, dtype=torch.bool)

    @property
    def num_obbs(self):
        return self._num_obbs
    @property
    def obbs(self):
        return self._obbs
    def obbs_global_translation(self)->torch.Tensor:
        return torch.stack([obb.global_translation for obb in self.obbs])
    def obbs_global_rotation(self)->torch.Tensor:
        return torch.stack([obb.global_rotation for obb in self.obbs])
    def obbs_vertices(self)->torch.Tensor:
        return torch.stack([obb.vertices for obb in self.obbs])
    def obbs_axes(self)->torch.Tensor:
        return torch.stack([obb.axes for obb in self.obbs])
    def obbs_extents(self)->torch.Tensor:
        return torch.stack([obb.extents for obb in self.obbs])
    def update_obbs_transform(self,global_rotations,global_translations):
        if global_rotations is None:
            global_rotations = [None] * self.num_obbs
        if global_translations is None:
            global_translations = [None] * self.num_obbs
        for i,(global_rotation,global_translation) in enumerate(zip(global_rotations,global_translations)):
            self._obbs[i].update_transform(global_translation, global_rotation)
    def _cal_obbs_separating_axes_tensor(self):
        r"""
        main_axes: (num_obbs,num_obbs,6,3)
        cross_axes: (num_obbs,num_obbs,9,3)
        :return: splitting axes tensor with shape (num_obbs,num_obbs,15,3)
        """
        main_axes = torch.concatenate(
            [self.obbs_axes().unsqueeze(1).repeat(1,self.num_obbs,1,1),
                    self.obbs_axes().unsqueeze(0).repeat(self.num_obbs,1,1,1)],dim=-2)
        assert main_axes.shape == (self.num_obbs,self.num_obbs,6,3)

        # obbs_axes: (num_obbs,3,3) -> (num_obbs,1,3,3) -> (num_obbs,num_obbs,3,3) -> (num_obbs,num_obbs,3,1,3)
        edge1 = self.obbs_axes().unsqueeze(1).repeat(1,self.num_obbs,1,1).unsqueeze(-2)
        # obbs_axes: (num_obbs,3,3) -> (1,num_obbs,3,3) -> (num_obbs,num_obbs,3,3) -> (num_obbs,num_obbs,1,3,3)
        edge2 = self.obbs_axes().unsqueeze(0).repeat(self.num_obbs,1,1,1).unsqueeze(-3)

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
            [self.obbs_global_translation().unsqueeze(1).repeat(1,self.num_obbs,1).unsqueeze(-2),
                    self.obbs_global_translation().unsqueeze(0).repeat(self.num_obbs,1,1).unsqueeze(-2)],dim=-2)
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
            [self.obbs_vertices().unsqueeze(1).repeat(1,self.num_obbs,1,1).unsqueeze(-3),
                    self.obbs_vertices().unsqueeze(0).repeat(self.num_obbs,1,1,1).unsqueeze(-3)],dim=-3)
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
    obb1 = OBB(
        initial_axes=torch.eye(3),
        extents=torch.tensor([1,1,1]),
        global_rotation=torch.tensor([0,0,0,1]),
        global_translation=torch.tensor([0,0,0]),
    )

    obb2 = OBB(
        initial_axes=-torch.eye(3),
        extents=torch.tensor([1,1,1]),
        global_rotation=torch.tensor([0,0,0,1]),
        global_translation=torch.tensor([3,0,0]),
    )

    # obb3 = OBB(
    #     initial_axes=torch.eye(3),
    #     extents=torch.tensor([1,1,1]),
    #     global_rotation=torch.tensor([0,0,0,1]),
    #     global_translation=torch.tensor([0,3,0]),
    # )

    obb_detector = OBBCollisionDetector([obb1, obb2])
    obb_detector.check_collision()


    # for i in range(100):
    #     start = time.time()
    #     # obb_detector.update_obbs_transform(
    #     #     global_rotations=torch.randn(obb_detector.num_obbs,4),
    #     #     global_translations=torch.randn(obb_detector.num_obbs,3),
    #     # )
    #     obb_detector.check_collision()
    #     end = time.time()
    #
    #     print(f"cal_obbs_separating_axes: {(end - start) }")




