# -*- coding: utf-8 -*-
"""
@Time ： 2024/11/6 11:02
@Auth ： shuoshuof
@File ：obb.py
@Project ：data_convert
"""
from typing import Union

import torch
import numpy as np

from poselib.poselib.core.rotation3d import *

class OBB:
    def __init__(
            self,
            center:torch.Tensor,
            axes:torch.Tensor,
            extents:torch.Tensor,
            global_rotation:torch.Tensor,
            global_translation:torch.Tensor,
    ):
        self._center = center
        self._axes = axes
        self._extents = extents
        self._global_rotation = global_rotation
        self._global_translation = global_translation

        assert self._center.shape == (3,)
        assert self._axes.shape == (3, 3)
        assert self._extents.shape == (3,)
        assert self._global_rotation.shape == (4,)
        assert self._global_translation.shape == (3,)

        self._vertices = self.cal_vertices()
        assert self._vertices.shape == (8, 3)
    @property
    def center(self):
        return self._center.clone()
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
    def cal_vertices(self):
        signs = torch.tensor([-1,1],dtype=torch.float32)
        vertices = torch.cartesian_prod(*signs[None,:]*self._extents[:,None])
        return quat_rotate(self.global_rotation.unsqueeze(0),vertices) + self.global_translation + self.center

    def update_transform(self,global_translation=None,global_rotation=None):
        if global_translation is not None:
            self._global_translation = global_translation
        if global_rotation is not None:
            self._global_rotation = global_rotation
        self._vertices = self.cal_vertices()

class OBBCollisionDetector:
    def __init__(self,obbs:List[OBB]):
        self._obbs = obbs
        self._num_obbs = len(obbs)
        # TODO: the collision of two near links from a robot may not be considered as collision
        self.collision_mask = torch.eye(self._num_obbs, dtype=torch.bool)

    @property
    def num_obbs(self):
        return self._num_obbs

    def get_obbs(self):
        return self._obbs

    def update_obbs_transform(self,global_rotations,global_translations):
        for i,(global_rotation,global_translation) in enumerate(zip(global_rotations,global_translations)):
            self._obbs[i].update_transform(global_translation, global_rotation)



if __name__ == '__main__':
    obb = OBB(
        center=torch.tensor([0,0,0]),
        axes=torch.eye(3),
        extents=torch.tensor([1,2,3]),
        global_rotation=torch.tensor([0,0,0,1]),
        global_translation=torch.tensor((1,2,3)),
    )
    print(obb.vertices)



