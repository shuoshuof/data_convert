from typing import Union
from abc import ABC, abstractmethod
import torch.nn as nn
import torch
from poselib.poselib.core.rotation3d import *
from poselib.poselib.skeleton.skeleton3d import SkeletonMotion,SkeletonState,SkeletonTree

from motion_convert.forward_model.base_forward_model import BaseForwardModel,forward_kinematics
Hu_DOF_IDX_MAPPING = [
    2, 0, 1, 1, 1, 0,
    2, 0, 1, 1, 1, 0,
    2,
    1, 0, 2, 1, 0, 1, 2, 1, 1,
    1, 0, 2, 1, 0, 1, 2, 1, 1,
    2, ]
Hu_DOF_LOWER = torch.tensor([
    -0.1745, -0.3491, -1.9635, 0.0997, -0.6981, -0.3665,
    -0.1745, -0.3491, -1.9635, 0.0997, -0.6981, -0.3665,
    -1.0,
    -3.1416, 0., -1.5708, 0., -1.5708, -0.7854, -0.7854, 0., -0.044,
    -3.1416, -1.5708, -1.5708, 0., -1.5708, -0.7854, -0.7854, 0., -0.044,
    -1.0])
Hu_DOF_UPPER = torch.tensor([
    0.1745, 0.3491, 1.9635, 2.618, 0.6981, 0.3665,
    0.1745, 0.3491, 1.9635, 2.618, 0.6981, 0.3665,
    1.0,
    1.0472, 1.5708, 1.5708, 1.5708, 1.5708, 0.7854, 0.7854, 0.044, 0.,
    1.0472, 0., 1.5708, 1.5708, 1.5708, 0.7854, 0.7854, 0.044, 0.,
    1.0])


class HuForwardModel(BaseForwardModel):
    def __init__(self,skeleton_tree,device='cuda:0'):
        super().__init__(skeleton_tree,device)
        self.joint_rotation_axis = torch.eye(3)[Hu_DOF_IDX_MAPPING]
    def forward_kinematics(self, motion_joint_angles, motion_root_translation, motion_root_rotation):
        motion_length,_ = motion_joint_angles.shape
        motion_rotation_axis = self.joint_rotation_axis.repeat(motion_length,1,1).clone().to(self.device)
        motion_local_rotation = quat_from_angle_axis(motion_joint_angles.reshape(-1),motion_rotation_axis.reshape(-1,3))
        motion_local_rotation = motion_local_rotation.reshape(motion_length,self.num_joints-1,4)
        # TODO: clip rotation
        motion_local_rotation = torch.concatenate([motion_root_rotation.unsqueeze(1), motion_local_rotation], dim=1)
        return super().forward_kinematics(motion_local_rotation=motion_local_rotation,motion_root_translation=motion_root_translation)
    def _clip_rotation(self, rotation):
        pass





