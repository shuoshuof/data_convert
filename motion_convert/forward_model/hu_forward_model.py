from typing import Union
from abc import ABC, abstractmethod
import torch.nn as nn
import torch
from poselib.poselib.core.rotation3d import *
from poselib.poselib.skeleton.skeleton3d import SkeletonMotion,SkeletonState,SkeletonTree

from motion_convert.forward_model.base_forward_model import BaseForwardModel,cal_forward_kinematics
from motion_convert.robot_config.Hu import Hu_DOF_AXIS,Hu_DOF_LOWER,Hu_DOF_UPPER



class HuForwardModel(BaseForwardModel):
    def __init__(self,skeleton_tree,device='cuda:0'):
        super().__init__(skeleton_tree,device)
        self.joint_rotation_axis = torch.eye(3)[Hu_DOF_AXIS]
    def forward_kinematics(self, motion_joint_angles, motion_root_translation, motion_root_rotation,clip_angles):
        motion_length,_,_ = motion_joint_angles.shape
        if clip_angles:
            motion_joint_angles = self._clip_angles(motion_joint_angles)
        motion_rotation_axis = self.joint_rotation_axis.repeat(motion_length,1,1).clone().to(self.device)
        motion_local_rotation = quat_from_angle_axis(motion_joint_angles.reshape(-1),motion_rotation_axis.reshape(-1,3))
        motion_local_rotation = motion_local_rotation.reshape(motion_length,self.num_joints-1,4)
        motion_local_rotation = torch.concatenate([motion_root_rotation, motion_local_rotation], dim=1)
        return super().forward_kinematics(motion_local_rotation=motion_local_rotation,motion_root_translation=motion_root_translation)

    def _clip_angles(self,motion_joint_angles):
        hu_dof_low_limit = Hu_DOF_LOWER.reshape(1,-1,1).to(self.device)
        hu_dof_high_limit = Hu_DOF_UPPER.reshape(1,-1,1).to(self.device)
        # motion_joint_angles = torch.sigmoid(motion_joint_angles) * (hu_dof_high_limit - hu_dof_low_limit) + hu_dof_low_limit
        clamped_motion_joint_angles = torch.clamp(motion_joint_angles.clone(), min=hu_dof_low_limit, max=hu_dof_high_limit)
        motion_joint_angles = (clamped_motion_joint_angles-motion_joint_angles).detach() + motion_joint_angles
        return motion_joint_angles



if __name__ == '__main__':
    import pickle
    from poselib.poselib.visualization.common import plot_skeleton_H

    with open('asset/zero_pose/hu_zero_pose.pkl', 'rb') as f:
        hu_zero_pose:SkeletonState = pickle.load(f)
    device = 'cuda:0'
    hu_forward_model = HuForwardModel(hu_zero_pose.skeleton_tree)
    motion_length = 1000
    motion_joint_angles = torch.zeros(motion_length,hu_zero_pose.num_joints-1,1).to(device)

    motion_joint_angles[:,2,:] = -1.5708
    motion_joint_angles[:,8,:] = -1.5708
    motion_joint_angles[:,3,:] = 1.5708
    motion_joint_angles[:,9,:] = 1.5708
    motion_joint_angles[:,12,:] = 1.5708

    motion_root_translation = torch.zeros(motion_length,3).to(device)
    motion_root_rotation = torch.tensor([[[0.,0.,0.,1.]]]).repeat(motion_length,1,1).to(device)
    motion_global_rotation, motion_global_translation = hu_forward_model.forward_kinematics(motion_joint_angles=motion_joint_angles,
                                                                             motion_root_translation=motion_root_translation,
                                                                             motion_root_rotation=motion_root_rotation,
                                                                             clip_angles=True)

    new_sk_state = SkeletonState.from_rotation_and_root_translation(
        hu_zero_pose.skeleton_tree,
        motion_global_rotation.cpu().detach(),
        motion_global_translation[:,0,].cpu().detach(),
        is_local=False
    )

    new_motion = SkeletonMotion.from_skeleton_state(new_sk_state,fps=30)

    plot_skeleton_H([new_motion])

    with open('motion_data/test_motion.pkl','wb') as f:
        pickle.dump(new_motion,f)

    from body_visualizer.visualizer import BodyVisualizer
    from motion_convert.robot_config.Hu import hu_graph

    body_visualizer = BodyVisualizer(hu_graph)
    for pos in motion_global_translation.detach().cpu():
        body_visualizer.step(pos)