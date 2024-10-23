from typing import Union
import torch

from poselib.poselib.skeleton.skeleton3d import SkeletonMotion,SkeletonState,SkeletonTree
from poselib.poselib.core.rotation3d import *

from motion_convert.retarget_optimizer.base_retarget_optimizer import BaseRetargetOptimizer
from motion_convert.forward_model.hu_forward_model import HuForwardModel
from motion_convert.robot_config.Hu import Hu_DOF_IDX_MAPPING as hu_rotation_axis_mapping

class HuRetargetOptimizer(BaseRetargetOptimizer):
    def __init__(self,hu_sk_tree:SkeletonTree):
        super(HuRetargetOptimizer,self).__init__()
        self.hu_sk_tree = hu_sk_tree
        self.forward_model = HuForwardModel(skeleton_tree=hu_sk_tree)
        self.loss_fun = HuRetargetLossFun()
    def train(self, motion_data:SkeletonMotion, max_epoch: int, lr: float,process_idx, **kwargs)->SkeletonMotion:
        self.motion_local_rotation = motion_data.local_rotation.to(device=self.device)
        # self.motion_root_rotation = motion_data.local_rotation[:, 0, :].to(device=self.device)
        self.motion_length, self.joint_num, _ = self.motion_local_rotation.shape
        self.motion_root_translation = motion_data.root_translation.to(device=self.device)

        super(HuRetargetOptimizer,self).train(motion_data,max_epoch,lr,process_idx,**kwargs)

        motion_global_rotation, motion_global_translation = \
            self.forward_model.forward_kinematics(self.motion_joint_angles,self.motion_root_translation,self.motion_root_rotation)

        retargeted_skeleton_state = SkeletonState.from_rotation_and_root_translation(
            self.hu_sk_tree,
            motion_global_rotation.detach().cpu(),
            self.motion_root_translation.cpu(),
            is_local=False
        )
        retargeted_skeleton_motion = SkeletonMotion.from_skeleton_state(retargeted_skeleton_state,fps=motion_data.fps)

        return retargeted_skeleton_motion

    def _init_params(self,**kwargs):
        self.motion_joint_angles = torch.zeros(self.motion_length, self.joint_num-1, 1,
                                               dtype=torch.float32,requires_grad=True, device=self.device)
        self.motion_root_rotation = torch.tensor([[[0,0,0,1]]]*self.motion_length,
                                                 dtype=torch.float32, requires_grad=True,device=self.device)
        return {"motion_joint_angles": self.motion_joint_angles,"motion_root_rotation": self.motion_root_rotation}
    def _model_forward(self, motion_joint_angles,motion_root_rotation):
        motion_global_rotation, motion_global_translation = \
            self.forward_model.forward_kinematics(motion_joint_angles,self.motion_root_translation,motion_root_rotation)
        return motion_global_translation
    def _cal_loss(self, forward_model_output, motion_data:SkeletonMotion) -> Union[torch.Tensor, torch.nn.Module]:
        return self.loss_fun(forward_model_output, motion_data.global_translation.to(self.device))
    def _set_lr_scheduler(self):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=50, factor=0.5)
        return scheduler
    def _set_optimizer(self, lr: float, **kwargs):
        return torch.optim.Adam(self.params.values(), lr=lr, **kwargs)


class HuRetargetLossFun(torch.nn.Module):
    def __init__(self,device="cuda:0"):
        super().__init__()
        self.device = device
        self.error_loss = torch.nn.MSELoss(reduction='none')
        self.loss_weight = torch.Tensor([
            0,
            0, 0, 0, 1, 1,1,
            0, 0, 0, 1, 1,1,
            0,
            2, 2, 1, 1, 1, 1, 2, 1, 1,
            2, 2, 1, 1, 1, 1, 2, 1, 1,
            2,
        ]).to(self.device).view(1,33,1)
    def forward(self, input, target):
        motion_length, _, _ = input.shape
        loss = self.error_loss(input, target)
        loss = loss * self.loss_weight
        return loss.mean()
