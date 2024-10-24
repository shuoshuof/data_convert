from typing import Union
import torch

from poselib.poselib.skeleton.skeleton3d import SkeletonMotion,SkeletonState,SkeletonTree
from poselib.poselib.core.rotation3d import *

from motion_convert.retarget_optimizer.base_retarget_optimizer import BaseRetargetOptimizer
from motion_convert.forward_model.base_forward_model import BaseForwardModel

class NoitomRetargetOptimizer(BaseRetargetOptimizer):
    def __init__(self, noitom_sk_tree:SkeletonTree):
        super(NoitomRetargetOptimizer,self).__init__()
        self.noitom_sk_tree = noitom_sk_tree
        self.forward_model = BaseForwardModel(skeleton_tree=noitom_sk_tree)
    def train(self, motion_global_translation, max_epoch: int, lr: float, process_idx, **kwargs)->SkeletonMotion:
        motion_global_translation = motion_global_translation.to(self.device)
        self.motion_length,self.joint_num,_ = motion_global_translation.shape
        self.motion_root_translation = motion_global_translation[:, 0, :].to(self.device)

        super(NoitomRetargetOptimizer,self).train(motion_global_translation, max_epoch, lr, process_idx, **kwargs)

        motion_global_rotation, motion_global_translation = \
            self.forward_model.forward_kinematics(motion_local_rotation=self.motion_local_rotation,motion_root_translation=self.motion_root_translation)

        retargeted_skeleton_state = SkeletonState.from_rotation_and_root_translation(
            self.noitom_sk_tree,
            motion_global_rotation.detach().cpu(),
            self.motion_root_translation.cpu(),
            is_local=True
        )
        retargeted_skeleton_motion = SkeletonMotion.from_skeleton_state(retargeted_skeleton_state, fps=30)

        return retargeted_skeleton_motion

    def _init_params(self,**kwargs):
        self.motion_local_rotation = torch.tensor([[[0, 0, 0, 1]] * self.joint_num] * self.motion_length,
                                                  dtype=torch.float32, requires_grad=True, device=self.device)
        return {"motion_local_rotation": self.motion_local_rotation}
    def _model_forward(self, motion_local_rotation):
        motion_global_rotation, motion_global_translation = \
            self.forward_model.forward_kinematics(motion_local_rotation=motion_local_rotation,motion_root_translation=self.motion_root_translation)
        return motion_global_translation
    def _cal_loss(self, forward_model_output, motion_data) -> Union[torch.Tensor, torch.nn.Module]:
        return torch.mean((forward_model_output-motion_data)**2)
    def _set_lr_scheduler(self):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=50, factor=0.5)
        return scheduler
    def _set_optimizer(self, lr: float, **kwargs):
        return torch.optim.Adam(self.params.values(), lr=lr, **kwargs)