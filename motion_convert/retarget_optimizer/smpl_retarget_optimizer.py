from motion_convert.retarget_optimizer.base_retarget_optimizer import BaseRetargetOptimizer
import torch
import numpy as np
from smplx import SMPL

class BaseSMPLRetargetOptimizer(BaseRetargetOptimizer):
    def __init__(self,smpl_model_path, gender="NEUTRAL"):
        super().__init__()
        self.robot_model = SMPL(smpl_model_path, gender=gender).cuda()

    def train(self, motion_data, max_epoch: int, lr: float, **kwargs):
        self.motion_data = motion_data.shape
        super().train(motion_data, max_epoch, lr, **kwargs)

        # optimized_data = {
        #     'pose_aa': torch.concatenate([self.params['global_orient'],self.params['body_pose']],dim=1).detach().cpu().numpy(),
        #     'beta': np.zeros(10),
        #     'transl': self.params['transl'].detach().cpu().numpy(),
        #     'gender': 'neutral',
        #     'fps': 20
        # }

        optimized_data = {
            'pose_aa': torch.concatenate([self.params['global_orient'],self.params['body_pose']],dim=1).detach().cpu(),
            'beta': torch.zeros(10),
            'transl': self.params['transl'].detach().cpu(),
            'gender': 'neutral',
            'fps': 20
        }
        return optimized_data
    def _init_params(self,**kwargs):

        motion_length,_,_ = self.motion_data

        body_pose = torch.rand((motion_length,23*3),dtype=torch.float32, requires_grad=True, device=self.device)
        global_orient = torch.rand((motion_length,3),dtype=torch.float32, requires_grad=True, device=self.device)
        transl = torch.rand((motion_length,3),dtype=torch.float32, requires_grad=True, device=self.device)
        return {'body_pose':body_pose, 'global_orient':global_orient, 'transl':transl}

    def _cal_loss(self, forward_model_output,motion_data) -> torch.Tensor:
        motion_body_positions = motion_data[:, :24, :].clone().detach().cuda()
        smpl_body_positions = forward_model_output.joints[:, :24, :]
        loss = torch.mean((smpl_body_positions - motion_body_positions) ** 2)
        return loss
    def _set_lr_scheduler(self):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=50, factor=0.5)
        return scheduler
    def _set_optimizer(self, lr: float, **kwargs):
        return torch.optim.Adam(self.params.values(), lr=lr, **kwargs)
    

