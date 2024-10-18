from motion_convert.retarget_optimizer.base_retarget_optimizer import BaseRetargetOptimizer
import torch
import numpy as np
from smplx import SMPL

class BaseSMPLRetargetOptimizer(BaseRetargetOptimizer):
    def __init__(self,smpl_model_path, gender="NEUTRAL"):
        super().__init__()
        self.forward_model = SMPL(smpl_model_path, gender=gender).cuda()

    def train(self, motion_data, max_epoch: int, lr: float, **kwargs):
        super().train(motion_data, max_epoch, lr, **kwargs)

        optimized_data = {
            'pose_aa': torch.concatenate([self.params['global_orient'],self.params['body_pose']],dim=1).detach().cpu().numpy(),
            'beta': np.zeros(10),
            'trans': self.params['transl'].detach().cpu().numpy(),
            'gender': 'neutral',
            'fps': 20
        }
        return optimized_data
    def _init_params(self,motion_data_shape, **kwargs):

        num_frames,_,_ = motion_data_shape

        body_pose = torch.rand((num_frames,23*3),dtype=torch.float32, requires_grad=True, device="cuda")
        global_orient = torch.rand((num_frames,3),dtype=torch.float32, requires_grad=True, device="cuda")
        transl = torch.rand((num_frames,3),dtype=torch.float32, requires_grad=True, device="cuda")
        return {'body_pose':body_pose, 'global_orient':global_orient, 'transl':transl}

    def _loss_function(self, motion_data, forward_model_output) -> torch.Tensor:
        motion_body_positions = torch.tensor(motion_data[:, :24, :], dtype=torch.float32).cuda()
        smpl_body_positions = forward_model_output.joints[:, :24, :]
        loss = torch.mean((smpl_body_positions - motion_body_positions) ** 2)
        return loss
    def _set_lr_scheduler(self):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=50, verbose=True, factor=0.5)
        return scheduler
    def _set_optimizer(self, lr: float, **kwargs):
        return torch.optim.Adam(self.params.values(), lr=lr, **kwargs)