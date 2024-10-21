from typing import Union
from motion_convert.retarget_optimizer.base_retarget_optimizer import BaseRetargetOptimizer
import torch

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

class HuRetargetOptimizer(BaseRetargetOptimizer):
    def __init__(self,hu_forward_model):
        super(HuRetargetOptimizer,self).__init__()
        self.hu_forward_model = hu_forward_model
    def train(self, motion_data:torch.Tensor, max_epoch: int, lr: float,process_idx, **kwargs):
        super(HuRetargetOptimizer,self).train(motion_data,max_epoch,lr,process_idx,**kwargs)

    def _init_params(self,motion_data_shape,**kwargs):
        motion_length, joint_num, _ = motion_data_shape
        self.motion_joint_angles = torch.zeros(motion_length, joint_num, 1, dtype=torch.float32, device='cuda')
        return {"motion_joint_angles": self.motion_joint_angles}

    def _loss_function(self, motion_data, forward_model_output) -> torch.Tensor:

        pass
    def _set_lr_scheduler(self):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=50, factor=0.5)
        return scheduler
    def _set_optimizer(self, lr: float, **kwargs):
        return torch.optim.Adam(self.params.values(), lr=lr, **kwargs)