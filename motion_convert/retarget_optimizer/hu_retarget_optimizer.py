from typing import Union
from motion_convert.retarget_optimizer.base_retarget_optimizer import BaseRetargetOptimizer
import torch


class HuRetargetOptimizer(BaseRetargetOptimizer):
    def __init__(self,hu_forward_model):
        super(HuRetargetOptimizer,self).__init__()
        self.hu_forward_model = hu_forward_model
    def train(self, motion_data:torch.Tensor, max_epoch: int, lr: float,process_idx, **kwargs):
        super(HuRetargetOptimizer,self).train(motion_data,max_epoch,lr,process_idx,**kwargs)

    def _init_params(self,motion_data_shape,**kwargs):
