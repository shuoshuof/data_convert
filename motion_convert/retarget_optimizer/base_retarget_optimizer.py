from tqdm import tqdm
import torch
from abc import ABC, abstractmethod
from typing import Optional, Union
from poselib.poselib.skeleton.skeleton3d import SkeletonMotion

class BaseRetargetOptimizer(ABC):
    def __init__(self,device='cuda:0'):
        self.robot_model = None
        self.optimizer = None
        self.lr_scheduler = None
        self.params = None
        self.device = device
    def train(self, motion_data:Union[torch.Tensor,SkeletonMotion], max_epoch: int, lr: float,process_idx, **kwargs):

        self.params = self._init_params(**kwargs)
        self.optimizer = self._set_optimizer(lr, **kwargs)
        self.lr_scheduler = self._set_lr_scheduler()

        # with tqdm(total=max_epoch) as pbar:
        #     for epoch in range(max_epoch):
        #         self.optimizer.zero_grad()
        #         model_output = self._model_forward(**self.params)
        #         loss = self._cal_loss(model_output,motion_data)
        #
        #         self.train_step(loss)
        #
        #         pbar.set_description(f'Epoch {epoch + 1}/{max_epoch}')
        #         pbar.set_postfix(
        #             loss=loss.item(),
        #             lr=self.lr_scheduler.get_last_lr() if self.lr_scheduler else lr
        #         )
        #         pbar.update(1)
        for epoch in range(max_epoch):
            self.optimizer.zero_grad()
            model_output = self._model_forward(**self.params)
            loss = self._cal_loss(model_output,motion_data)

            self.train_step(loss)

    def train_step(self, loss):
        loss.backward()
        self.optimizer.step()
        if self.lr_scheduler:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(loss.item())
            else:
                self.lr_scheduler.step()

    def _model_forward(self, **kwargs):
        return self.robot_model(**kwargs)

    @abstractmethod
    def _cal_loss(self,forward_model_output,motion_data) -> Union[torch.Tensor,torch.nn.Module]:
        raise NotImplementedError

    @abstractmethod
    def _init_params(self,**kwargs) -> Union[list,dict]:
        raise NotImplementedError

    @abstractmethod
    def _set_optimizer(self, lr: float, **kwargs):
        raise NotImplementedError

    def _set_lr_scheduler(self) \
            -> Optional[torch.optim.lr_scheduler.LRScheduler]:
        # Default: No learning rate scheduler
        return None



