from tqdm import tqdm
import torch
from abc import ABC, abstractmethod
from typing import Optional, Union

class BaseRetargetOptimizer(ABC):
    def __init__(self):
        self.forward_model = None
        self.optimizer = None
        self.lr_scheduler = None
        self.params = None

    def train(self, motion_data:torch.Tensor, max_epoch: int, lr: float,process_idx, **kwargs):
        motion_data_shape = motion_data.shape

        self.params = self._init_params(motion_data_shape,**kwargs)
        self.optimizer = self._set_optimizer(lr, **kwargs)
        self.lr_scheduler = self._set_lr_scheduler()

        with tqdm(total=max_epoch) as pbar:
            for epoch in range(max_epoch):
                self.optimizer.zero_grad()
                model_output = self.forward_model(**self.params)
                loss = self._loss_function(motion_data, model_output)

                self.train_step(loss)

                pbar.set_description(f'Epoch {epoch + 1}/{max_epoch}')
                pbar.set_postfix(
                    loss=loss.item(),
                    lr=self.lr_scheduler.get_last_lr() if self.lr_scheduler else lr
                )
                pbar.update(1)

    def train_step(self, loss):
        loss.backward()
        self.optimizer.step()
        if self.lr_scheduler:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(loss.item())
            else:
                self.lr_scheduler.step()


    @abstractmethod
    def _loss_function(self, motion_data, forward_model_output) -> torch.Tensor:
        pass

    @abstractmethod
    def _init_params(self,motion_data_shape,**kwargs) -> Union[list,dict]:
        pass

    @abstractmethod
    def _set_optimizer(self, lr: float, **kwargs):
        pass

    def _set_lr_scheduler(self) \
            -> Optional[torch.optim.lr_scheduler.LRScheduler]:
        # Default: No learning rate scheduler
        return None



