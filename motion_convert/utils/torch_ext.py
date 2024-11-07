# -*- coding: utf-8 -*-
"""
@Time ： 2024/11/7 20:30
@Auth ： shuoshuof
@File ：torch_ext.py
@Project ：data_convert
"""
import torch

def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    else:
        return tensor


def to_torch(tensor):
    if torch.is_tensor(tensor):
        return tensor
    else:
        return torch.from_numpy(tensor)