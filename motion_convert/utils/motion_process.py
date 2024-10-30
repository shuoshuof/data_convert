import math
import numpy as np
import torch
from scipy.interpolate import interp1d

from poselib.poselib.skeleton.skeleton3d import SkeletonMotion,SkeletonState,SkeletonTree

def fix_root(motion:SkeletonMotion):
    new_root_translation = motion.root_translation.clone()
    new_root_translation[:,0] = 0
    new_state = SkeletonState.from_rotation_and_root_translation(
        motion.skeleton_tree, motion.local_rotation, new_root_translation, is_local=True)
    return SkeletonMotion.from_skeleton_state(new_state, motion.fps)

def move_feet_on_the_ground(motion):
    min_h = torch.min(motion.root_translation[:, 2])
    new_root_translation = motion.root_translation.clone()
    new_root_translation[:, 2] -= min_h

    new_state = SkeletonState.from_rotation_and_root_translation(
        motion.skeleton_tree, motion.local_rotation, new_root_translation, is_local=True)
    return SkeletonMotion.from_skeleton_state(new_state, motion.fps)

def height_adjustment(motion:SkeletonMotion, cal_interval=1.2, rate = 0.2,deg=4):
    # cal interval is in seconds
    # 消除高度漂移，但不会过滤跳跃,乔丹滞空最多为1.2s，cal_interval以此为极限

    motion_global_translation = motion.global_translation
    # motion_global_translation = filter_data(motion_global_translation,alpha=0.95)
    motion_min_height = torch.min(motion_global_translation[...,2].clone(),dim=1).values
    motion_length,_ = motion.root_translation.shape
    fps = motion.fps
    win_length = int(cal_interval * fps)
    if (motion_length-win_length)/(win_length//2)+1<deg:
        win_length  = math.floor(motion_length/(deg+1/2))
    new_motion_root_translation = motion.root_translation.clone()

    half_win = win_length // 2
    motion_indices = torch.arange(half_win, motion_length - half_win, step=win_length // 2)
    min_heights = []

    for motion_idx in motion_indices:
        motion_slice = motion_global_translation[motion_idx-half_win:motion_idx+half_win, :, :].clone()
        min_z = torch.min(motion_slice[:,:,2], dim=1).values
        sorted_z, sorted_indices = torch.sort(min_z)
        cal_num = max(int(rate * win_length), 1)
        mean_index = (sorted_indices[:cal_num].float().mean() + motion_idx - half_win)
        mean_height = sorted_z[:cal_num].mean()
        min_heights.append((mean_index.item(), mean_height.item()))

    ground_height = np.array(min_heights)
    indices = ground_height[:,0]
    heights = ground_height[:,1]

    coefficients = np.polyfit(indices,heights,deg)
    wave = np.poly1d(coefficients)

    interpolated_heights = torch.Tensor(wave(np.arange(0,motion_length)))

    new_motion_root_translation[:,2] -= interpolated_heights
    # new_motion_root_translation[:,2] += interpolated_heights-motion_min_height

    new_state = SkeletonState.from_rotation_and_root_translation(
        motion.skeleton_tree,
        motion.global_rotation,
        new_motion_root_translation,
        is_local=False
    )
    new_motion = SkeletonMotion.from_skeleton_state(new_state,fps)
    return new_motion

def rescale_motion_to_standard_size(motion_global_translation, standard_skeleton:SkeletonTree):
    rescaled_motion_global_translation = motion_global_translation.clone()
    for joint_idx,parent_idx in enumerate(standard_skeleton.parent_indices):
        if parent_idx == -1:
            pass
        else:
            scale =  torch.linalg.norm(motion_global_translation[:,joint_idx,:]-motion_global_translation[:,parent_idx,:],dim=1)/ \
                     torch.linalg.norm(standard_skeleton.local_translation[joint_idx,:],dim=0)
            rescaled_motion_global_translation[:,joint_idx,:] = rescaled_motion_global_translation[:,parent_idx,:] + \
                (motion_global_translation[:,joint_idx,:]-motion_global_translation[:,parent_idx,:])/scale.unsqueeze(1).repeat(1,3)
    return rescaled_motion_global_translation

def fix_joints(motion:SkeletonMotion, joint_indices:list):
    motion_length,_,_ = motion.local_rotation.shape
    new_root_translation = motion.root_translation.clone()
    new_local_rotation = motion.local_rotation.clone()
    new_local_rotation[:, joint_indices] = torch.Tensor([[[0,0,0,1]]]).repeat(motion_length,len(joint_indices),1)
    new_state = SkeletonState.from_rotation_and_root_translation(
        motion.skeleton_tree, new_local_rotation, new_root_translation, is_local=True)
    return SkeletonMotion.from_skeleton_state(new_state, motion.fps)

def get_mirror_motion(motion:SkeletonMotion):
    # xoz对称
    mirror_motion_global_rotation = motion.global_rotation.clone()
    mirror_motion_global_rotation[..., 0] = -mirror_motion_global_rotation[..., 0]
    mirror_motion_global_rotation[..., 2] = -mirror_motion_global_rotation[..., 2]
    mirror_indices = [0, 7,8,9,10,11,12, 1,2,3,4,5,6, 13, 23,24,25,26,27,28,29,30,31, 14,15,16,17,18,19,20,21,22, 32]
    mirror_motion_global_rotation = mirror_motion_global_rotation[:,mirror_indices,:]

    mirror_motion_root_translation = motion.root_translation.clone()
    mirror_motion_root_translation[..., 1] = -mirror_motion_root_translation[..., 1]

    mirror_state = SkeletonState.from_rotation_and_root_translation(
        motion.skeleton_tree,
        mirror_motion_global_rotation,
        mirror_motion_root_translation,
        is_local=False
    )
    return SkeletonMotion.from_skeleton_state(mirror_state,motion.fps)


class WeightedFilter:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.previous_data = None
    def filter(self, current_data):
        if self.previous_data is None:
            self.previous_data = current_data
        filtered_data = self.alpha * current_data + (1 - self.alpha) * self.previous_data
        self.previous_data = filtered_data
        return filtered_data


filter_dict = {'Weighted':WeightedFilter}


def filter_data(data,filter_name:str='Weighted',**kwargs):
    filter = filter_dict[filter_name](**kwargs)
    if isinstance(data,np.ndarray):
        return np.stack([filter.filter(d) for d in data])
    else:
        return torch.stack([filter.filter(d) for d in data])


class MotionProcessManager:
    def __init__(self,**kwargs):
        self.operations =  {
        'fix_root': fix_root,
        'move_to_ground': move_feet_on_the_ground,
        'filter': lambda motion: SkeletonMotion.from_skeleton_state(
            SkeletonState.from_rotation_and_root_translation(
                motion.skeleton_tree,
                filter_data(motion.local_rotation),
                motion.global_translation[:, 0, :],
                is_local=True
            ),
            fps=motion.fps
        ),
        'fix_joints': lambda motion: fix_joints(motion, joint_indices=[18, 19, 20, 21, 22, 27, 28, 29, 30, 31, 32]),
        'height_adjustment':lambda motion: height_adjustment(motion,kwargs.get('cal_interval',1.2),kwargs.get('rate',0.2))
    }

    def process_motion(self, motion,**kwargs):
        for key, operation in self.operations.items():
            if kwargs.get(key, False):
                motion = operation(motion).clone()
        return motion

