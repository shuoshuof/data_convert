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

def cal_ground_height(motion:SkeletonMotion, cal_interval=2, rate = 0.2):
    # cal interval is in seconds

    motion_min_height = torch.min(motion.global_translation[:,:,2])
    motion_length,_ = motion.root_translation.shape
    fps = motion.fps
    win_length = cal_interval * fps

    new_motion_root_translation = motion.root_translation.clone()

    ground_height = []
    for motion_idx in range(0, len(motion.local_rotation), win_length):
        motion_slice = motion.local_rotation[motion_idx:motion_idx+win_length]
        min_z = torch.min(motion_slice[:,:,2],dim=1).values
        sorted_z, indices = torch.sort(min_z)
        ground_height.append([motion_idx,sorted_z[:int(rate*win_length)].mean()])

    indices = torch.Tensor(ground_height[:][0]).numpy()
    height = torch.Tensor(ground_height[:][1]).numpy()

    motion_ground_height = np.arange(0,motion_length)
    interp_func = interp1d(indices,height,fill_value="extrapolate")
    interpolated_heights = torch.Tensor(interp_func(motion_ground_height))




# @torch.jit.script
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
    return [filter.filter(d) for d in data]


class MotionProcessManager:
    def __init__(self,):
        self.operations =  {
        'fix_root': fix_root,
        'move_to_ground': move_feet_on_the_ground,
        'filter': lambda motion: SkeletonMotion.from_skeleton_state(
            SkeletonState.from_rotation_and_root_translation(
                motion.skeleton_tree,
                torch.stack(filter_data(motion.local_rotation)),
                motion.global_translation[:, 0, :],
                is_local=True
            ),
            fps=motion.fps
        ),
        'fix_joints': lambda motion: fix_joints(motion, joint_indices=[18, 19, 20, 21, 22, 27, 28, 29, 30, 31, 32])
    }

    def process_motion(self, motion,**kwargs):
        for key, operation in self.operations.items():
            if kwargs.get(key, False):
                motion = operation(motion).clone()
        return motion

