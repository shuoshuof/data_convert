import math
import numpy as np
import torch
from scipy.interpolate import interp1d

from poselib.poselib.skeleton.skeleton3d import SkeletonMotion,SkeletonState,SkeletonTree
from poselib.poselib.core.rotation3d import *

from motion_convert.robot_config.Hu import Hu_DOF_LOWER,Hu_DOF_UPPER,Hu_DOF_AXIS
from motion_convert.utils.transform3d import quat_in_xyz_axis

def fix_root(motion:SkeletonMotion):
    new_motion_root_translation = motion.root_translation.clone()
    new_motion_global_rotation = motion.global_rotation.clone()

    motion_length,num_joints,_ = motion.global_rotation.shape
    root_quat_x, root_quat_y, root_quat_z = quat_in_xyz_axis(motion.global_root_rotation[0])
    new_motion_global_rotation = quat_mul(quat_inverse(root_quat_z).reshape(1,1,4).repeat(motion_length,num_joints,1),new_motion_global_rotation)
    new_motion_root_translation = quat_rotate(quat_inverse(root_quat_z).reshape(1,4).repeat(motion_length,1),new_motion_root_translation)
    new_motion_root_translation -= motion.root_translation[0]

    new_state = SkeletonState.from_rotation_and_root_translation(
        motion.skeleton_tree, new_motion_global_rotation, new_motion_root_translation, is_local=False)
    return SkeletonMotion.from_skeleton_state(new_state, motion.fps)

def move_feet_on_the_ground(motion:SkeletonMotion):
    min_h = torch.min(motion.global_translation[...,2].clone(),dim=1).values
    new_root_translation = motion.root_translation.clone()
    new_root_translation[:, 2] -= min_h

    new_state = SkeletonState.from_rotation_and_root_translation(
        motion.skeleton_tree, motion.local_rotation, new_root_translation, is_local=True)
    return SkeletonMotion.from_skeleton_state(new_state, motion.fps)

def height_adjustment(motion:SkeletonMotion, cal_interval=1.2, rate = 0.2,deg=3):
    # cal interval is in seconds
    # 消除高度漂移，但不会过滤跳跃,乔丹滞空最多为1.2s，cal_interval以此为极限
    # TODO: 根据motion的方差来设置deg
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

# def fix_ankles(motion:SkeletonMotion):
#     # TODO:不能用global rotation,没有考虑root rotation
#     ankle_indices = [5,11]
#
#     motion_length,_,_ = motion.global_rotation.shape
#     new_root_translation = motion.root_translation.clone()
#     new_global_rotation = motion.global_rotation.clone()
#
#     # motion_root_rotation_yaw = quat_yaw_rotation(motion.global_root_rotation).unsqueeze(1).repeat(1,len(ankle_indices),1)
#     #
#
#
#     quat_x, quat_y, quat_z = quat_in_xyz_axis(motion.global_root_rotation)
#     # quat_x = quat_x.unsqueeze(1).repeat(1,len(ankle_indices),1)
#     # quat_y = quat_y.unsqueeze(1).repeat(1,len(ankle_indices),1)
#     # quat_z = quat_z.unsqueeze(1).repeat(1,len(ankle_indices),1)
#
#     # new_global_rotation[:, ankle_indices] = quat_normalize(quat_mul_three(quat_z.unsqueeze(1),quat_x.unsqueeze(1),
#     #                                                        torch.Tensor([[[0,0,0,1]]]).repeat(motion_length,len(ankle_indices),1)))
#
#     new_global_rotation[:, ankle_indices] = quat_mul_norm(quat_z.unsqueeze(1),torch.Tensor([[[0,0,0,1]]]).repeat(motion_length,len(ankle_indices),1))
#
#     # new_global_rotation[:, ankle_indices] = quat_mul_norm(motion_root_rotation_yaw,torch.Tensor([[[0,0,0,1]]]).repeat(motion_length,len(ankle_indices),1))
#     # new_global_rotation[:, ankle_indices] = \
#     #     quat_mul_norm(motion.global_rotation[:, [0], :],torch.Tensor([[[0,0,0,1]]]).repeat(motion_length,len(ankle_indices),1))
#
#     new_state = SkeletonState.from_rotation_and_root_translation(
#         motion.skeleton_tree, new_global_rotation, new_root_translation, is_local=False)
#     # quat_from_xyz(quat_to_eular(motion.global_root_rotation))
#     # motion_length,_,_ = motion.local_rotation.shape
#     # new_root_translation = motion.root_translation.clone()
#     # new_local_rotation = motion.local_rotation.clone()
#     # new_local_rotation[:, ankle_indices] = torch.Tensor([[[0,0,0,1]]]).repeat(motion_length,len(ankle_indices),1)
#     # new_state = SkeletonState.from_rotation_and_root_translation(
#     #     motion.skeleton_tree, new_local_rotation, new_root_translation, is_local=True)
#
#     return SkeletonMotion.from_skeleton_state(new_state, motion.fps)


def fix_ankles(motion: SkeletonMotion):
    # TODO:不能用global rotation,没有考虑root rotation

    motion_length, _, _ = motion.global_rotation.shape
    new_root_translation = motion.root_translation.clone()
    new_global_rotation = motion.global_rotation.clone()

    # left_hip_yaw_quat = motion.global_rotation[:,1,:]
    # right_hip_yaw_quat = motion.global_rotation[:,7,:]
    #
    # left_hip_roll_quat = motion.global_rotation[:,2,:]
    # right_hip_roll_quat = motion.global_rotation[:,8,:]

    left_knee_quat = motion.global_rotation[:,4,:]
    right_knee_quat = motion.global_rotation[:,10,:]

    left_quat_x,left_quat_y,left_quat_z = quat_in_xyz_axis(left_knee_quat)
    right_quat_x,right_quat_y,right_quat_z = quat_in_xyz_axis(right_knee_quat)

    new_global_rotation[:,5,:] = quat_normalize(
        quat_mul_three(left_quat_z,left_quat_x,torch.Tensor([[0,0,0,1]]).repeat(motion_length,1)))
    new_global_rotation[:,11,:] = quat_normalize(
        quat_mul_three(right_quat_z,right_quat_x,torch.Tensor([[0,0,0,1]]).repeat(motion_length,1)))

    left_ankle_quat = new_global_rotation[:,5,:].clone()
    right_ankle_quat = new_global_rotation[:,11,:].clone()

    left_ankle_quat_x,left_ankle_quat_y,left_ankle_quat_z = quat_in_xyz_axis(left_ankle_quat)
    right_ankle_quat_x,right_ankle_quat_y,right_ankle_quat_z = quat_in_xyz_axis(right_ankle_quat)

    new_global_rotation[:,6,:] = quat_normalize(
        quat_mul_three(left_ankle_quat_z,left_ankle_quat_y,torch.Tensor([[0,0,0,1]]).repeat(motion_length,1)))
    new_global_rotation[:,12,:] = quat_normalize(
        quat_mul_three(right_ankle_quat_z,right_ankle_quat_y,torch.Tensor([[0,0,0,1]]).repeat(motion_length,1)))

    # new_global_rotation[:,5,:] = quat_mul_norm(left_hip_yaw_quat,torch.Tensor([[0,0,0,1]]).repeat(motion_length,1))
    # new_global_rotation[:,11,:] = quat_mul_norm(right_hip_yaw_quat,torch.Tensor([0,0,0,1]).repeat(motion_length,1))
    #
    # new_global_rotation[:,6,:] = quat_mul_norm(left_hip_roll_quat,torch.Tensor([[0,0,0,1]]).repeat(motion_length,1))
    # new_global_rotation[:,12,:] = quat_mul_norm(right_hip_roll_quat,torch.Tensor([[0,0,0,1]]).repeat(motion_length,1))



    new_state = SkeletonState.from_rotation_and_root_translation(
        motion.skeleton_tree, new_global_rotation, new_root_translation, is_local=False)

    return SkeletonMotion.from_skeleton_state(new_state, motion.fps)

# def check_limit(motion:SkeletonMotion):
#     # TODO: 0度时轴是不对的
#     motion_length,num_joints,_ = motion.local_rotation.shape
#
#     motion_root_rotation = motion.local_rotation[:,0,:]
#     motion_joint_angles,_ = quat_to_angle_axis(motion.local_rotation[:,1:,:])
#
#     hu_dof_low_limit = Hu_DOF_LOWER.reshape(1, -1, 1)
#     hu_dof_high_limit = Hu_DOF_UPPER.reshape(1, -1, 1)
#
#     clamped_motion_joint_angles = torch.clamp(motion_joint_angles.unsqueeze(-1).clone(), min=hu_dof_low_limit, max=hu_dof_high_limit)
#
#     motion_rotation_axis = torch.eye(3)[Hu_DOF_AXIS].repeat(motion_length,1,1)
#
#     clamped_motion_local_rotation =quat_from_angle_axis(clamped_motion_joint_angles.reshape(-1),motion_rotation_axis.reshape(-1,3))
#     clamped_motion_local_rotation = clamped_motion_local_rotation.reshape(motion_length, num_joints - 1, 4)
#
#     new_motion_local_rotation = torch.concatenate([motion_root_rotation.unsqueeze(1), clamped_motion_local_rotation], dim=1)
#
#     clamped_state = SkeletonState.from_rotation_and_root_translation(
#         motion.skeleton_tree,
#         new_motion_local_rotation,
#         motion_root_rotation,
#         is_local=True
#     )
#     return SkeletonMotion.from_skeleton_state(clamped_state,motion.fps)


def check_limit(motion:SkeletonMotion):

    motion_length,num_joints,_ = motion.local_rotation.shape

    motion_root_rotation = motion.local_rotation[:,0,:]
    motion_joints_exp_map = quat_to_exp_map(motion.local_rotation[:,1:,:])

    hu_dof_low_limit = Hu_DOF_LOWER.reshape(1, -1, 1)
    hu_dof_high_limit = Hu_DOF_UPPER.reshape(1, -1, 1)

    clamped_exp_map = torch.clamp(motion_joints_exp_map,min=hu_dof_low_limit,max=hu_dof_high_limit)

    motion_rotation_axis = torch.eye(3)[Hu_DOF_AXIS].repeat(motion_length, 1, 1)

    clamped_exp_map = clamped_exp_map*motion_rotation_axis

    clamped_motion_local_rotation = exp_map_to_quat(clamped_exp_map)


    new_motion_local_rotation = torch.concatenate([motion_root_rotation.unsqueeze(1), clamped_motion_local_rotation], dim=1)

    clamped_state = SkeletonState.from_rotation_and_root_translation(
        motion.skeleton_tree,
        new_motion_local_rotation,
        motion.root_translation,
        is_local=True
    )
    return SkeletonMotion.from_skeleton_state(clamped_state,motion.fps)

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
        # height_adjustment and move_to_ground should in the last two
        self.operations =  {
        'fix_root': fix_root,
        'filter': lambda motion: SkeletonMotion.from_skeleton_state(
            SkeletonState.from_rotation_and_root_translation(
                motion.skeleton_tree,
                filter_data(motion.local_rotation),
                motion.global_translation[:, 0, :],
                is_local=True
            ),
            fps=motion.fps
        ),
        'fix_joints': lambda motion: fix_joints(motion, joint_indices=kwargs.get('joint_indices',[18, 19, 20, 21, 22, 27, 28, 29, 30, 31, 32])),
        'fix_ankles': fix_ankles,
        'height_adjustment':lambda motion: height_adjustment(motion,kwargs.get('cal_interval',1.2),kwargs.get('rate',0.2)),
        'move_to_ground': move_feet_on_the_ground,
    }

    def process_motion(self, motion,**kwargs):
        for key, operation in self.operations.items():
            if kwargs.get(key, False):
                motion = operation(motion).clone()
        motion = check_limit(motion)
        return motion

