import math
import numpy as np
import torch
import pickle
from collections import OrderedDict
from typing import Union

from scipy.interpolate import interp1d
from scipy.interpolate import splrep

from poselib.poselib.skeleton.skeleton3d import SkeletonMotion,SkeletonState,SkeletonTree
from poselib.poselib.core.rotation3d import *

from motion_convert.robot_config.Hu import Hu_DOF_LOWER,Hu_DOF_UPPER,Hu_DOF_AXIS
from motion_convert.utils.transform3d import quat_in_xyz_axis,quat_slerp

def zero_root(motion:SkeletonMotion,adjust_all_axis=False):
    new_motion_root_translation = motion.root_translation.clone()
    new_motion_global_rotation = motion.global_rotation.clone()

    motion_length,num_joints,_ = motion.global_rotation.shape
    root_quat_x, root_quat_y, root_quat_z = quat_in_xyz_axis(motion.global_root_rotation[0])
    if adjust_all_axis:
        adjust_quat = quat_inverse(motion.global_root_rotation[0])
    else:
        adjust_quat = quat_inverse(root_quat_z)
    new_motion_global_rotation = quat_mul(adjust_quat.reshape(1,1,4).repeat(motion_length,num_joints,1),new_motion_global_rotation)
    new_motion_root_translation = quat_rotate(adjust_quat.reshape(1,4).repeat(motion_length,1),new_motion_root_translation)

    new_motion_root_translation -= new_motion_root_translation[0].clone()

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


def add_zero_pose_head(motion: SkeletonMotion, slerp_frame: int = 60, head_frame:int=30):
    # hu_start_pose = SkeletonState.zero_pose(skeleton_tree=motion.skeleton_tree)
    with open('asset/start_pose/hu_start_pose.pkl','rb') as f:
        hu_start_pose:SkeletonState = pickle.load(f)

    # the knee may out of range
    hu_start_pose = clip_zero_pose(hu_start_pose)
    motion_start_local_rotation = motion.local_rotation[0, :, :]

    # slerp local rotation
    slerp_motion_local_rotation = []
    for i in range(slerp_frame):
        t = torch.tensor(i / slerp_frame)
        slerp_motion_local_rotation.append(quat_slerp(hu_start_pose.local_rotation, motion_start_local_rotation, t))
    slerp_motion_local_rotation = torch.stack(slerp_motion_local_rotation, dim=0)
    slerped_rotation_state = SkeletonState.from_rotation_and_root_translation(
        motion.skeleton_tree,
        slerp_motion_local_rotation,
        torch.zeros((slerp_frame, 3)),
        is_local=True
    )

    # cal root translation
    min_h = torch.min(slerped_rotation_state.global_translation[..., 2].clone(), dim=1).values
    new_root_translation = slerped_rotation_state.root_translation.clone()
    new_root_translation[:, 2] -= min_h

    slerped_motion_local_rotation = torch.concatenate(
        [slerped_rotation_state.local_rotation, motion.local_rotation.clone()], dim=0)
    slerped_motion_root_translation = torch.concatenate([new_root_translation, motion.root_translation.clone()], dim=0)

    start_pose_head_local_rotation = hu_start_pose.local_rotation.clone()[None,...].repeat(head_frame,1,1)
    start_pose_head_root_translation = new_root_translation.clone()[[0]].repeat(head_frame,1)

    result_motion_local_rotation = torch.concatenate(
        [start_pose_head_local_rotation, slerped_motion_local_rotation], dim=0)
    result_motion_root_translation = torch.concatenate(
        [start_pose_head_root_translation, slerped_motion_root_translation], dim=0)

    result_state = SkeletonState.from_rotation_and_root_translation(
        motion.skeleton_tree,
        result_motion_local_rotation,
        result_motion_root_translation,
        is_local=True
    )

    return SkeletonMotion.from_skeleton_state(result_state, motion.fps)


def flatten_feet(motion: SkeletonMotion):
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


def clip_zero_pose(zero_pose:SkeletonState):

    num_joints,_ = zero_pose.local_rotation.shape

    zero_pose_root_rotation = zero_pose.local_rotation[0,:]
    zero_pose_joints_exp_map = quat_to_exp_map(zero_pose.local_rotation[1:,:])

    hu_dof_low_limit = Hu_DOF_LOWER.reshape(-1, 1)
    hu_dof_high_limit = Hu_DOF_UPPER.reshape(-1, 1)

    clamped_exp_map = torch.clamp(zero_pose_joints_exp_map,min=hu_dof_low_limit,max=hu_dof_high_limit)

    motion_rotation_axis = torch.eye(3)[Hu_DOF_AXIS]

    clamped_exp_map = clamped_exp_map*motion_rotation_axis

    clamped_motion_local_rotation = exp_map_to_quat(clamped_exp_map)


    new_motion_local_rotation = torch.concatenate([zero_pose_root_rotation.unsqueeze(0), clamped_motion_local_rotation], dim=0)

    clamped_state = SkeletonState.from_rotation_and_root_translation(
        zero_pose.skeleton_tree,
        new_motion_local_rotation,
        zero_pose.root_translation,
        is_local=True
    )

    return clamped_state

def clip_dof_pos(motion:Union[SkeletonMotion,SkeletonState]):

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
    if isinstance(motion,SkeletonMotion):
        return SkeletonMotion.from_skeleton_state(clamped_state,motion.fps)

    return clamped_state

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

def motion_concatenate(motions:List[Union[SkeletonMotion,SkeletonState]]):
    r"""
    all the motions should have the same skeleton tree
    :param motions:
    :return:
    """
    motions_local_rotation = []
    motions_root_translation = []
    for motion in motions:
        motions_local_rotation.append(motion.local_rotation.clone())
        motions_root_translation.append(motion.root_translation.clone())

    new_sk_state = SkeletonState.from_rotation_and_root_translation(
        motions[0].skeleton_tree,
        torch.cat(motions_local_rotation,dim=0),
        torch.cat(motions_root_translation,dim=0),
        is_local=True
    )
    return SkeletonMotion.from_skeleton_state(new_sk_state,motions[0].fps)

def hu_zero_motion():
    with open('asset/hu_pose/hu_v5_zero_pose.pkl', 'rb') as f:
        hu_v5_zero_pose:SkeletonState = pickle.load(f)

    new_local_rotation = hu_v5_zero_pose.local_rotation.unsqueeze(0).repeat(100,1,1)
    new_root_translation = hu_v5_zero_pose.root_translation.unsqueeze(0).repeat(100,1)

    new_sk_state = SkeletonState.from_rotation_and_root_translation(
        hu_v5_zero_pose.skeleton_tree,
        new_local_rotation,
        new_root_translation,
        is_local=True
    )
    return new_sk_state

# def left_to_rigth_euler(pose_euler):
#     pose_euler[:, :, 0] = pose_euler[:, :, 0] * -1
#     pose_euler[:, :, 2] = pose_euler[:, :, 2] * -1
#     pose_euler = pose_euler[:, left_right_idx, :]
#     return pose_euler
#
#
# def flip_smpl(pose, trans=None):
#     """
#     Pose input batch * 72
#     """
#     curr_spose = sRot.from_rotvec(pose.reshape(-1, 3))
#     curr_spose_euler = curr_spose.as_euler("ZXY", degrees=False).reshape(pose.shape[0], 24, 3)
#     curr_spose_euler = left_to_rigth_euler(curr_spose_euler)
#     curr_spose_rot = sRot.from_euler("ZXY", curr_spose_euler.reshape(-1, 3), degrees=False)
#     curr_spose_aa = curr_spose_rot.as_rotvec().reshape(pose.shape[0], 24, 3)
#     if trans != None:
#         pass
#         # target_root_mat = curr_spose.as_matrix().reshape(pose.shape[0], 24, 3, 3)[:, 0]
#         # root_mat = curr_spose_rot.as_matrix().reshape(pose.shape[0], 24, 3, 3)[:, 0]
#         # apply_mat = np.matmul(target_root_mat[0], np.linalg.inv(root_mat[0]))
#
#     return curr_spose_aa.reshape(-1, 72)

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
        self.operations =  OrderedDict({
        'zero_root': lambda motion: zero_root(motion,adjust_all_axis=kwargs.get('adjust_all_axis',False)),
        'add_zero_pose_head': lambda motion: add_zero_pose_head(motion,slerp_frame=kwargs.get('slerp_frame',60)),
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
        'flatten_feet': flatten_feet,
        'height_adjustment':lambda motion: height_adjustment(motion,kwargs.get('cal_interval',1.2),kwargs.get('rate',0.2)),
        'move_to_ground': move_feet_on_the_ground,
    })

    def process_motion(self, motion,**kwargs):
        for key, operation in self.operations.items():
            if kwargs.get(key, False):
                motion = operation(motion).clone()
        motion = clip_dof_pos(motion)
        return motion

