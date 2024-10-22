from typing import Union
from abc import ABC, abstractmethod
import torch.nn as nn
import torch
from poselib.poselib.core.rotation3d import *
from poselib.poselib.skeleton.skeleton3d import SkeletonMotion,SkeletonState,SkeletonTree


class BaseForwardModel(ABC):
    def __init__(self,skeleton_tree:SkeletonTree,device='cuda:0'):
        self.sk_local_translation = skeleton_tree.local_translation
        self.parent_indices = skeleton_tree.parent_indices
        self.num_joints:int = skeleton_tree.num_joints
        self.device = device
    def forward_kinematics(self,**kwargs):
        return cal_forward_kinematics(**kwargs, sk_parent_indices=self.parent_indices,sk_local_translation=self.sk_local_translation)



def cal_forward_kinematics(motion_local_rotation, motion_root_translation, sk_parent_indices:list, sk_local_translation):
    """
    Args:
        motion_local_rotation (torch.Tensor): (L, J, 4)
        motion_root_translation (torch.Tensor): (L, 3)
        sk_parent_indices (torch.Tensor):  (J)
        sk_local_translation (torch.Tensor): (J, 3)
    Returns:
        tuple:
            - motion_global_rotation (torch.Tensor): (N, J, 4)。
            - motion_global_translation (torch.Tensor): (N, J, 3)。
    """
    motion_global_rotation = []
    motion_global_translation = []
    for joint_idx, parent_idx in enumerate(sk_parent_indices):
        if parent_idx == -1:
            motion_global_rotation.append(motion_local_rotation[:,joint_idx,:])
            motion_global_translation.append(motion_root_translation)
        else:
            motion_global_rotation.append(
                quat_mul_norm(motion_global_rotation[parent_idx],motion_local_rotation[:, joint_idx, :]))
            motion_global_translation.append(quat_rotate(motion_global_rotation[parent_idx], sk_local_translation[joint_idx, :])
                                             + motion_global_translation[parent_idx])

    motion_global_rotation = torch.stack(motion_global_rotation, dim=1)
    motion_global_translation = torch.stack(motion_global_translation, dim=1)
    return motion_global_rotation, motion_global_translation

if __name__ == '__main__':
    import joblib
    import os
    import pickle
    from poselib.poselib.visualization.common import plot_skeleton_H
    motion_path = 'test_data/converted_data/walking1-10_16.pkl'
    with open(motion_path,'rb') as f:
        data = joblib.load(f)

    file_name = os.path.basename(motion_path).split('.')[0]
    data = data[file_name]

    motion_global_rotation = torch.Tensor(data['pose_quat_global'])
    motion_root_translation = torch.Tensor(data['root_trans_offset'])

    with open('asset/smpl/smpl_skeleton_tree.pkl', 'rb') as f:
        skeleton_tree:SkeletonTree = pickle.load(f)

    sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_tree,
                                                     motion_global_rotation,
                                                     motion_root_translation,
                                                     is_local=False)
    sk_motion = SkeletonMotion.from_skeleton_state(sk_state,30)

    base_forward_model = BaseForwardModel(skeleton_tree)
    test_motion_global_rotation, test_motion_global_translation = \
        base_forward_model.forward_kinematics(motion_local_rotation=sk_motion.local_rotation,motion_root_translation=sk_motion.root_translation)

    new_sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_tree,
                                                                    test_motion_global_rotation,
                                                                    motion_root_translation,
                                                                    is_local=False)
    new_sk_motion = SkeletonMotion.from_skeleton_state(new_sk_state, 30)

    print(f'max global rotation error: {(new_sk_motion.global_rotation-sk_motion.global_rotation).max()}')
    print(f'max translation error: {(test_motion_global_translation-sk_motion.global_translation).abs().max()}')

    plot_skeleton_H([sk_motion,new_sk_motion])
