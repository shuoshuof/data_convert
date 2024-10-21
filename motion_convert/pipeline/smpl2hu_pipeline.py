import os
from tqdm import tqdm
import copy
from abc import ABC, abstractmethod
from typing import Optional, Union
import joblib
import pickle
import math
import torch
import numpy as np
from scipy.spatial.transform import Rotation as sRot

from poselib.poselib.core.rotation3d import quat_mul,quat_rotate
from poselib.poselib.skeleton.skeleton3d import SkeletonTree,SkeletonMotion,SkeletonState
from poselib.poselib.visualization.common import *

from body_visualizer.body_visualizer import BodyVisualizer
from motion_convert.pipeline.base_pipeline import BasePipeline
from motion_convert.format_convert.smpl2isaac import convert2isaac
from motion_convert.utils.rotation import *

JOINT_MAPPING_v1 = {
    'Pelvis': 'pelvis_link',
    'L_Hip': 'left_hip_pitch_link',
    'L_Knee': 'left_knee_link',
    'L_Ankle': 'left_ankle_link',
    'R_Hip': 'right_hip_pitch_link',
    'R_Knee': 'right_knee_link',
    'R_Ankle': 'right_ankle_link',
    # 'Torso': 'torso_link',
    # # 'Spine': 'torso_link',
    'Chest': 'torso_link',
    'Head': 'neck_link',
    'L_Shoulder': 'left_shoulder_roll_link',
    'L_Elbow': 'left_elbow_pitch_link',
    'L_Wrist': 'left_wrist_yaw_link',
    'R_Shoulder': 'right_shoulder_roll_link',
    'R_Elbow': 'right_elbow_pitch_link',
    'R_Wrist': 'right_wrist_yaw_link',
}

class SMPL2HuPipeline(BasePipeline):
    def __init__(self, motion_dir: str, save_dir: str, num_processes: int = None):
        super().__init__(motion_dir, save_dir, num_processes)
        self.smpl_t_pose,self.hu_t_pose,self.smpl_skeleton_tree,self.hu_skeleton_tree = self._load_asset()
        
    def _read_data(self, **kwargs) -> Optional[list]:
        motion_paths = [os.path.join(self.motion_dir, f) for f in os.listdir(self.motion_dir) if
                              os.path.isfile(os.path.join(self.motion_dir, f))]
        return motion_paths
    def _split_data(self,data,**kwargs)->Optional[list]:
        return np.array_split(data,self.num_processes)
    def _load_asset(self)->[SkeletonState, SkeletonState,SkeletonTree, SkeletonTree]:
        with open('asset/t_pose/smpl_t_pose.pkl','rb') as f:
            smpl_t_pose = pickle.load(f)
        with open('asset/t_pose/hu_t_pose.pkl','rb') as f:
            hu_t_pose = pickle.load(f)
        with open('asset/smpl/smpl_skeleton_tree.pkl','rb') as f:
            smpl_skeleton_tree = pickle.load(f)
        hu_skeleton_tree = hu_t_pose.skeleton_tree
        return smpl_t_pose,hu_t_pose,smpl_skeleton_tree,hu_skeleton_tree
    def _rebuild_with_smpl_t_pose(self,motion:SkeletonMotion):
        motion_fps = motion.fps

        motion_global_translation = motion.global_translation
        motion_length, num_keypoint, _ = motion_global_translation.shape
        new_motion_global_rotation = motion.global_rotation
        new_motion_root_translation = motion.root_translation

        t_pose_local_translation = self.smpl_t_pose.local_translation.repeat(motion_length,1,1)
        parent_indices = self.smpl_t_pose.skeleton_tree.parent_indices

        rebuild_indices = [16,17,18,21,22,23]

        for joint_index in rebuild_indices:
            new_motion_global_rotation[:,parent_indices[joint_index],:] = \
                quat_between_two_vecs(t_pose_local_translation[:,joint_index],motion_global_translation[:,joint_index]-motion_global_translation[:,parent_indices[joint_index]])

        new_skeleton_state = SkeletonState.from_rotation_and_root_translation(
            self.smpl_t_pose.skeleton_tree,
            new_motion_global_rotation,
            new_motion_root_translation,
            is_local=False
        )
        new_motion = SkeletonMotion.from_skeleton_state(new_skeleton_state,fps=motion_fps)

        rebuild_error = torch.abs(new_motion.global_translation-motion_global_translation).max()
        print(f"Rebuild error :{rebuild_error:.5f}")

        return new_motion


    def _process_data(self,data_chunk,results,process_idx,**kwargs):
        for path in data_chunk:
            with open(path,'rb') as f:
                motion_data = joblib.load(f)
            motion_key = os.path.basename(path).split('.')[0]
            motion = motion_data[motion_key]
            motion_fps = motion['fps']

            motion_global_rotation = motion['pose_quat_global']
            motion_root_translation = motion['root_trans_offset']

            smpl_motion = SkeletonState.from_rotation_and_root_translation(
                self.smpl_skeleton_tree,
                motion_global_rotation,
                motion_root_translation,
                is_local=False
            )

            # plot_skeleton_H([smpl_motion.zero_pose(self.smpl_skeleton_tree)])

            smpl_motion = SkeletonMotion.from_skeleton_state(smpl_motion,fps=motion_fps)
            # TODO： 腰部旋转很可能有问题
            rebuilt_motion = self._rebuild_with_smpl_t_pose(smpl_motion)

            target_motion = copy.deepcopy(rebuilt_motion).retarget_to_by_tpose(
                joint_mapping=JOINT_MAPPING_v1,
                source_tpose=self.smpl_t_pose,
                target_tpose=self.hu_t_pose,
                rotation_to_target_skeleton=torch.Tensor([0,0,0,1]),
                scale_to_target_skeleton=1.,
            )



            plot_skeleton_H([smpl_motion,rebuilt_motion,target_motion])




if __name__ == '__main__':

    smpl2hu_pipeline = SMPL2HuPipeline(motion_dir='test_data/converted_data',
                                      save_dir='test_data/hu_data')
    smpl2hu_pipeline.run(debug=True)