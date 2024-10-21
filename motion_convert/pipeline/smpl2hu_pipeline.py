import os
from tqdm import tqdm
import math
import torch
from abc import ABC, abstractmethod
from typing import Optional, Union
import joblib
import pickle
import numpy as np
from scipy.spatial.transform import Rotation as sRot

from poselib.poselib.core.rotation3d import quat_mul,quat_rotate
from poselib.poselib.skeleton.skeleton3d import SkeletonTree,SkeletonMotion,SkeletonState
from poselib.poselib.visualization.common import *

from body_visualizer.body_visualizer import BodyVisualizer
from motion_convert.pipeline.base_pipeline import BasePipeline
from motion_convert.format_convert.smpl2isaac import convert2isaac

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
        self.smpl_t_pose,self.hu_t_pose = self._load_t_pose()
        self.smpl_skeleton_tree = self.smpl_t_pose.skeleton_tree
        self.hu_skeleton_tree = self.hu_t_pose.skeleton_tree
        body_visualizer = BodyVisualizer('smpl24')
        body_visualizer.step(self.smpl_t_pose.global_translation)
    def _read_data(self, **kwargs) -> Optional[list]:
        motion_paths = [os.path.join(self.motion_dir, f) for f in os.listdir(self.motion_dir) if
                              os.path.isfile(os.path.join(self.motion_dir, f))]
        return motion_paths
    def _split_data(self,data,**kwargs)->Optional[list]:
        return np.array_split(data,self.num_processes)
    def _load_t_pose(self)->[SkeletonState,SkeletonState]:
        with open('asset/t_pose/smpl_t_pose.pkl','rb') as f:
            smpl_t_pose = pickle.load(f)
        with open('asset/t_pose/hu_t_pose.pkl','rb') as f:
            hu_t_pose = pickle.load(f)
        return smpl_t_pose,hu_t_pose
    def _rebuild_with_smpl_t_pose(self,motion:SkeletonMotion):
        motion_fps = motion.fps
        motion_global_translation = motion.global_translation
        motion_local_translation = motion.local_translation
        motion_length, num_keypoint, _ = motion_global_translation.shape

        new_motion_global_rotation = motion.global_rotation
        new_motion_root_translation = motion.root_translation
        new_motion_parent_indices = self.smpl_t_pose.skeleton_tree.parent_indices

        rebuild_indices = [16,17,18,21,22,23]

        for joint_index in rebuild_indices:
            new_motion_global_rotation[:,joint_index,:] = 0

    def cal_rotation(self,vec1,vec2):

        pass


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
            smpl_motion = SkeletonMotion.from_skeleton_state(smpl_motion,fps=motion_fps)

            plot_skeleton_H([smpl_motion])



if __name__ == '__main__':

    smpl2hu_pipeline = SMPL2HuPipeline(motion_dir='test_data/converted_data',
                                      save_dir='test_data/hu_data')
    smpl2hu_pipeline.run(debug=True)