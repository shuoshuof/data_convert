import numpy as np
import pickle
import copy
from collections import OrderedDict

import torch

from poselib.poselib.skeleton.skeleton3d import SkeletonTree,SkeletonMotion,SkeletonState
from poselib.poselib.visualization.common import plot_skeleton_H
from poselib.poselib.core.rotation3d import *

from body_visualizer.visualizer import BodyVisualizer

from motion_convert.robot_config.Hu import hu_graph,HU_JOINT_NAMES
from motion_convert.utils.transform3d import coord_transform



if __name__ == '__main__':
    with open('asset/zero_pose/hu_zero_pose.pkl','rb') as f:
        hu_zero_pose:SkeletonState = pickle.load(f)

    left_hip_pitch_angle_axis = torch.tensor([0,1,0,-0.4])
    right_hip_pitch_angle_axis = torch.tensor([0,1,0,-0.4])

    left_knee_angle_axis = torch.tensor([0,1,0,0.8])
    right_knee_angle_axis = torch.tensor([0,1,0,0.8])

    left_hip_pitch_quat = quat_from_angle_axis(left_hip_pitch_angle_axis[-1],left_hip_pitch_angle_axis[:3])
    right_hip_pitch_quat = quat_from_angle_axis(right_hip_pitch_angle_axis[-1],right_hip_pitch_angle_axis[:3])

    left_knee_quat = quat_from_angle_axis(left_knee_angle_axis[-1],left_knee_angle_axis[:3])
    right_knee_quat = quat_from_angle_axis(right_knee_angle_axis[-1],right_knee_angle_axis[:3])

    hu_zero_pose.local_rotation[[3,4,9,10]] = torch.concatenate([left_hip_pitch_quat[None,:],left_knee_quat[None,:],right_hip_pitch_quat[None,:],right_knee_quat[None,:]],dim=0)

    hu_start_pose = SkeletonState.from_rotation_and_root_translation(
        hu_zero_pose.skeleton_tree,
        hu_zero_pose.local_rotation,
        hu_zero_pose.root_translation,
        is_local=True
    )

    with open('asset/start_pose/hu_start_pose.pkl','wb') as f:
        pickle.dump(hu_start_pose,f)

    plot_skeleton_H([hu_start_pose])



