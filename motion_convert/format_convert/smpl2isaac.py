import torch,joblib
import numpy as np

import pickle
from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from scipy.spatial.transform import Rotation as sRot
from body_visualizer.body_visualizer import BodyVisualizer


smpl_joint_names = [
    "Pelvis",
    "L_Hip",
    "R_Hip",
    "Torso",
    "L_Knee",
    "R_Knee",
    "Spine",
    "L_Ankle",
    "R_Ankle",
    "Chest",
    "L_Toe",
    "R_Toe",
    "Neck",
    "L_Thorax",
    "R_Thorax",
    "Head",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
    "L_Hand",
    "R_Hand",
]
isaac_skeleton_tree_names = ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe',
                      'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist',
                      'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']



def convert2isaac(data):
    pose_aa, transl,beta,fps = data['pose_aa'], data['transl'], data['beta'], data['fps']
    pose_aa = pose_aa.reshape(-1,24,3)

    smpl_2_isaac = [smpl_joint_names.index(q) for q in isaac_skeleton_tree_names if q in smpl_joint_names]
    isaac_2_smpl = [isaac_skeleton_tree_names.index(q) for q in smpl_joint_names if q in isaac_skeleton_tree_names]

    pose_aa_isaac = pose_aa[..., smpl_2_isaac, :]

    motion_length = pose_aa.shape[0]

    pose_quat = sRot.from_rotvec(pose_aa_isaac.reshape(-1, 3)).as_quat().reshape(motion_length, 24, 4)
    pose_quat = torch.Tensor(pose_quat)


    with open('asset/smpl/smpl_skeleton_tree.pkl', 'rb') as f:
        skeleton_tree = pickle.load(f)
    root_trans_offset = transl + skeleton_tree.local_translation[0]

    new_sk_state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree,
        # This is the wrong skeleton tree (location wise) here, but it's fine since we only use the parent relationship here.
        pose_quat,
        root_trans_offset,
        is_local=True)

    pose_quat_global = (
            sRot.from_quat(new_sk_state.global_rotation.reshape(-1, 4).numpy()) * sRot.from_quat(
        [0.5, 0.5, 0.5, 0.5]).inv()).as_quat().reshape(motion_length, -1, 4)  # should fix pose_quat as well here...

    new_sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_tree,
                                                                    torch.from_numpy(pose_quat_global),
                                                                    root_trans_offset, is_local=False)

    new_global_rotation = new_sk_state.global_rotation
    new_root_translation = new_sk_state.root_translation
    new_local_rotation = new_sk_state.local_rotation

    rotated_sk_state = SkeletonState.from_rotation_and_root_translation(new_sk_state.skeleton_tree,
                                                                        new_local_rotation,
                                                                        new_root_translation,
                                                                        is_local=True)

    pose_quat = rotated_sk_state.local_rotation
    pose_quat_global = rotated_sk_state.global_rotation
    root_trans_offset = rotated_sk_state.root_translation


    # bd_visualizer = BodyVisualizer("smpl24",static_frame=False)
    # motion_global_translation = rotated_sk_state.global_translation[:,isaac_2_smpl,:]
    # for global_translation in motion_global_translation:
    #     bd_visualizer.step(global_translation)

    new_motion_out = {}
    new_motion_out['pose_quat_global'] = pose_quat_global
    new_motion_out['pose_quat'] = pose_quat
    new_motion_out['trans_orig'] = transl
    new_motion_out['root_trans_offset'] = root_trans_offset
    new_motion_out['beta'] = beta
    new_motion_out['gender'] = "neutral"
    new_motion_out['pose_aa'] = pose_aa
    new_motion_out['fps'] = fps

    return new_motion_out