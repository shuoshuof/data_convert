import numpy as np
import torch
from poselib.poselib.core.rotation3d import *
from poselib.poselib.skeleton.skeleton3d import SkeletonMotion,SkeletonState,SkeletonTree
from typing import Dict
import copy
@torch.jit.script
def quat_between_two_vecs(vec1, vec2):
    '''calculate a quaternion that rotates from vector v1 to vector v2'''
    if torch.norm(vec1,dim=-1).max() <= 1e-6 or torch.norm(vec2,dim=-1).max() <= 1e-6:
        return torch.tensor([[0, 0, 0, 1]]*vec1.shape[0], dtype=torch.float32)

    vec1 = vec1 / torch.linalg.norm(vec1, dim=-1, keepdim=True)
    vec2 = vec2 / torch.linalg.norm(vec2, dim=-1, keepdim=True)
    cross_prod = torch.cross(vec1, vec2, dim=-1)
    dots = torch.sum(vec1 * vec2, dim=-1, keepdim=True)
    real_part = (1 + dots)  # Adding 1 to ensure the angle calculation is stable
    quat = torch.cat([cross_prod, real_part], dim=-1)
    quat = quat_normalize(quat)
    return quat


def coord_transform(p,order:list=None,dir=None):
    if order is not None:
        p = p[...,order]
    if dir is not None:
        p = p* dir
    return p
@torch.jit.script
def cal_joint_quat(standard_pose_local_translation, motion_local_translation):
    # https://igl.ethz.ch/projects/ARAP/svd_rot.pdf
    A = torch.einsum('bij,bjk->bik', motion_local_translation.permute(0, 2, 1), standard_pose_local_translation)
    U, _, Vt = torch.linalg.svd(A)
    R_matrix = torch.einsum('bij,bjk->bik', U, Vt)

    det = torch.linalg.det(R_matrix)
    Vt[det < 0, -1, :] *= -1
    R_matrix = torch.einsum('bij,bjk->bik', U, Vt)

    # rotation = sRot.from_matrix(R_matrix)
    # quats = rotation.as_quat()
    quats = quat_from_rotation_matrix(R_matrix)
    return quats

def retarget_to_by_tpose(
        skeleton_state,
        joint_mapping: Dict[str, str],
        source_tpose: "SkeletonState",
        target_tpose: "SkeletonState",
        rotation_to_target_skeleton,
        scale_to_target_skeleton: float,
) -> "SkeletonState":
    return retarget_to(
        skeleton_state,
        joint_mapping,
        source_tpose.local_rotation,
        source_tpose.root_translation,
        target_tpose.skeleton_tree,
        target_tpose.local_rotation,
        target_tpose.root_translation,
        rotation_to_target_skeleton,
        scale_to_target_skeleton,
    )


def retarget_to(
        skeleton_state: SkeletonState,
        joint_mapping: Dict[str, str],
        source_tpose_local_rotation,
        source_tpose_root_translation: np.ndarray,
        target_skeleton_tree: SkeletonTree,
        target_tpose_local_rotation,
        target_tpose_root_translation: np.ndarray,
        rotation_to_target_skeleton,
        scale_to_target_skeleton: float,
        z_up: bool = True,
) -> "SkeletonState":
    # STEP 0: Preprocess
    source_tpose = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree=copy.deepcopy(skeleton_state.skeleton_tree),
        r=source_tpose_local_rotation,
        t=source_tpose_root_translation,
        is_local=True,
    )

    target_tpose = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree=copy.deepcopy(target_skeleton_tree),
        r=target_tpose_local_rotation,
        t=target_tpose_root_translation,
        is_local=True,
    )

    # STEP 1: Drop the irrelevant joints
    pairwise_translation = skeleton_state._get_pairwise_average_translation()
    node_names = list(joint_mapping)
    new_skeleton_tree = skeleton_state.skeleton_tree.keep_nodes_by_names(node_names, pairwise_translation)

    # TODO: combine the following steps before STEP 3
    source_tpose = source_tpose._transfer_to(new_skeleton_tree)
    source_state = skeleton_state._transfer_to(new_skeleton_tree)

    source_tpose = source_tpose._remapped_to(joint_mapping, copy.deepcopy(target_skeleton_tree))
    source_state = source_state._remapped_to(joint_mapping, copy.deepcopy(target_skeleton_tree))

    # STEP 2: Rotate the source to align with the target
    new_local_rotation = source_tpose.local_rotation.clone()
    new_local_rotation[..., 0, :] = quat_mul_norm(rotation_to_target_skeleton, source_tpose.local_rotation[..., 0, :])

    source_tpose = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree=source_tpose.skeleton_tree,
        r=new_local_rotation,
        t=quat_rotate(rotation_to_target_skeleton, source_tpose.root_translation),
        is_local=True,
    )

    new_local_rotation = source_state.local_rotation.clone()
    new_local_rotation[..., 0, :] = quat_mul_norm(rotation_to_target_skeleton, source_state.local_rotation[..., 0, :])
    source_state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree=source_state.skeleton_tree,
        r=new_local_rotation,
        t=quat_rotate(rotation_to_target_skeleton, source_state.root_translation),
        is_local=True,
    )

    # STEP 3: Normalize to match the target scale
    root_translation_diff = (source_state.root_translation - source_tpose.root_translation) * scale_to_target_skeleton

    # STEP 4: the global rotation from source state relative to source tpose and
    # re-apply to the target
    current_skeleton_tree = source_state.skeleton_tree
    target_tpose_global_rotation = source_state.global_rotation[0, :].clone()
    for current_index, name in enumerate(current_skeleton_tree):
        if name in target_tpose.skeleton_tree:
            target_tpose_global_rotation[current_index, :] = target_tpose.global_rotation[
                                                             target_tpose.skeleton_tree.index(name), :]

    global_rotation_diff = quat_mul_norm(source_state.global_rotation, quat_inverse(source_tpose.global_rotation))
    new_global_rotation = quat_mul_norm(global_rotation_diff, target_tpose_global_rotation)

    # STEP 5: Putting 3 and 4 together
    current_skeleton_tree = source_state.skeleton_tree
    shape = source_state.global_rotation.shape[:-1]
    shape = shape[:-1] + target_tpose.global_rotation.shape[-2:-1]
    new_global_rotation_output = quat_identity(shape)
    for current_index, name in enumerate(target_skeleton_tree):
        while name not in current_skeleton_tree:
            name = target_skeleton_tree.parent_of(name)
        parent_index = current_skeleton_tree.index(name)
        new_global_rotation_output[:, current_index, :] = new_global_rotation[:, parent_index, :]

    source_state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree=target_skeleton_tree,
        r=new_global_rotation_output,
        t=target_tpose.root_translation + root_translation_diff,
        is_local=False,
    ).local_repr()

    return source_state


