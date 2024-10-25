import numpy as np
import torch
from poselib.poselib.core.rotation3d import *

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


def coord_transform(p,order:list,dir=None):
    p = p[...,order]
    if dir is not None:
        p = p* dir
    return p

def cal_joint_quat(standard_pose_local_translation, motion_local_translation):
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



