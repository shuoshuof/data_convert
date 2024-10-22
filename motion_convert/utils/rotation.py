import torch
from poselib.poselib.core.rotation3d import quat_normalize
@torch.jit.script
def quat_between_two_vecs(vec1, vec2):
    if torch.norm(vec1,dim=-1).max() <= 1e-6 or torch.norm(vec2,dim=-1).max() <= 1e-6:
        return torch.tensor([[0, 0, 0, 1]]*vec1.shape[0], dtype=torch.float32)

    vec1 = vec1 / torch.linalg.norm(vec1, dim=-1, keepdim=True)
    vec2 = vec2 / torch.linalg.norm(vec2, dim=-1, keepdim=True)
    cross_prod = torch.cross(vec1, vec2, dim=-1)
    dots = torch.sum(vec1 * vec2, dim=-1, keepdim=True)
    real_part = (1 + dots)  # Adding 1 to ensure the angle calculation is stable
    quat = torch.cat([cross_prod, real_part], dim=-1)
    # quat = quat / torch.norm(quat, dim=-1, keepdim=True)  # Normalize quaternion to ensure unit quaternion
    quat = quat_normalize(quat)

    return quat



