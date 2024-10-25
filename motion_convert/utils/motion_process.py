from poselib.poselib.skeleton.skeleton3d import SkeletonMotion,SkeletonState,SkeletonTree
import torch
def fix_root(motion:SkeletonMotion):
    new_root_translation = motion.root_translation.clone()
    new_root_translation[:,0] = 0
    new_state = SkeletonState.from_rotation_and_root_translation(
        motion.skeleton_tree, motion.local_rotation, new_root_translation, is_local=True)
    return SkeletonMotion.from_skeleton_state(new_state, motion.fps)


def move_feet_on_the_ground(motion):
    min_h = torch.min(motion.root_translation[:, 2])
    new_root_translation = motion.root_translation.clone()
    new_root_translation[:, 2] -= min_h

    new_state = SkeletonState.from_rotation_and_root_translation(
        motion.skeleton_tree, motion.local_rotation, new_root_translation, is_local=True)
    return SkeletonMotion.from_skeleton_state(new_state, motion.fps)

def rescale_motion_to_standard_size(motion_global_translation, standard_skeleton:SkeletonTree):
    rescaled_motion_global_translation = motion_global_translation.clone()
    limit = []
    for joint_idx,parent_idx in enumerate(standard_skeleton.parent_indices):
        if parent_idx == -1:
            pass
        else:
            scale =  torch.linalg.norm(motion_global_translation[:,joint_idx,:]-motion_global_translation[:,parent_idx,:],dim=1)/ \
                     torch.linalg.norm(standard_skeleton.local_translation[joint_idx,:],dim=0)
            rescaled_motion_global_translation[:,joint_idx,:] = rescaled_motion_global_translation[:,parent_idx,:] + \
                (motion_global_translation[:,joint_idx,:]-motion_global_translation[:,parent_idx,:])/scale.unsqueeze(1).repeat(1,3)
            # limit.append(scale.max())
            # limit.append(scale.min())
    return rescaled_motion_global_translation

def fix_joints(motion, joint_indices:list):
    motion_length,_,_ = motion.root_translation.shape
    new_root_translation = motion.root_translation.clone()
    new_local_rotation = motion.local_rotation.clone()
    new_local_rotation[:, joint_indices] = torch.Tensor([[0,0,0,1]]*motion_length)
    new_state = SkeletonState.from_rotation_and_root_translation(
        motion.skeleton_tree, motion.local_rotation, new_root_translation, is_local=True)
    return SkeletonMotion.from_skeleton_state(new_state, motion.fps)



