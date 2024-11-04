from poselib.poselib.skeleton.skeleton3d import SkeletonMotion,SkeletonState,SkeletonTree

def motion2isaac(motion:SkeletonMotion):
    motion_dict = {}
    motion_dict['pose_quat_global'] = motion.global_rotation.numpy()
    motion_dict['pose_quat'] = motion.local_rotation.numpy()
    motion_dict['trans_orig'] = None
    motion_dict['root_trans_offset'] = motion.root_translation.numpy()
    motion_dict['target'] = "hu"
    motion_dict['pose_aa'] = None
    motion_dict['fps'] = motion.fps
    motion_dict['project_loss'] = 1e-5
    return motion_dict