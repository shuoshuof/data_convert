from bvh import Bvh
import numpy as np
import pickle
import copy

from poselib.poselib.skeleton.skeleton3d import SkeletonTree,SkeletonMotion,SkeletonState
from poselib.poselib.visualization.common import plot_skeleton_H

from motion_convert.robot_config.NOITOM import *
from motion_convert.utils.transform3d import coord_transform

if __name__ == "__main__":

    with open('asset/noitom/noitom_t_pose.bvh') as f:
        noitom_data = Bvh(f.read())

    t_pose_local_translation = []
    for joint_name in NOITOM_JOINT_NAMES:
        t_pose_local_translation.append(noitom_data.joint_offset(joint_name))
    t_pose_local_translation = np.array(t_pose_local_translation)/100

    t_pose_local_translation = coord_transform(t_pose_local_translation,order=[2,0,1],dir=np.array([1,1,1]))

    noitom_sk_tree = SkeletonTree.from_dict(
        {'node_names': np.array(NOITOM_JOINT_NAMES),
         'parent_indices':{'arr': np.array(noitom_parent_indices), 'context': {'dtype': 'int64'}},
         'local_translation':{'arr': t_pose_local_translation.copy(), 'context': {'dtype': 'float32'}}}
    )


    noitom_t_pose = SkeletonState.zero_pose(noitom_sk_tree)

    with open('asset/t_pose/noitom_t_pose.pkl', 'wb') as f:
        pickle.dump(noitom_t_pose, f)

    zero_pose_local_translation = copy.deepcopy(t_pose_local_translation)

    zero_pose_local_translation[15,:] = zero_pose_local_translation[15,[0,2,1]]
    zero_pose_local_translation[19,:] = -zero_pose_local_translation[19,[0,2,1]]
    zero_pose_local_translation[16,:] = -zero_pose_local_translation[16,[1,0,2,]]
    zero_pose_local_translation[20,:] = zero_pose_local_translation[20,[1,0,2,]]

    zero_pose_noitom_sk_tree = SkeletonTree.from_dict(
        {'node_names': np.array(NOITOM_JOINT_NAMES),
         'parent_indices':{'arr': np.array(noitom_parent_indices), 'context': {'dtype': 'int64'}},
         'local_translation':{'arr': zero_pose_local_translation.copy(), 'context': {'dtype': 'float32'}}}
    )

    noitom_zero_pose = SkeletonState.zero_pose(zero_pose_noitom_sk_tree)



    plot_skeleton_H([noitom_t_pose,noitom_zero_pose])
    # from body_visualizer.body_visualizer import BodyVisualizer
    # bd_vis = BodyVisualizer(noitom_graph)
    # pos = noitom_zero_pose.global_translation
    #
    # for p in pos:
    #     bd_vis.step(pos)
    with open('asset/zero_pose/noitom_zero_pose.pkl', 'wb') as f:
        pickle.dump(noitom_zero_pose, f)



