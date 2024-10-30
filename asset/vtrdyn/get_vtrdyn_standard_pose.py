from bvh import Bvh
import numpy as np
import pickle
import copy
from collections import OrderedDict

from poselib.poselib.skeleton.skeleton3d import SkeletonTree,SkeletonMotion,SkeletonState
from poselib.poselib.visualization.common import plot_skeleton_H

from motion_convert.robot_config.VTRDYN import VTRDYN_JOINT_NAMES,vtrdyn_parent_indices,vtrdyn_graph
from motion_convert.robot_config.VTRDYN import VTRDYN_JOINT_NAMES_LITE,VTRDYN_CONNECTIONS_LITE,vtrdyn_lite_graph

from motion_convert.utils.transform3d import coord_transform

if __name__ == "__main__":

    with open('asset/vtrdyn/vtrdyn_t_pose.bvh') as f:
        vtrdyn_data = Bvh(f.read())

    t_pose_local_translation = []
    for joint_name in VTRDYN_JOINT_NAMES:
        t_pose_local_translation.append(vtrdyn_data.joint_offset(joint_name))
    t_pose_local_translation = np.array(t_pose_local_translation)/100

    t_pose_local_translation = coord_transform(t_pose_local_translation,order=[2,0,1],dir=np.array([1,-1,1]))

    vtrdyn_sk_tree = SkeletonTree.from_dict(
        {'node_names': np.array(VTRDYN_JOINT_NAMES),
         'parent_indices':{'arr': np.array(vtrdyn_parent_indices), 'context': {'dtype': 'int64'}},
         'local_translation':{'arr': t_pose_local_translation.copy(), 'context': {'dtype': 'float32'}}}
    )


    vtrdyn_t_pose = SkeletonState.zero_pose(vtrdyn_sk_tree)


    with open('asset/t_pose/vtrdyn_t_pose.pkl', 'wb') as f:
        pickle.dump(vtrdyn_t_pose, f)

    zero_pose_local_translation = copy.deepcopy(t_pose_local_translation)

    zero_pose_local_translation[15,:] = zero_pose_local_translation[15,[0,2,1]]
    zero_pose_local_translation[19,:] = -zero_pose_local_translation[19,[0,2,1]]
    zero_pose_local_translation[16,:] = -zero_pose_local_translation[16,[1,0,2,]]
    zero_pose_local_translation[20,:] = zero_pose_local_translation[20,[1,0,2,]]

    zero_pose_vtrdyn_sk_tree = SkeletonTree.from_dict(
        {'node_names': np.array(VTRDYN_JOINT_NAMES),
         'parent_indices':{'arr': np.array(vtrdyn_parent_indices), 'context': {'dtype': 'int64'}},
         'local_translation':{'arr': zero_pose_local_translation.copy(), 'context': {'dtype': 'float32'}}}
    )

    vtrdyn_zero_pose = SkeletonState.zero_pose(zero_pose_vtrdyn_sk_tree)

    vtrdyn_lite_sk_tree = copy.deepcopy(vtrdyn_zero_pose).skeleton_tree.keep_nodes_by_names(VTRDYN_JOINT_NAMES_LITE)

    vtrdyn_lite_zero_pose = SkeletonState.zero_pose(vtrdyn_lite_sk_tree)




    # plot_skeleton_H([vtrdyn_t_pose,vtrdyn_zero_pose,vtrdyn_lite_zero_pose])
    from body_visualizer.visualizer import BodyVisualizer
    bd_vis = BodyVisualizer(vtrdyn_lite_graph)
    pos = vtrdyn_lite_zero_pose.global_translation

    bd_vis.step(pos)

    with open('asset/zero_pose/vtrdyn_zero_pose.pkl', 'wb') as f:
        pickle.dump(vtrdyn_zero_pose, f)
    with open('asset/zero_pose/vtrdyn_lite_zero_pose.pkl', 'wb') as f:
        pickle.dump(vtrdyn_lite_zero_pose, f)






