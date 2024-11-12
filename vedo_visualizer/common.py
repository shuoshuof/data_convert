# -*- coding: utf-8 -*-
"""
@Time ： 2024/11/11 14:54
@Auth ： shuoshuof
@File ：common.py
@Project ：data_convert
"""
from typing import List, Union

from motion_convert.utils.motion_process import motion_concatenate

from poselib.poselib.skeleton.skeleton3d import SkeletonMotion,SkeletonState
from poselib.poselib.visualization.common import plot_skeleton_H

from vedo_visualizer.base_visualizer import RobotVisualizer
from vedo_visualizer.vedo_robot import HuVedoRobot, VedoRobot



class SkMotionVisualizer(RobotVisualizer):
    def __init__(self, num_subplots:int, robot,data:List[Union[SkeletonMotion,SkeletonState]]) -> None:
        super().__init__(num_subplots,robot,data)

    def update_robot(self):

        for i in range(self.num_subplots):
            if self.counter>=len(self.data[i]):
                continue
            parent_indices = self.data[i].skeleton_tree.parent_indices.clone()
            parent_indices[0]=0
            link_global_rotations = self.data[i].global_rotation[self.counter]
            link_global_translations = self.data[i].global_translation[self.counter]
            self.robots[i].robot_transform(link_global_translations=link_global_translations,
                                           link_global_rotations=link_global_rotations)

def hu_v2_to_hu_v5(motion:SkeletonMotion):
    v2tov5_indices = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                      29, 30, 31, 32]
    with open('asset/hu_pose/hu_v5_zero_pose.pkl', 'rb') as f:
        hu_v5_zero_pose:SkeletonState = pickle.load(f)
    new_local_rotations = motion.local_rotation.clone()[:,v2tov5_indices]
    new_root_translations = motion.root_translation.clone()
    new_root_translations -= new_root_translations[0,...].clone()
    new_sk_state = SkeletonState.from_rotation_and_root_translation(
        hu_v5_zero_pose.skeleton_tree,
        new_local_rotations,
        new_root_translations,
        is_local=True
    )
    return SkeletonMotion.from_skeleton_state(new_sk_state,fps=motion.fps)

def hu_zero_motion():
    with open('asset/hu_pose/hu_v5_zero_pose.pkl', 'rb') as f:
        hu_v5_zero_pose:SkeletonState = pickle.load(f)

    new_local_rotation = hu_v5_zero_pose.local_rotation.unsqueeze(0).repeat(100,1,1)
    new_root_translation = hu_v5_zero_pose.root_translation.unsqueeze(0).repeat(100,1)

    new_sk_state = SkeletonState.from_rotation_and_root_translation(
        hu_v5_zero_pose.skeleton_tree,
        new_local_rotation,
        new_root_translation,
        is_local=True
    )
    return new_sk_state

def vis_sk_motion(motions:List[Union[SkeletonMotion,SkeletonState]]):
    robot = HuVedoRobot()

    # new_motions = [hu_v2_to_hu_v5(motion) for motion in motions]
    # new_motions = [hu_zero_motion() ]

    vis = SkMotionVisualizer(len(motions),robot,new_motions)
    vis.show()



if __name__ == '__main__':
    import pickle
    with open('motion_data/11_7_walk/hu_motion/walk_small_step2_11_07_22_motion.pkl','rb') as f:
        motion = pickle.load(f)

    vis_sk_motion([motion])




