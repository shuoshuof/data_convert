import copy
import os
from tqdm import tqdm
import math
import torch
from abc import ABC, abstractmethod
from typing import Optional, Union
import joblib
import numpy as np
import pickle
import pandas as pd

from body_visualizer.visualizer import BodyVisualizer
from poselib.poselib.skeleton.skeleton3d import SkeletonTree,SkeletonMotion,SkeletonState
from poselib.poselib.visualization.common import plot_skeleton_H

from motion_convert.pipeline.base_pipeline import BasePipeline,PipelineArgs
from motion_convert.utils.transform3d import *
from motion_convert.retarget_optimizer.hu_retarget_optimizer import HuRetargetOptimizer
from motion_convert.utils.motion_process import MotionProcessManager,rescale_motion_to_standard_size,get_mirror_motion

from motion_convert.robot_config.Hu import VTRDYN_LITE2HU_JOINT_MAPPING,hu_graph
from motion_convert.robot_config.VTRDYN import vtrdyn_lite_graph,VTRDYN_CONNECTIONS_LITE,VTRDYN_JOINT_NAMES_LITE

from motion_convert.format_convert.convert import motion2isaac

def get_motion_translation(data):
    motion_length = len(data)
    motion_global_translation = np.zeros((motion_length, len(VTRDYN_JOINT_NAMES_LITE), 3))
    for joint_idx, joint_name in enumerate(VTRDYN_JOINT_NAMES_LITE):
        motion_global_translation[:, joint_idx, 0] = data[f'{joint_name} position X(m)']
        motion_global_translation[:, joint_idx, 1] = data[f'{joint_name} position Y(m)']
        motion_global_translation[:, joint_idx, 2] = data[f'{joint_name} position Z(m)']
    return motion_global_translation

class ConvertVtrdynLitePipeline(BasePipeline):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.vtrdyn_zero_pose,self.hu_zero_pose,self.hu_skeleton_tree = self._load_asset()
    def _read_data(self, **kwargs) -> Optional[list]:
        motion_paths = [os.path.join(self.motion_dir, f) for f in os.listdir(self.motion_dir) if
                              os.path.isfile(os.path.join(self.motion_dir, f))]
        return motion_paths
    def _split_data(self,data,**kwargs):
        return np.array_split(data,self.num_processes)

    def _load_asset(self):
        with open('asset/zero_pose/vtrdyn_lite_zero_pose.pkl','rb') as f:
            vtrdyn_zero_pose: SkeletonState = pickle.load(f)
        with open('asset/zero_pose/hu_zero_pose.pkl','rb') as f:
            hu_zero_pose: SkeletonState = pickle.load(f)
        hu_skeleton_tree = hu_zero_pose.skeleton_tree
        return vtrdyn_zero_pose,hu_zero_pose,hu_skeleton_tree

    def _rebuild_with_vtrdyn_lite_zero_pose(self,motion_global_translation,fps=30):
        motion_length, num_joint, _ = motion_global_translation.shape

        rebuilt_motion_root_translation = motion_global_translation[:,0,:].clone()
        rebuilt_motion_global_rotation = torch.Tensor([[[0,0,0,1]]]).repeat(motion_length,num_joint,1)

        zero_pose_local_translation = \
            self.vtrdyn_zero_pose.local_translation.repeat(motion_length, 1, 1)
        parent_indices = self.vtrdyn_zero_pose.skeleton_tree.parent_indices

        joint0_quat = cal_joint_quat(
            zero_pose_local_translation[:,[4,1,7]],
            motion_global_translation[:,[4,1,7]]-motion_global_translation[:,[0]]
        )

        joint8_quat = cal_joint_quat(
            zero_pose_local_translation[:,[11,10,15]],
            motion_global_translation[:,[11,10,15]]-motion_global_translation[:,[8]]
        )
        rebuilt_motion_global_rotation[:,[0,8],:] = \
            torch.concatenate([joint0_quat.unsqueeze(1),joint8_quat.unsqueeze(1)],dim=1)

        for joint_idx,parent_idx in enumerate(parent_indices):
            if joint_idx == 0 or parent_idx == 0 or parent_idx==8:
                continue
            else:
                rebuilt_motion_global_rotation[:,parent_idx,:] = quat_between_two_vecs(
                    zero_pose_local_translation[:,joint_idx],
                    motion_global_translation[:,joint_idx]-motion_global_translation[:,parent_idx]
                )
        rebuilt_skeleton_state = SkeletonState.from_rotation_and_root_translation(
            self.vtrdyn_zero_pose.skeleton_tree,
            rebuilt_motion_global_rotation,
            rebuilt_motion_root_translation,
            is_local=False
        )
        rebuilt_motion = SkeletonMotion.from_skeleton_state(rebuilt_skeleton_state,fps=fps)

        rebuilt_error = torch.abs(rebuilt_motion.global_translation-motion_global_translation).max()
        # print(f"Rebuild error :{rebuild_error:.5f}")

        return rebuilt_motion,round(float(rebuilt_error),4)


    def _process_data(self,data_chunk,results,process_idx,debug,**kwargs):
        hu_retarget_optimizer = HuRetargetOptimizer(self.hu_skeleton_tree,kwargs.get('clip_angle',True))
        process_manager = MotionProcessManager(**kwargs)
        for path  in data_chunk:
            file_name = os.path.basename(path).split('.')[0]
            mocap_data = pd.read_csv(path)
            motion_global_translation = get_motion_translation(mocap_data)
            motion_global_translation = torch.Tensor(motion_global_translation)
            motion_global_translation = coord_transform(motion_global_translation,dir=torch.Tensor([-1,-1,1]))
            # vis_vtrdyn(motion_global_translation)
            motion_global_translation = rescale_motion_to_standard_size(motion_global_translation, self.vtrdyn_zero_pose.skeleton_tree)
            # vis_vtrdyn(motion_global_translation)
            fps = kwargs.get('fps', 30)
            rebuilt_motion,rebuilt_error = self._rebuild_with_vtrdyn_lite_zero_pose(motion_global_translation,fps)

            # vis_vtrdyn(rebuilt_motion.global_translation)

            target_motion = copy.deepcopy(rebuilt_motion).retarget_to_by_tpose(
                joint_mapping=VTRDYN_LITE2HU_JOINT_MAPPING,
                source_tpose=self.vtrdyn_zero_pose,
                target_tpose=self.hu_zero_pose,
                rotation_to_target_skeleton=torch.Tensor([0, 0, 0, 1]),
                scale_to_target_skeleton=1.,
            )
            # vis_hu(target_motion.global_translation)

            max_epoch = kwargs.get('max_epoch',500)
            lr = kwargs.get('lr',1e-1)
            retargeted_motion,retargeted_error = hu_retarget_optimizer.train(
                motion_data=target_motion,
                max_epoch=max_epoch,
                lr=lr,
                process_idx=process_idx
            )
            # vis_hu(retargeted_motion.global_translation)
            # vis_new_hu(retargeted_motion)
            # plot_skeleton_H([rebuilt_motion,target_motion,retargeted_motion])
            result_motion = process_manager.process_motion(retargeted_motion, **kwargs)
            # vis_hu(retargeted_motion.global_translation)

            motion_list = []
            motion_list.append([file_name,result_motion])
            if kwargs.get('generate_mirror', False):
                motion_list.append([file_name+'_mirror',get_mirror_motion(result_motion)])

            motion_save_dir = self.save_dir+'_motion'
            os.makedirs(motion_save_dir,exist_ok=True)
            for name, motion in motion_list:
                motion_dict = motion2isaac(motion)
                save_dict  = {name:motion_dict}
                with open(f'{self.save_dir}/{name}.pkl', 'wb') as f:
                    joblib.dump(save_dict,f)
                with open(f'{motion_save_dir}/{name}_motion.pkl', 'wb') as f:
                    joblib.dump(motion,f)

            info = {'file_name':file_name,'rebuilt_error':rebuilt_error,'retargeted_error':retargeted_error}

            results.append(info)


def vis_vtrdyn(motion_global_translation):
    bd_vis = BodyVisualizer(vtrdyn_lite_graph, static_frame=False)
    motion_global_translation = motion_global_translation.reshape(-1,19,3)
    for global_trans in motion_global_translation:
        bd_vis.step(global_trans)
def vis_hu(motion_global_translation):
    bd_vis = BodyVisualizer(hu_graph,static_frame=False)
    motion_global_translation = motion_global_translation.reshape(-1,33,3)
    for global_trans in motion_global_translation:
        bd_vis.step(global_trans)
def vis_new_hu(motion:SkeletonMotion):
    with open('asset/zero_pose/new_hu_zero_pose.pkl','rb') as f:
        new_zero_pose:SkeletonState = pickle.load(f)
    new_local_rotation = motion.local_rotation.clone()
    motion_length =new_local_rotation.shape[0]
    new_local_rotation[:,[5,6,11,12],:] = torch.Tensor([[[0,0,0,1]]]).repeat(motion_length,4,1)
    new_sk_state = SkeletonState.from_rotation_and_root_translation(
        new_zero_pose.skeleton_tree,
        new_local_rotation,
        motion.root_translation,
        is_local=True
    )
    from body_visualizer.common import vis_motion
    new_motion = SkeletonMotion.from_skeleton_state(new_sk_state,fps=motion.fps)
    vis_motion(new_motion,graph='hu',static_frame=False)

if __name__ == '__main__':
    vtrdyn_lite_pipeline = ConvertVtrdynLitePipeline(motion_dir='motion_data/10_29_vtrdyn_mocap/mocap',
                                                     save_dir='motion_data/10_29_vtrdyn_mocap/hu')
    args = PipelineArgs(
        max_epoch=500,
        filter=True,
        fix_joints=True,
        joint_indices=[i for i in range(13,33)],
        fix_ankles=True,
        zero_root=True,
        clip_angle=True,
        height_adjustment=False,
        move_to_ground=True,
        generate_mirror=True,
        save_info=True,
    )
    vtrdyn_lite_pipeline.run(
        debug=False,
        **args
    )

