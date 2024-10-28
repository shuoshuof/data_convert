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

from motion_convert.pipeline.base_pipeline import BasePipeline
from motion_convert.utils.transform3d import *
from motion_convert.retarget_optimizer.hu_retarget_optimizer import HuRetargetOptimizer
from motion_convert.utils.motion_process import  rescale_motion_to_standard_size, MotionProcessManager

from motion_convert.robot_config.Hu import VTRDYN2HU_JOINT_MAPPING,hu_graph
from motion_convert.robot_config.VTRDYN import vtrdyn_graph,VTRDYN_JOINT_NAMES

# from scripts.process_vtrdyn_mocap import *
def get_motion_translation(data):
    motion_length = len(data)
    motion_global_translation = np.zeros((motion_length, len(VTRDYN_JOINT_NAMES), 3))
    for joint_idx, joint_name in enumerate(VTRDYN_JOINT_NAMES):
        motion_global_translation[:, joint_idx, 0] = data[f'{joint_name} position X(m)']
        motion_global_translation[:, joint_idx, 1] = data[f'{joint_name} position Y(m)']
        motion_global_translation[:, joint_idx, 2] = data[f'{joint_name} position Z(m)']
    return motion_global_translation

class ConvertVtrdynPipeline(BasePipeline):
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
        with open('asset/zero_pose/vtrdyn_zero_pose.pkl','rb') as f:
            vtrdyn_zero_pose: SkeletonState = pickle.load(f)
        with open('asset/zero_pose/hu_zero_pose.pkl','rb') as f:
            hu_zero_pose: SkeletonState = pickle.load(f)
        hu_skeleton_tree = hu_zero_pose.skeleton_tree
        return vtrdyn_zero_pose,hu_zero_pose,hu_skeleton_tree
    def _rebuild_with_vtrdyn_zero_pose(self,motion_global_translation,fps=30):
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

        joint10_quat = cal_joint_quat(
            zero_pose_local_translation[:,[17,13,11]],
            motion_global_translation[:,[17,13,11]]-motion_global_translation[:,[10]]
        )
        rebuilt_motion_global_rotation[:,[0,10],:] = \
            torch.concatenate([joint0_quat.unsqueeze(1),joint10_quat.unsqueeze(1)],dim=1)
        # joint8_quat = cal_joint_quat(
        #     zero_pose_local_translation[:,[17,13,11]],
        #     motion_global_translation[:,[17,13,11]]-motion_global_translation[:,[10]]
        # )
        # rebuilt_motion_global_rotation[:,[0,8],:] = \
        #     torch.concatenate([joint0_quat.unsqueeze(1),joint8_quat.unsqueeze(1)],dim=1)

        for joint_idx,parent_idx in enumerate(parent_indices):
            # if joint_idx == 0 or parent_idx == 0 or parent_idx==8:
            if joint_idx == 0 or parent_idx == 0 or parent_idx==10:
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

        rebuild_error = torch.abs(rebuilt_motion.global_translation-motion_global_translation).max()
        print(f"Rebuild error :{rebuild_error:.5f}")

        return rebuilt_motion


    def _process_data(self,data_chunk,results,process_idx,debug,**kwargs):
        hu_retarget_optimizer = HuRetargetOptimizer(self.hu_skeleton_tree)
        process_manager = MotionProcessManager()
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
            rebuilt_motion = self._rebuild_with_vtrdyn_zero_pose(motion_global_translation,fps)

            # vis_vtrdyn(rebuilt_motion.global_translation)

            target_motion = copy.deepcopy(rebuilt_motion).retarget_to_by_tpose(
                joint_mapping=VTRDYN2HU_JOINT_MAPPING,
                source_tpose=self.vtrdyn_zero_pose,
                target_tpose=self.hu_zero_pose,
                rotation_to_target_skeleton=torch.Tensor([0, 0, 0, 1]),
                scale_to_target_skeleton=1.,
            )
            # vis_hu(target_motion.global_translation)

            max_epoch = kwargs.get('max_epoch',500)
            lr = kwargs.get('lr',1e-1)
            retargeted_motion = hu_retarget_optimizer.train(
                motion_data=target_motion,
                max_epoch=max_epoch,
                lr=lr,
                process_idx=process_idx
            )
            vis_hu(retargeted_motion.global_translation)
            # plot_skeleton_H([rebuilt_motion,target_motion,retargeted_motion])
            result_motion = process_manager.process_motion(retargeted_motion, **kwargs)


            motion_dict = {}
            motion_dict['pose_quat_global'] = result_motion.global_rotation.numpy()
            motion_dict['pose_quat'] = result_motion.local_rotation.numpy()
            motion_dict['trans_orig'] = None
            motion_dict['root_trans_offset'] = result_motion.root_translation.numpy()
            motion_dict['target'] = "hu"
            motion_dict['pose_aa'] = None
            motion_dict['fps'] = result_motion.fps
            motion_dict['project_loss'] = 1e-5

            save_dict  = {file_name:motion_dict}
            with open(f'{self.save_dir}/{file_name}.pkl', 'wb') as f:
                joblib.dump(save_dict,f)
            with open(f'{self.save_dir}/{file_name}_motion.pkl', 'wb') as f:
                joblib.dump(result_motion,f)


def vis_vtrdyn(motion_global_translation):
    bd_vis = BodyVisualizer(vtrdyn_graph,static_frame=False)
    motion_global_translation = motion_global_translation.reshape(-1,21,3)
    for global_trans in motion_global_translation:
        bd_vis.step(global_trans)
def vis_hu(motion_global_translation):
    bd_vis = BodyVisualizer(hu_graph,static_frame=False)
    motion_global_translation = motion_global_translation.reshape(-1,33,3)
    for global_trans in motion_global_translation:
        bd_vis.step(global_trans)

if __name__ == '__main__':
    vtrdyn_pipeline = ConvertVtrdynPipeline(motion_dir='motion_data/10_28/mocap',
                                            save_dir='motion_data/10_28/hu',)
    vtrdyn_pipeline.run(debug=False,max_epoch=400,filter=False,fix_joints=True)
