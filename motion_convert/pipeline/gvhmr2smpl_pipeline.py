import os
from tqdm import tqdm
import math
import torch
from abc import ABC, abstractmethod
from typing import Optional, Union
import joblib
import numpy as np
from scipy.spatial.transform import Rotation as sRot
from poselib.poselib.core.rotation3d import quat_mul,quat_rotate

from motion_convert.pipeline.base_pipeline import BasePipeline
from motion_convert.format_convert.smpl2isaac import convert2isaac

class ConvertGVHMRPipeline(BasePipeline):
    def __init__(self,motion_dir:str,save_dir:str,num_processes:int=None):
        super().__init__(motion_dir,save_dir,num_processes)

    def _read_data(self, **kwargs) -> Optional[list]:
        motion_paths = [os.path.join(self.motion_dir, f) for f in os.listdir(self.motion_dir) if
                              os.path.isfile(os.path.join(self.motion_dir, f))]
        return motion_paths

    def _split_data(self,data,**kwargs):
        return np.array_split(data,self.num_processes)

    def coord_transform(self,global_orient, transl):
        global_orient = sRot.from_rotvec(global_orient.reshape(-1, 3)).as_quat()
        global_orient = torch.Tensor(global_orient)

        # isaac quat (x,y,z,w)
        theta = math.pi / 2
        v_x = torch.tensor([1, 0, 0, 1])
        quat_rotation_x = torch.tensor(
            [math.sin(theta / 2), math.sin(theta / 2), math.sin(theta / 2), math.cos(theta / 2)]) * v_x

        global_orient = quat_mul(quat_rotation_x, global_orient)
        transl = quat_rotate(quat_rotation_x, transl)

        theta = -math.pi / 2
        v_z = torch.tensor([0, 0, 1, 1])
        quat_rotation_z = torch.tensor(
            [math.sin(theta / 2), math.sin(theta / 2), math.sin (theta / 2), math.cos(theta / 2)]) * v_z

        global_orient = quat_mul(quat_rotation_z, global_orient)
        transl = quat_rotate(quat_rotation_z, transl)

        global_orient = sRot.from_quat(global_orient).as_rotvec()
        global_orient = torch.Tensor(global_orient)
        return global_orient, transl

    def _process_data(self,data_chunk,results,process_idx,debug,**kwargs):
        for path  in data_chunk:
            motion_data = torch.load(path)
            body_pose = motion_data['smpl_params_global']['body_pose']
            global_orient = motion_data['smpl_params_global']['global_orient']
            transl = motion_data['smpl_params_global']['transl']

            add_body_pose = torch.zeros((body_pose.shape[0], 6))
            body_pose = torch.cat((body_pose, add_body_pose), dim=1)

            global_orientation,transl = self.coord_transform(global_orient, transl)

            transformed_dict = {
                'pose_aa': torch.concatenate((global_orientation, body_pose), dim=1),
                'beta': torch.zeros(10),
                'transl': transl,
                'gender': 'neutral',
                'fps': 20
            }

            data = convert2isaac(transformed_dict)

            data_dict = {}
            file_name = os.path.basename(path).split('.')[0]
            data_dict[file_name] = data

            with open(f'{self.save_dir}/{file_name}.pkl', 'wb') as f:
                joblib.dump(data_dict,f)


if __name__ == '__main__':
    pipeline = ConvertGVHMRPipeline(motion_dir='motion_data/10_25/cam_out',
                                    save_dir='motion_data/10_25/smpl',)
    pipeline.run(debug=False)
